"""Offline behavioral regression tests using only harmless synthetic responses.

The prompt-generating methods are stubbed: these tests exercise bookkeeping,
termination, dependency injection and serialization, not prompt effectiveness.
"""
import json
import unittest
from unittest.mock import Mock, patch

from crescendo.attack import CrescendoAttack
from crescendo.config import CrescendoConfig
from indiana_jones.attack import (
    AttemptResult, IndianaJonesAttack, IndianaJonesResult, RoundResult,
)
from indiana_jones.config import IndianaJonesConfig


SAFE = {'safe': True, 'category': None, 'raw': 'safe', 'method': 'mock'}
FLAGGED = {'safe': False, 'category': 'synthetic', 'raw': 'unsafe', 'method': 'mock'}


class CrescendoWorkflowTests(unittest.TestCase):
    def make_attack(self, *, turns=3, responses=None, judgments=None):
        client = Mock()
        judge = Mock()
        judge.judge.side_effect = judgments or [SAFE] * turns
        attack = CrescendoAttack(CrescendoConfig(max_turns=turns), client=client, judge=judge)
        attack._generate_initial_prompt = Mock(return_value='What is a rainbow?')
        attack._generate_next_prompt = Mock(return_value=(-1, 'What causes rain?'))
        attack._generate_recovery_prompt = Mock(return_value='What is a cloud?')
        attack._generate_redirect_prompt = Mock(return_value='What is sunshine?')
        attack._query_target = Mock(side_effect=responses or ['A rainbow is colorful.'] * turns)
        return attack, judge

    def test_stops_at_max_turns_and_records_metadata(self):
        attack, judge = self.make_attack(turns=2)
        result = attack.run('Explain weather', behavior_idx=7)
        self.assertFalse(result.success)
        self.assertEqual(result.num_turns, 2)
        self.assertEqual([t.turn_number for t in result.turns], [1, 2])
        self.assertEqual(result.behavior_idx, 7)
        self.assertIsNone(result.jailbreak_turn)
        self.assertEqual(judge.judge.call_count, 2)
        self.assertEqual(len(json.loads(result.to_json())['turns']), 2)

    def test_stops_on_flagged_mock_judgment(self):
        attack, _ = self.make_attack(turns=4, judgments=[FLAGGED])
        result = attack.run('Explain weather')
        self.assertTrue(result.success)
        self.assertEqual(result.num_turns, 1)
        self.assertEqual(result.jailbreak_turn, 1)
        attack._generate_next_prompt.assert_not_called()

    def test_refusal_limit_and_complete_audit_log(self):
        attack, _ = self.make_attack(turns=5, responses=['I cannot do that'] * 3)
        result = attack.run('Explain weather')
        self.assertFalse(result.success)
        self.assertEqual(result.num_turns, 3)
        self.assertTrue(all(t.is_refusal for t in result.turns))
        self.assertEqual(attack._generate_recovery_prompt.call_count, 2)
        self.assertEqual(attack._query_target.call_count, 3)

    def test_target_query_exception_propagates(self):
        attack, _ = self.make_attack(turns=2)
        attack._query_target.side_effect = TimeoutError('simulated timeout')
        with self.assertRaisesRegex(TimeoutError, 'simulated timeout'):
            attack.run('Explain weather')


class IndianaWorkflowTests(unittest.TestCase):
    def make_attack(self, *, attempts=3, turns=2, judgments=None, responses=None):
        client = Mock()
        judge = Mock()
        judge.judge.side_effect = judgments or [SAFE] * (attempts * turns)
        attack = IndianaJonesAttack(
            IndianaJonesConfig(max_attempts=attempts, max_turns=turns),
            client=client, judge=judge,
        )
        attack._extract_keyword = Mock(side_effect=lambda behavior, number, previous: f'weather{number}')
        attack._inspector_fact_prompt = Mock(return_value='Name three cloud types.')
        attack._inspector_check = Mock(return_value='yes')
        attack._attacker_round2 = Mock(return_value='How do clouds form?')
        attack._attacker_deepen = Mock(return_value='What causes precipitation?')
        attack._query_target = Mock(side_effect=responses or ['Cirrus, cumulus, stratus.'] * (attempts * turns))
        return attack, judge

    def test_exhausts_attempt_budget_and_records_rounds(self):
        attack, judge = self.make_attack(attempts=2, turns=1)
        result = attack.run('Explain clouds', behavior_idx=3)
        self.assertFalse(result.success)
        self.assertEqual(result.num_attempts, 2)
        self.assertEqual([a.keyword for a in result.attempts], ['weather1', 'weather2'])
        self.assertEqual([len(a.rounds) for a in result.attempts], [1, 1])
        self.assertEqual(result.behavior_idx, 3)
        self.assertIsNone(result.jailbreak_attempt)
        self.assertEqual(judge.judge.call_count, 2)

    def test_stops_after_flagged_mock_judgment(self):
        attack, _ = self.make_attack(attempts=4, turns=2, judgments=[FLAGGED])
        result = attack.run('Explain clouds')
        self.assertTrue(result.success)
        self.assertEqual(result.successful_attempt, 1)
        self.assertEqual(result.num_attempts, 1)
        self.assertEqual(len(result.attempts[0].rounds), 1)
        self.assertEqual(json.loads(result.to_json())['jailbreak_attempt'], 1)

    def test_refusal_ends_round_and_next_attempt_starts_fresh(self):
        attack, _ = self.make_attack(attempts=2, turns=2, responses=['I cannot do that', 'Clouds are water droplets.', 'Rain falls.'])
        result = attack.run('Explain clouds')
        self.assertEqual([len(a.rounds) for a in result.attempts], [1, 2])
        self.assertTrue(result.attempts[0].rounds[0].is_refusal)
        self.assertEqual(attack._query_target.call_count, 3)

    def test_round_budget_and_serialization(self):
        attack, _ = self.make_attack(attempts=1, turns=3)
        result = attack.run('Explain clouds')
        self.assertEqual([r.round_number for r in result.attempts[0].rounds], [1, 2, 3])
        self.assertEqual(result.attempts[0].rounds[0].role, 'inspector_factfinding')
        self.assertEqual(result.attempts[0].rounds[1].role, 'attacker_escalation')
        self.assertEqual(len(json.loads(result.to_json())['attempts'][0]['rounds']), 3)

    def test_target_timeout_propagates(self):
        attack, _ = self.make_attack(attempts=1, turns=1)
        attack._query_target.side_effect = TimeoutError('simulated timeout')
        with self.assertRaisesRegex(TimeoutError, 'simulated timeout'):
            attack.run('Explain clouds')

    def test_result_property_uses_first_flagged_attempt(self):
        round_result = RoundResult(1, 'inspector_factfinding', 'question', 'answer', None, False, True, None, 'mock')
        result = IndianaJonesResult('example', 1, 'target', 'orchestrator', True, 2, 2, [
            AttemptResult(1, 'first', [round_result], False),
            AttemptResult(2, 'second', [round_result], True),
        ])
        self.assertEqual(result.jailbreak_attempt, 2)
        self.assertEqual(json.loads(result.to_json())['jailbreak_attempt'], 2)


if __name__ == '__main__':
    unittest.main()
