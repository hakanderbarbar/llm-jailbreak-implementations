import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from shared.chat import prepare_messages
from shared.prompts import PromptLibrary
from shared.run_utils import load_behaviors, parse_behavior_selection, save_result
from shared.models import AttackResult, TurnResult
from shared.judge import is_refusal, LlamaGuardJudge
from shared.ollama_client import OllamaClient
from crescendo.config import CrescendoConfig
from indiana_jones.config import IndianaJonesConfig
from crescendo.attack import CrescendoAttack, _parse_orchestrator_output
from indiana_jones.attack import IndianaJonesAttack, _parse_yes_no


class UtilityTests(TestCase):
    def test_selection(self):
        self.assertEqual(parse_behavior_selection('0,2-4,2', 5), [0, 2, 3, 4])
        self.assertEqual(parse_behavior_selection('all', 3), [0, 1, 2])
        for value in ['-1', '3', '2-1', 'bad']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_behavior_selection(value, 3)

    def test_csv(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('goal,target\nexample,answer\n', encoding='utf-8')
            self.assertEqual(load_behaviors(str(path)), [{'idx': 0, 'goal': 'example', 'target': 'answer'}])
            path.write_text('wrong,header\na,b\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_behaviors(str(path))

    def test_prompts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.json'
            path.write_text(json.dumps({'greeting': 'Hello {name}!'}), encoding='utf-8')
            prompts = PromptLibrary(path)
            self.assertEqual(prompts.render('greeting', name='Ada'), 'Hello Ada!')
            with self.assertRaises(ValueError):
                prompts.render('greeting')

    def test_messages_not_mutated(self):
        original = [{'role': 'user', 'content': 'hello'}]
        result = prepare_messages(original, 'system')
        self.assertEqual(len(result), 2)
        result[1]['content'] = 'changed'
        self.assertEqual(original[0]['content'], 'hello')
        self.assertEqual(len(prepare_messages(result, 'system')), 2)

    def test_results(self):
        result = AttackResult('example', 1, 'model', 'orchestrator', False, 1,
                              [TurnResult(1, 'prompt', 'response', False, False, None, 'safe')])
        self.assertIsNone(result.jailbreak_turn)
        with TemporaryDirectory() as directory:
            save_result(result, Path(directory))
            saved = json.loads((Path(directory) / 'behavior_001.json').read_text())
            self.assertEqual(saved['turns'][0]['response'], 'response')

    def test_parsers(self):
        self.assertEqual(_parse_orchestrator_output('ESCALATION_SCORE: 12\nHello'), (10, 'Hello'))
        self.assertEqual(_parse_orchestrator_output('Hello'), (-1, 'Hello'))
        self.assertEqual(_parse_yes_no('Yes, plausible'), 'yes')
        self.assertIsNone(_parse_yes_no('uncertain'))
        self.assertTrue(is_refusal('I cannot do that'))

    def test_ollama_mock(self):
        with patch('shared.ollama_client.requests.post') as post:
            post.return_value.json.return_value = {'message': {'content': 'ok'}}
            self.assertEqual(OllamaClient().chat('example', [{'role': 'user', 'content': 'hi'}], system='system'), 'ok')
            self.assertEqual(post.call_args.kwargs['json']['messages'][0]['role'], 'system')

    def test_judge_mock(self):
        judge = LlamaGuardJudge()
        judge.client = Mock()
        judge.client.chat.return_value = 'safe'
        self.assertTrue(judge.judge('hello')['safe'])
        judge.client.chat.return_value = 'unsafe\nS2'
        self.assertEqual(judge.judge('hello')['category'], 's2')

    def test_injected_clients(self):
        client, judge, target = Mock(), Mock(), Mock()
        for config, attack_cls in [(CrescendoConfig(use_black_box=True), CrescendoAttack),
                                   (IndianaJonesConfig(use_black_box=True), IndianaJonesAttack)]:
            attack = attack_cls(config, client=client, judge=judge, target=target)
            self.assertIs(attack.client, client)
            self.assertIs(attack.judge, judge)
            self.assertIs(attack.target, target)
