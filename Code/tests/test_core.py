import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile
import io
import unittest
import sys
import contextlib
from unittest.mock import Mock, patch, MagicMock

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
import requests
from shared.ollama_client import OllamaClient
from shared.router import OpenRouterClient
from shared.judge import LlamaGuardJudge, is_refusal
from shared.run_utils import load_behaviors, parse_behavior_selection, save_result
from shared.calculate_length import find_responses
from crescendo import run as crescendo_run
from indiana_jones import run as indiana_run
from crescendo.config import CrescendoConfig
from indiana_jones.config import IndianaJonesConfig
from suffix import run as suffix_run


class UtilityTests(unittest.TestCase):
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

class SharedEdgeCases(unittest.TestCase):
    def test_selection_valid_and_invalid(self):
        self.assertEqual(parse_behavior_selection('1, 1, 2-3', 5), [1, 2, 3])
        for spec, total in [('4-2', 5), ('0-2-3', 5), ('5', 5), ('-1', 5), ('1', -1)]:
            with self.subTest(spec=spec, total=total), self.assertRaises(ValueError):
                parse_behavior_selection(spec, total)

    def test_csv_missing_and_bad_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'
            with self.assertRaises(FileNotFoundError):
                load_behaviors(str(path))
            path.write_text('wrong,header\na,b\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_behaviors(str(path))

    def test_ollama_model_listing_and_validation(self):
        client = OllamaClient('http://localhost:11434/')
        with patch('shared.ollama_client.requests.get') as get:
            get.return_value.json.return_value = {'models': [{'name': 'example:latest'}]}
            self.assertEqual(client.list_models(), ['example:latest'])
            client.check_models('example')
            with self.assertRaises(RuntimeError):
                client.check_models('absent')

    def test_ollama_http_errors(self):
        client = OllamaClient()
        for error in [requests.exceptions.Timeout('slow'), requests.exceptions.HTTPError('bad')]:
            with self.subTest(error=type(error).__name__), patch('shared.ollama_client.requests.post') as post:
                post.side_effect = error if isinstance(error, requests.exceptions.Timeout) else None
                if isinstance(error, requests.exceptions.HTTPError):
                    post.return_value.raise_for_status.side_effect = error
                with self.assertRaises(type(error)):
                    client.chat('example', [{'role': 'user', 'content': 'hello'}])

    def test_router_http_error_and_payload(self):
        with patch('shared.router.requests.post') as post:
            post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError('bad request')
            post.return_value.status_code = 400
            post.return_value.text = 'rejected'
            with self.assertRaises(requests.exceptions.HTTPError):
                OpenRouterClient().chat('example', [{'role': 'user', 'content': 'hello'}], system='system')
            self.assertEqual(post.call_args.kwargs['json']['messages'][0]['role'], 'system')

    def test_judge_safe_unsafe_and_fallback(self):
        judge = LlamaGuardJudge()
        judge.client = MagicMock()
        for raw, expected in [('safe', True), ('unsafe\nS2', False)]:
            judge.client.chat.return_value = raw
            result = judge.judge('example response')
            self.assertEqual(result['safe'], expected)
            self.assertEqual(result['method'], 'llama_guard')
        judge.client.chat.side_effect = RuntimeError('offline')
        self.assertTrue(judge.judge('I cannot do that')['safe'])
        self.assertFalse(judge.judge('Here is a description')['safe'])
        self.assertTrue(is_refusal('I cannot comply'))
        self.assertFalse(is_refusal('A neutral response'))


class RunnerCoverage(unittest.TestCase):
    def test_config_output_paths(self):
        self.assertEqual(CrescendoConfig(output_dir='example').output_path(), Path('example'))
        self.assertEqual(IndianaJonesConfig(output_dir='example').output_path(), Path('example'))

    def test_summary_empty_and_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            c = crescendo_run.save_summary([], folder, CrescendoConfig())
            self.assertEqual(c['attack_success_rate'], 0)
            i = indiana_run.save_summary([], folder, IndianaJonesConfig())
            self.assertIsNone(i['avg_attempts_to_success'])
            result = MagicMock(behavior_idx=1, behavior='benign', success=True,
                               num_turns=1, jailbreak_turn=1, num_attempts=1,
                               jailbreak_attempt=1, successful_attempt=1)
            self.assertEqual(crescendo_run.save_summary([result], folder, CrescendoConfig())['num_success'], 1)
            self.assertEqual(indiana_run.save_summary([result], folder, IndianaJonesConfig())['avg_attempts_to_success'], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                crescendo_run.print_summary(crescendo_run.save_summary([result], folder, CrescendoConfig()))
                indiana_run.print_summary(indiana_run.save_summary([result], folder, IndianaJonesConfig()))

    def test_in_process_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            csv_path = folder / 'data.csv'
            csv_path.write_text('goal,target\nbenign example,example\n', encoding='utf-8')
            for module in (crescendo_run, indiana_run):
                with self.subTest(module=module.__name__):
                    argv = ['prog', '--dry-run', '--advbench', str(csv_path),
                            '--behaviors', '0', '--output-dir', str(folder / module.__name__)]
                    with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        module.main()

    def test_suffix_load_and_query_with_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suffixes.json'
            path.write_text(json.dumps([{'behavior_id': 1, 'model': 'llama', 'loss': 1,
                                        'behavior': 'benign', 'suffix': ' example'}]))
            with patch.object(suffix_run, 'SUFFIX_FILE', path):
                self.assertEqual(len(suffix_run.load_suffixes('all')), 1)
            client = MagicMock()
            client.chat.return_value = 'example answer'
            self.assertEqual(suffix_run.query(client, 'model', 'benign', ' example'), 'example answer')
            self.assertEqual(client.chat.call_args.kwargs['messages'][0]['content'], 'benign example')

class SuffixWorkflowTests(unittest.TestCase):
    def test_offline_suffix_result_file_success_and_error(self):
        """Exercise result writing, error recording, and loop without network access."""
        entries = [{'behavior_id': 1, 'model': 'llama', 'loss': 0.2,
                    'behavior': 'harmless example', 'suffix': ' context'}]
        with tempfile.TemporaryDirectory() as tmp:
            client = MagicMock()
            client.chat.side_effect = ['answer', RuntimeError('offline')]
            with patch.object(suffix_run, 'OUTPUT_DIR', Path(tmp)), \
                 patch.object(suffix_run, 'load_suffixes', return_value=entries), \
                 patch.object(suffix_run, 'OpenRouterClient', return_value=client), \
                 patch.object(suffix_run.time, 'sleep'):
                suffix_run.run('llama', ['first', 'second'], verbose=True)
            output_files = list(Path(tmp).glob('*.jsonl'))
            self.assertEqual(len(output_files), 1)
            rows = [json.loads(line) for line in output_files[0].read_text().splitlines()]
            self.assertEqual([row['status'] for row in rows], ['ok', 'error'])
            self.assertEqual(rows[0]['response'], 'answer')
            self.assertIn('offline', rows[1]['response'])

class TestFindResponses(unittest.TestCase):

    def test_single_response(self):
        self.assertEqual(
            find_responses({"response": "hello"}),
            ["hello"]
        )

    def test_nested_dictionary(self):
        data = {"outer": {"response": "nested"}}

        self.assertEqual(
            find_responses(data),
            ["nested"]
        )

    def test_list_of_responses(self):
        data = [
            {"response": "one"},
            {"response": "two"}
        ]

        self.assertEqual(
            find_responses(data),
            ["one", "two"]
        )

    def test_ignores_non_string_responses(self):
        data = {
            "response": 42,
            "other": {"response": "valid"}
        }

        self.assertEqual(
            find_responses(data),
            ["valid"]
        )

    def test_no_responses(self):
        self.assertEqual(
            find_responses({"message": "hello"}),
            []
        )

    def test_empty_input(self):
        self.assertEqual(find_responses({}), [])
        self.assertEqual(find_responses([]), [])

if __name__ == '__main__':
    unittest.main()