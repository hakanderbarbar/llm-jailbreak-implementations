"""Offline integration and subprocess smoke tests; no real model calls."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.ollama_client import OllamaClient
from shared.router import OpenRouterClient
from shared.run_utils import load_behaviors, parse_behavior_selection, save_result
from shared.models import AttackResult, TurnResult
from shared.prompts import PromptLibrary
from suffix import run as suffix_run

ROOT = Path(__file__).resolve().parents[1]


class WorkflowIntegrationTests(unittest.TestCase):
    def test_csv_selection_serialization_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'behaviors.csv'
            with source.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['goal', 'target'])
                writer.writeheader()
                writer.writerows([{'goal': 'alpha', 'target': 'a'}, {'goal': 'beta', 'target': 'b'}])
            rows = load_behaviors(str(source))
            for idx in parse_behavior_selection('1', len(rows)):
                row = rows[idx]
                result = AttackResult(row['goal'], idx, 'mock-model', 'mock-orchestrator', False, 1,
                                      [TurnResult(1, 'question', 'safe answer', False, False, None, 'safe')])
                save_result(result, folder)
            data = json.loads((folder / 'behavior_001.json').read_text(encoding='utf-8'))
            self.assertEqual(data['behavior'], 'beta')
            self.assertEqual(data['turns'][0]['response'], 'safe answer')
            self.assertIsNone(data['jailbreak_turn'])

    def test_both_cli_dry_runs_as_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'behaviors.csv'
            source.write_text('goal,target\nexample topic,example answer\n', encoding='utf-8')
            for module in ('crescendo.run', 'indiana_jones.run'):
                with self.subTest(module=module):
                    command = [sys.executable, '-m', module, '--dry-run', '--advbench', str(source),
                               '--behaviors', '0', '--output-dir', str(Path(tmp) / module)]
                    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertIn('example topic', completed.stdout)

    def test_prompt_files_are_valid_and_parameterizable(self):
        required = {
            'crescendo': ('orchestrator_system', 'initial_prompt_template', 'next_turn_template', 'recovery_template', 'redirect_template'),
            'indiana_jones': ('target_system_prompt', 'inspector_system', 'inspector_fact_prompt_template',
                              'inspector_check_template', 'attacker_system', 'attacker_round2_template',
                              'attacker_deepen_template', 'keyword_system', 'keyword_template'),
        }
        for package, keys in required.items():
            with self.subTest(package=package):
                library = PromptLibrary(ROOT / package / 'prompts.json')
                for key in keys:
                    self.assertIn(key, library.templates)
                    self.assertTrue(library.templates[key].strip())

    def test_mocked_http_clients_contract(self):
        with patch('shared.ollama_client.requests.post') as post:
            post.return_value.json.return_value = {'message': {'content': 'local response'}}
            self.assertEqual(OllamaClient().chat('mock', [{'role': 'user', 'content': 'hi'}]), 'local response')
            self.assertEqual(post.call_args.kwargs['json']['model'], 'mock')
        with patch('shared.router.requests.post') as post:
            post.return_value.json.return_value = {'choices': [{'message': {'content': 'remote response'}}]}
            self.assertEqual(OpenRouterClient().chat('mock', [{'role': 'user', 'content': 'hi'}]), 'remote response')
            self.assertEqual(post.call_args.kwargs['json']['model'], 'mock')

    def test_suffix_data_filter_and_best_loss(self):
        sample = [
            {'behavior_id': 1, 'model': 'llama', 'loss': 2, 'behavior': 'example', 'suffix': ' A'},
            {'behavior_id': 1, 'model': 'llama', 'loss': 1, 'behavior': 'example', 'suffix': ' B'},
            {'behavior_id': 1, 'model': 'mistral', 'loss': 3, 'behavior': 'example', 'suffix': ' C'},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suffix_results.json'
            path.write_text(json.dumps(sample), encoding='utf-8')
            with patch.object(suffix_run, 'SUFFIX_FILE', path):
                self.assertEqual(suffix_run.load_suffixes('llama')[0]['suffix'], ' B')
                self.assertEqual(len(suffix_run.load_suffixes('all')), 2)


if __name__ == '__main__':
    unittest.main()
