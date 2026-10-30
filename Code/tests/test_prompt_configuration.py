"""Offline tests for externally configured experiment templates."""
import json
import tempfile
import unittest
from pathlib import Path

from shared.prompts import PromptLibrary
from crescendo.config import CrescendoConfig
from crescendo.attack import CrescendoAttack
from indiana_jones.config import IndianaJonesConfig
from indiana_jones.attack import IndianaJonesAttack


class PromptConfigurationTests(unittest.TestCase):
    def test_default_templates_load(self):
        for name, count in (("crescendo", 5), ("indiana_jones", 9)):
            path = Path(__file__).resolve().parents[1] / name / "prompts.json"
            self.assertEqual(len(PromptLibrary(path).templates), count)

    def test_template_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps({"greeting": "Hello, {name}!"}), encoding="utf-8")
            self.assertEqual(PromptLibrary(path).render("greeting", name="student"), "Hello, student!")
            with self.assertRaises(ValueError):
                PromptLibrary(path).render("greeting")

    def test_config_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps({"custom": "Example"}), encoding="utf-8")
            for cls, config in ((CrescendoAttack, CrescendoConfig(prompt_file=str(path))),
                                (IndianaJonesAttack, IndianaJonesConfig(prompt_file=str(path)))):
                instance = cls(config, client=object(), judge=object())
                self.assertEqual(instance.prompts.render("custom"), "Example")

    def test_invalid_json_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text('["not", "a mapping"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                PromptLibrary(path)


if __name__ == "__main__":
    unittest.main()
