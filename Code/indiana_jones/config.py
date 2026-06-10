"""
Configuration for Indiana Jones Attack.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndianaJonesConfig:
    target_model: str = "llama3.1"
    orchestrator_model: str = "mistral"
    judge_model: str = "llama-guard3"
    max_attempts: int = 5
    max_turns: int = 2
    temperature: float = 0.7
    orchestrator_temperature: float = 0.9
    seed: int = 42
    ollama_url: str = "http://localhost:11434"
    advbench_path: str = "../submodules/llm-attacks/data/advbench/harmful_behaviors.csv"
    output_dir: str = "results/indiana_jones"

    def output_path(self) -> Path:
        return Path(self.output_dir)