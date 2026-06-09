"""
Configuration for the Indiana Jones (TombRaider) Attack.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndianaJonesConfig:
    # --- Models (Ollama model names) ---
    target_model: str = "llama3.1"          # Model being attacked
    orchestrator_model: str = "mistral"     # Generates historical context
    judge_model: str = "llama-guard3"       # Evaluates harmful content

    # --- Attack parameters ---
    max_attempts: int = 5      # Number of different historical contexts to try
    max_turns: int = 3         # Turns per attempt (1 = single-shot, 2-3 = follow-up)
    temperature: float = 0.7
    orchestrator_temperature: float = 0.9
    seed: int = 42

    # --- Infrastructure ---
    ollama_url: str = "http://localhost:11434"

    # --- Paths ---
    advbench_path: str = "../submodules/llm-attacks/data/advbench/harmful_behaviors.csv"
    output_dir: str = "results/indiana_jones"

    def output_path(self) -> Path:
        return Path(self.output_dir)
