"""
Configuration for the Crescendo Multi-Turn Attack.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CrescendoConfig:
    # --- Models (Ollama model names) ---
    target_model: str = "llama3.1"          # Model being attacked
    orchestrator_model: str = "mistral"     # Generates escalating prompts
    judge_model: str = "llama-guard3"       # Evaluates harmful content

    # --- Attack parameters ---
    max_turns: int = 10                     # Max conversation turns per behavior
    temperature: float = 0.7               # Target model temperature
    orchestrator_temperature: float = 0.9  # Higher = more creative escalation
    seed: int = 42                         # For reproducibility (where supported)

    # --- Use a blackbox model ---
    use_black_box: bool = False

    # --- Infrastructure ---
    ollama_url: str = "http://localhost:11434"

    # --- Paths ---
    # Relative to Code/ directory (adjust if running from elsewhere)
    advbench_path: str = "../submodules/llm-attacks/data/advbench/harmful_behaviors.csv"
    output_dir: str = "results/crescendo"

    def output_path(self) -> Path:
        return Path(self.output_dir)
