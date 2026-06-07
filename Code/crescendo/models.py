"""
Data models for storing Crescendo attack results.
All dataclasses use asdict() for JSON serialisation.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import time
import json


@dataclass
class TurnResult:
    turn_number: int
    prompt: str           # What the attacker sent
    response: str         # What the target model replied
    is_refusal: bool      # Keyword-based refusal check
    is_jailbreak: bool    # Judge verdict (unsafe=True)
    judge_category: Optional[str]  # e.g. "S2" (Llama Guard category)
    judge_raw: str        # Raw judge output for debugging


@dataclass
class AttackResult:
    behavior: str          # AdvBench goal
    behavior_idx: int      # Index in AdvBench CSV
    target_model: str
    orchestrator_model: str
    success: bool          # Any turn judged as jailbreak
    num_turns: int         # How many turns were executed
    turns: List[TurnResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def jailbreak_turn(self) -> Optional[int]:
        """Returns the turn number where the first jailbreak occurred, or None."""
        for t in self.turns:
            if t.is_jailbreak:
                return t.turn_number
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["jailbreak_turn"] = self.jailbreak_turn
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
