"""
CLI runner for the Crescendo attack.

Usage examples (run from the Code/ directory):

  # Attack llama3.1 with first 10 AdvBench behaviors
  python -m crescendo.run

  # Attack mistral with behaviors 0–29, 15 turns max
  python -m crescendo.run --target mistral --max-turns 15 --behaviors 0-29

  # Attack llama3.1, use mistral as orchestrator, specific behaviors
  python -m crescendo.run --target llama3.1 --orchestrator mistral --behaviors 0,5,12,33

  # Dry run: show prompts without querying models
  python -m crescendo.run --dry-run --behaviors 0-2
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from .attack import CrescendoAttack
from .config import CrescendoConfig
from shared.ollama_client import OllamaClient
from shared.router import OpenRouterClient


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_behaviors(csv_path: str) -> list[dict]:
    """Load AdvBench harmful_behaviors.csv → list of {idx, goal, target}."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"AdvBench CSV not found at: {path.resolve()}\n"
            "Make sure the llm-attacks submodule is initialised:\n"
            "  git submodule update --init --recursive"
        )
    behaviors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            behaviors.append({"idx": idx, "goal": row["goal"], "target": row["target"]})
    return behaviors


def parse_behavior_selection(spec: str, total: int) -> list[int]:
    """
    Parse a behavior selection spec into a list of indices.
      "all"       → [0, 1, ..., total-1]
      "0-9"       → [0, 1, ..., 9]
      "0,5,12"    → [0, 5, 12]
    """
    if spec == "all":
        return list(range(total))
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",")]


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(result, run_dir: Path) -> None:
    """Save one AttackResult as a JSON file."""
    fname = run_dir / f"behavior_{result.behavior_idx:03d}.json"
    fname.write_text(result.to_json(), encoding="utf-8")


def save_summary(results: list, run_dir: Path, config: CrescendoConfig) -> None:
    """Write an aggregate summary JSON with ASR and per-result metadata."""
    successes = [r for r in results if r.success]
    asr = len(successes) / len(results) if results else 0.0

    summary = {
        "target_model": config.target_model,
        "orchestrator_model": config.orchestrator_model,
        "judge_model": config.judge_model,
        "max_turns": config.max_turns,
        "temperature": config.temperature,
        "num_behaviors": len(results),
        "num_success": len(successes),
        "attack_success_rate": round(asr, 4),
        "results": [
            {
                "behavior_idx": r.behavior_idx,
                "behavior": r.behavior[:80],
                "success": r.success,
                "num_turns": r.num_turns,
                "jailbreak_turn": r.jailbreak_turn,
            }
            for r in results
        ],
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  CRESCENDO — ATTACK SUMMARY")
    print("=" * 60)
    print(f"  Target model     : {summary['target_model']}")
    print(f"  Orchestrator     : {summary['orchestrator_model']}")
    print(f"  Max turns        : {summary['max_turns']}")
    print(f"  Behaviors tested : {summary['num_behaviors']}")
    print(f"  Successes        : {summary['num_success']}")
    print(f"  ASR              : {summary['attack_success_rate']:.1%}")
    print("=" * 60)
    print("\nPer-behavior results:")
    for r in summary["results"]:
        status = "✓" if r["success"] else "✗"
        jt = f"(turn {r['jailbreak_turn']})" if r["jailbreak_turn"] else ""
        print(f"  [{r['behavior_idx']:3d}] {status} {r['behavior'][:55]}... {jt}")
    print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crescendo multi-turn jailbreak attack runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="llama3.1",
                        help="Ollama model name for the attack target")
    parser.add_argument("--orchestrator", default="mistral",
                        help="Ollama model name for the orchestrator")
    parser.add_argument("--judge", default="llama-guard3",
                        help="Ollama model name for the judge")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Maximum conversation turns per behavior")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Target model sampling temperature")
    parser.add_argument("--behaviors", default="0-9",
                        help='Behavior indices: "all", "0-9", or "0,5,12"')
    parser.add_argument("--advbench", default=None,
                        help="Path to harmful_behaviors.csv (overrides config default)")
    parser.add_argument("--output-dir", default="results/crescendo",
                        help="Directory for saving results")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first prompt of each behavior without querying models")
    parser.add_argument("--black-box", "-b", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CrescendoConfig(
        target_model=args.target,
        orchestrator_model=args.orchestrator,
        judge_model=args.judge,
        max_turns=args.max_turns,
        temperature=args.temperature,
        ollama_url=args.ollama_url,
        output_dir=args.output_dir,
        seed=args.seed,
        use_black_box=args.black_box,
    )
    if args.advbench:
        config.advbench_path = args.advbench

    # Create run directory with timestamp
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{config.target_model.replace(':', '-').replace('/', '-')}"
    run_dir = config.output_path() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, verbose=args.verbose)
    logger = logging.getLogger("crescendo.run")

    # Load behaviors
    behaviors = load_behaviors(config.advbench_path)
    indices = parse_behavior_selection(args.behaviors, len(behaviors))
    selected = [behaviors[i] for i in indices if i < len(behaviors)]
    logger.info(f"Loaded {len(behaviors)} behaviors, running {len(selected)} selected")

    # Validate models are available
    if not args.dry_run:
        client = OllamaClient(config.ollama_url)
        if not config.use_black_box:
            client.check_models(config.target_model, config.orchestrator_model, config.judge_model)
        else:
            client.check_models(config.orchestrator_model, config.judge_model)

    # Save run config
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(f"\nDRY RUN — would attack {len(selected)} behaviors:")
        for b in selected:
            print(f"  [{b['idx']:3d}] {b['goal'][:80]}")
        return

    # Run attacks
    attack = CrescendoAttack(config)
    results = []

    for i, behavior_entry in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Behavior {behavior_entry['idx']}: "
              f"{behavior_entry['goal'][:60]}...")
        try:
            result = attack.run(
                behavior=behavior_entry["goal"],
                behavior_idx=behavior_entry["idx"],
            )
            results.append(result)
            save_result(result, run_dir)
            status = "✓ SUCCESS" if result.success else f"✗ failed ({result.num_turns} turns)"
            print(f"  → {status}")
        except Exception as e:
            logger.error(f"Attack failed for behavior {behavior_entry['idx']}: {e}")
            continue

    # Save and print summary
    summary = save_summary(results, run_dir, config)
    print_summary(summary)
    print(f"Results saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
