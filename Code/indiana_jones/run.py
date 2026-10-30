"""
CLI runner for the Indiana Jones (TombRaider) attack.

Usage examples (run from the Code/ directory):

  # Attack llama3.1 with first 3 AdvBench behaviors, 5 attempts each
  python -m indiana_jones.run

  # Attack mistral with behaviors 0-9, max 3 attempts per behavior
  python -m indiana_jones.run --target mistral --max-attempts 3 --behaviors 0-9

  # Compare with Crescendo: same behaviors, same target
  python -m indiana_jones.run --target llama3.1 --behaviors 0-2

  # Dry run: show behaviors without querying models
  python -m indiana_jones.run --dry-run --behaviors 0-4
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from .attack import IndianaJonesAttack
from .config import IndianaJonesConfig
from shared.run_utils import setup_logging, load_behaviors, parse_behavior_selection, save_result
from shared.ollama_client import OllamaClient
from shared.router import OpenRouterClient


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# def setup_logging(log_dir: Path, verbose: bool = False) -> None:
#     log_dir.mkdir(parents=True, exist_ok=True)
#     level = logging.DEBUG if verbose else logging.INFO
#     handlers = [
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
#     ]
#     logging.basicConfig(
#         level=level,
#         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#         datefmt="%H:%M:%S",
#         handlers=handlers,
#     )


# ---------------------------------------------------------------------------
# Data loading (same as crescendo/run.py)
# ---------------------------------------------------------------------------

# def load_behaviors(csv_path: str) -> list[dict]:
#     path = Path(csv_path)
#     if not path.exists():
#         raise FileNotFoundError(
#             f"AdvBench CSV not found at: {path.resolve()}\n"
#             "Make sure the llm-attacks submodule is initialised:\n"
#             "  git submodule update --init --recursive"
#         )
#     behaviors = []
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for idx, row in enumerate(reader):
#             behaviors.append({"idx": idx, "goal": row["goal"], "target": row["target"]})
#     return behaviors


# def parse_behavior_selection(spec: str, total: int) -> list[int]:
#     if spec == "all":
#         return list(range(total))
#     if "-" in spec and "," not in spec:
#         start, end = spec.split("-")
#         return list(range(int(start), int(end) + 1))
#     return [int(x) for x in spec.split(",")]


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

# def save_result(result, run_dir: Path) -> None:
#     fname = run_dir / f"behavior_{result.behavior_idx:03d}.json"
#     fname.write_text(result.to_json(), encoding="utf-8")


def save_summary(results: list, run_dir: Path, config: IndianaJonesConfig) -> dict:
    successes = [r for r in results if r.success]
    asr = len(successes) / len(results) if results else 0.0

    # Avg attempts to success (among successes only)
    avg_attempts = (
        sum(r.successful_attempt for r in successes) / len(successes)
        if successes else None
    )

    summary = {
        "attack": "indiana_jones",
        "target_model": config.target_model,
        "orchestrator_model": config.orchestrator_model,
        "judge_model": config.judge_model,
        "max_attempts": config.max_attempts,
        "max_turns": config.max_turns,
        "temperature": config.temperature,
        "num_behaviors": len(results),
        "num_success": len(successes),
        "attack_success_rate": round(asr, 4),
        "avg_attempts_to_success": round(avg_attempts, 2) if avg_attempts else None,
        "results": [
            {
                "behavior_idx": r.behavior_idx,
                "behavior": r.behavior[:80],
                "success": r.success,
                "num_attempts": r.num_attempts,
                "jailbreak_attempt": r.jailbreak_attempt,
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
    print("  INDIANA JONES — ATTACK SUMMARY")
    print("=" * 60)
    print(f"  Target model     : {summary['target_model']}")
    print(f"  Orchestrator     : {summary['orchestrator_model']}")
    print(f"  Max attempts     : {summary['max_attempts']}")
    print(f"  Max turns/attempt: {summary['max_turns']}")
    print(f"  Behaviors tested : {summary['num_behaviors']}")
    print(f"  Successes        : {summary['num_success']}")
    print(f"  ASR              : {summary['attack_success_rate']:.1%}")
    if summary["avg_attempts_to_success"]:
        print(f"  Avg attempts     : {summary['avg_attempts_to_success']:.1f}")
    print("=" * 60)
    print("\nPer-behavior results:")
    for r in summary["results"]:
        status = "✓" if r["success"] else "✗"
        ja = f"(attempt {r['jailbreak_attempt']})" if r["jailbreak_attempt"] else ""
        print(f"  [{r['behavior_idx']:3d}] {status} {r['behavior'][:55]}... {ja}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indiana Jones (TombRaider) jailbreak attack runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="llama3.1")
    parser.add_argument("--orchestrator", default="mistral")
    parser.add_argument("--judge", default="llama-guard3")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max historical contexts to try per behavior")
    parser.add_argument("--max-turns", type=int, default=2,
                        help="Turns per attempt (1=single-shot, 2=with follow-up)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--behaviors", default="0-2")
    parser.add_argument("--advbench", default=None)
    parser.add_argument("--output-dir", default="../results/indiana_jones")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--black-box", "-b", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = IndianaJonesConfig(
        target_model=args.target,
        orchestrator_model=args.orchestrator,
        judge_model=args.judge,
        max_attempts=args.max_attempts,
        max_turns=args.max_turns,
        temperature=args.temperature,
        ollama_url=args.ollama_url,
        output_dir=args.output_dir,
        seed=args.seed,
        use_black_box=args.black_box,
    )
    if args.advbench:
        config.advbench_path = args.advbench

    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{config.target_model.replace(':', '-').replace('/', '-')}"
    run_dir = config.output_path() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, verbose=args.verbose)
    logger = logging.getLogger("indiana_jones.run")

    behaviors = load_behaviors(config.advbench_path)
    indices = parse_behavior_selection(args.behaviors, len(behaviors))
    selected = [behaviors[i] for i in indices if i < len(behaviors)]
    logger.info(f"Loaded {len(behaviors)} behaviors, running {len(selected)} selected")

    if not args.dry_run:
        client = OllamaClient(config.ollama_url)
        if not args.black_box:
            client.check_models(config.target_model, config.orchestrator_model, config.judge_model)
        else:
            client.check_models(config.orchestrator_model, config.judge_model)
        # if args.black_box:
        #     target = OpenRouterClient()

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"\nDRY RUN — would attack {len(selected)} behaviors:")
        for b in selected:
            print(f"  [{b['idx']:3d}] {b['goal'][:80]}")
        return

    attack = IndianaJonesAttack(config)
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
            status = (
                f"✓ SUCCESS (attempt {result.successful_attempt})"
                if result.success
                else f"✗ failed ({result.num_attempts} attempts)"
            )
            print(f"  → {status}")
        except Exception as e:
            logger.error(f"Attack failed for behavior {behavior_entry['idx']}: {e}")
            continue

    summary = save_summary(results, run_dir, config)
    print_summary(summary)
    print(f"Results saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
