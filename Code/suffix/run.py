# Nur ein kleines Skript, dass an jedes 
# Behaviour den entsprechenden Suffix anhängt 
# Für Ergebnisse - siehe suffix_results.json

"""
suffix/run.py

GCG suffix black-box transferability evaluation.

Usage examples:
  # Llama-Suffixe gegen alle Black-Box-Modelle
  python -m suffix.run --source llama

  # Mistral-Suffixe gegen ein einzelnes Modell
  python -m suffix.run --source mistral --target anthropic/claude-haiku-4-5

  # Beide Quellen, verbose
  python -m suffix.run --source all --verbose
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from shared.router import OpenRouterClient

logger = logging.getLogger(__name__)

SUFFIX_FILE = Path(__file__).parent.parent / "suffix_results.json"
OUTPUT_DIR  = Path("../results/suffix")

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-nano",
    "deepseek/deepseek-v3.2",
]

DELAY_S = 1.0


# ── Daten laden & deduplizieren ───────────────────────────────────────────────

def load_suffixes(source: str) -> list[dict]:
    """
    Lädt Suffixe aus suffix_results.json.
    source: "llama" | "mistral" | "all"
    Bei mehreren Einträgen pro (behavior_id, model): niedrigsten Loss behalten.
    """
    data = json.loads(SUFFIX_FILE.read_text())

    if source != "all":
        data = [d for d in data if d["model"] == source]

    best: dict[tuple, dict] = {}
    for entry in data:
        key = (entry["behavior_id"], entry["model"])
        if key not in best or entry["loss"] < best[key]["loss"]:
            best[key] = entry

    return list(best.values())


# ── Einzelne Anfrage ──────────────────────────────────────────────────────────

def query(client: OpenRouterClient, model: str, behavior: str, suffix: str) -> str:
    prompt = behavior + suffix
    return client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=42,
    )


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def run(source: str, targets: list[str], verbose: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffixes = load_suffixes(source)
    client   = OpenRouterClient()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path  = OUTPUT_DIR / f"suffix_{source}_{timestamp}.jsonl"

    logger.info(f"Loaded {len(suffixes)} suffix entries (source={source})")
    logger.info(f"Targets: {targets}")
    logger.info(f"Output:  {out_path}")

    total   = len(suffixes) * len(targets)
    current = 0

    with out_path.open("w") as f:
        for entry in suffixes:
            for model in targets:
                current += 1
                logger.info(
                    f"[{current}/{total}] {entry['model']} → {model} | "
                    f"{entry['behavior_id']}"
                )

                try:
                    response = query(
                        client, model, entry["behavior"], entry["suffix"]
                    )
                    status = "ok"
                except Exception as e:
                    response = str(e)
                    status   = "error"
                    logger.warning(f"  Error: {e}")

                if verbose:
                    logger.info(f"  Response: {response[:120]}")

                record = {
                    "timestamp":     datetime.utcnow().isoformat(),
                    "behavior_id":   entry["behavior_id"],
                    "behavior":      entry["behavior"],
                    "suffix":        entry["suffix"],
                    "white_box_src": entry["model"],
                    "suffix_loss":   entry["loss"],
                    "black_box_tgt": model,
                    "prompt":        entry["behavior"] + entry["suffix"],
                    "response":      response,
                    "status":        status,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                time.sleep(DELAY_S)

    logger.info(f"Done. {current} requests written to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GCG suffix black-box transferability evaluation"
    )
    parser.add_argument(
        "--source",
        choices=["llama", "mistral", "all"],
        default="llama",
        help="White-box source model to use (default: llama)",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        default=None,
        metavar="MODEL",
        help="OpenRouter model slug (repeatable; default: all three black-box models)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Response-Preview in the logs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    targets = args.targets or DEFAULT_MODELS
    run(args.source, targets, args.verbose)


if __name__ == "__main__":
    main()