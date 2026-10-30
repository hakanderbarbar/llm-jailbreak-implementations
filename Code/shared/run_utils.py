"""Reusable CLI infrastructure for experiment runners."""
import csv
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / 'run.log', encoding='utf-8')],
        force=True,
    )


def load_behaviors(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f'AdvBench CSV not found at: {path.resolve()}')
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not {'goal', 'target'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain goal and target columns')
        return [{'idx': i, 'goal': row['goal'], 'target': row['target']} for i, row in enumerate(reader)]


def parse_behavior_selection(spec: str, total: int) -> list[int]:
    if total < 0:
        raise ValueError('total must be nonnegative')
    if spec.strip() == 'all':
        return list(range(total))
    result = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            bounds = part.split('-')
            if len(bounds) != 2:
                raise ValueError(f'Invalid range: {part}')
            start, end = map(int, bounds)
            if start > end:
                raise ValueError(f'Descending range: {part}')
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    if any(i < 0 or i >= total for i in result):
        raise ValueError('Behavior index outside dataset bounds')
    return list(dict.fromkeys(result))


def save_result(result, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f'behavior_{result.behavior_idx:03d}.json').write_text(result.to_json(), encoding='utf-8')
