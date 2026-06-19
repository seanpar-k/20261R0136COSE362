"""Normalize Lost Levels text maps to the TileFlow 14-row data contract.

Raw VGLC Lost Levels files are not all 14 rows tall. This script preserves
bottom alignment so ground and obstacle structures stay at the expected rows:

- shorter than 14 rows: prepend sky rows
- taller than 14 rows: keep the bottom 14 rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.common.data import CHAR2IDX, MAP_HEIGHT, PAD_TILE


def normalize_rows_bottom_aligned(rows: list[str]) -> list[str]:
    if not rows:
        return [PAD_TILE] * MAP_HEIGHT

    width = max(len(row) for row in rows)
    cleaned_rows = []
    for row in rows:
        cleaned = "".join(ch if ch in CHAR2IDX else PAD_TILE for ch in row)
        cleaned_rows.append((cleaned + PAD_TILE * width)[:width])

    if len(cleaned_rows) >= MAP_HEIGHT:
        return cleaned_rows[-MAP_HEIGHT:]

    pad_row = PAD_TILE * width
    return [pad_row] * (MAP_HEIGHT - len(cleaned_rows)) + cleaned_rows


def preprocess_file(path: Path) -> tuple[int, int]:
    raw_rows = path.read_text(encoding="utf-8").splitlines()
    normalized = normalize_rows_bottom_aligned(raw_rows)
    path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return len(raw_rows), len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pattern", default="lost-levels-*.txt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {data_dir / args.pattern}")

    changed = []
    for path in paths:
        before, after = preprocess_file(path)
        if before != after:
            changed.append((path.name, before, after))

    print(f"normalized_files={len(paths)}")
    print(f"changed_row_counts={len(changed)}")
    for name, before, after in changed:
        print(f"{name}: {before} -> {after}")


if __name__ == "__main__":
    main()
