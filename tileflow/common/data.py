"""Shared data contract for TileFlow.

All model-facing level exchange uses 14 strings of width 80 and the fixed
13-character VGLC vocabulary used by the current Mario/Lost Levels dataset.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

MAP_HEIGHT = 14
DEFAULT_W = 80
SEED = 42
VOCAB = ["-", "X", "S", "Q", "E", "?", "<", ">", "[", "]", "B", "b", "o"]
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}
PAD_TILE = "-"


@dataclass(frozen=True)
class LevelWindow:
    source: str
    window_index: int
    level: list[str]


def normalize_level(level: Iterable[str], width: int = DEFAULT_W) -> list[str]:
    """Return exactly 14 rows of exactly `width` known-vocab characters."""
    rows = list(level)[:MAP_HEIGHT]
    rows.extend([PAD_TILE] * (MAP_HEIGHT - len(rows)))
    out: list[str] = []
    for row in rows:
        cleaned = "".join(ch if ch in CHAR2IDX else PAD_TILE for ch in row)
        out.append((cleaned + PAD_TILE * width)[:width])
    return out


def load_level(path: str | Path, width: int | None = None) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if width is None:
        width = max((len(row) for row in rows), default=DEFAULT_W)
    return normalize_level(rows, width=width)


def sliding_windows(level: list[str], width: int = DEFAULT_W, stride: int = 10) -> list[list[str]]:
    rows = normalize_level(level, width=max(width, max(map(len, level), default=width)))
    max_w = max(len(row) for row in rows)
    if max_w <= width:
        return [normalize_level(rows, width=width)]
    windows = []
    for start in range(0, max_w - width + 1, stride):
        windows.append([row[start : start + width] for row in rows])
    return windows


def _selected_paths(folder: Path, file_names: Iterable[str] | None = None) -> list[Path]:
    if file_names is None:
        return sorted(folder.glob("*.txt"))
    wanted = set(file_names)
    return [folder / name for name in sorted(wanted) if (folder / name).exists()]


def load_levels(
    folder: str | Path,
    width: int = DEFAULT_W,
    stride: int = 10,
    file_names: Iterable[str] | None = None,
) -> list[list[str]]:
    folder = Path(folder)
    levels: list[list[str]] = []
    for path in _selected_paths(folder, file_names):
        raw = path.read_text(encoding="utf-8").splitlines()
        levels.extend(sliding_windows(raw, width=width, stride=stride))
    return levels


def load_level_windows(
    folder: str | Path,
    width: int = DEFAULT_W,
    stride: int = 10,
    file_names: Iterable[str] | None = None,
) -> list[LevelWindow]:
    folder = Path(folder)
    records: list[LevelWindow] = []
    for path in _selected_paths(folder, file_names):
        raw = path.read_text(encoding="utf-8").splitlines()
        for i, window in enumerate(sliding_windows(raw, width=width, stride=stride)):
            records.append(LevelWindow(source=path.stem, window_index=i, level=window))
    return records


def make_level_splits(folder: str | Path, eval_count: int = 2, seed: int = SEED) -> dict[str, list[str]]:
    paths = sorted(str(p.name) for p in Path(folder).glob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    eval_files = sorted(paths[:eval_count])
    train_files = sorted(paths[eval_count:])
    return {"train": train_files, "eval": eval_files}


def make_level_splits_with_eval_files(folder: str | Path, eval_files: Iterable[str]) -> dict[str, list[str]]:
    paths = sorted(str(p.name) for p in Path(folder).glob("*.txt"))
    eval_set = set(eval_files)
    missing = sorted(eval_set - set(paths))
    if missing:
        raise FileNotFoundError(f"Missing eval files in {folder}: {missing}")
    return {
        "train": [name for name in paths if name not in eval_set],
        "eval": [name for name in paths if name in eval_set],
    }


def encode_level(level: list[str], width: int = DEFAULT_W) -> np.ndarray:
    rows = normalize_level(level, width=width)
    arr = np.zeros((len(VOCAB), MAP_HEIGHT, width), dtype=np.float32)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            arr[CHAR2IDX[ch], r, c] = 1.0
    return arr


def decode_level(onehot_or_idx: np.ndarray) -> list[str]:
    arr = np.asarray(onehot_or_idx)
    if arr.ndim == 3:
        idx = arr.argmax(axis=0)
    elif arr.ndim == 2:
        idx = arr
    else:
        raise ValueError("Expected (C,H,W) one-hot/logits or (H,W) indices.")
    h, w = idx.shape
    if h != MAP_HEIGHT:
        raise ValueError(f"Expected height {MAP_HEIGHT}, got {h}.")
    return ["".join(IDX2CHAR[int(idx[r, c])] for c in range(w)) for r in range(MAP_HEIGHT)]


def write_common_artifacts(root: str | Path, splits: dict[str, list[str]] | None = None) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "vocab.json").write_text(json.dumps(VOCAB, indent=2), encoding="utf-8")
    if splits is not None:
        (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
