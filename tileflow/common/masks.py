"""Deterministic mask set shared by every model and evaluator."""

from __future__ import annotations

import numpy as np

from .data import DEFAULT_W, MAP_HEIGHT, SEED


def outpaint_right(width: int = DEFAULT_W) -> np.ndarray:
    mask = np.ones((MAP_HEIGHT, width), dtype=bool)
    mask[:, width // 2 :] = False
    return mask


def inpaint_center(width: int = DEFAULT_W, center_width: int = 24) -> np.ndarray:
    mask = np.ones((MAP_HEIGHT, width), dtype=bool)
    start = (width - center_width) // 2
    mask[:, start : start + center_width] = False
    return mask


def center_expand(width: int = DEFAULT_W, center_width: int = 24) -> np.ndarray:
    mask = np.zeros((MAP_HEIGHT, width), dtype=bool)
    start = (width - center_width) // 2
    mask[:, start : start + center_width] = True
    return mask


def random_rect(seed: int = SEED, width: int = DEFAULT_W) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.ones((MAP_HEIGHT, width), dtype=bool)
    for _ in range(int(rng.integers(1, 3))):
        rect_w = int(rng.integers(max(8, width // 10), max(9, width // 4)))
        rect_h = int(rng.integers(4, MAP_HEIGHT + 1))
        c0 = int(rng.integers(0, width - rect_w + 1))
        r0 = int(rng.integers(0, MAP_HEIGHT - rect_h + 1))
        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = False
    return mask


def make_mask_set(seed: int = SEED, W: int = DEFAULT_W) -> list[tuple[str, np.ndarray]]:
    return [
        ("outpaint_right", outpaint_right(W)),
        ("inpaint_center", inpaint_center(W)),
        ("center_expand", center_expand(W)),
        ("random_rect", random_rect(seed, W)),
    ]
