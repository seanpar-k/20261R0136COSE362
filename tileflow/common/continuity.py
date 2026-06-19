"""Boundary continuity metrics and light repairs for TileFlow."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from tileflow.common.data import MAP_HEIGHT, normalize_level

COIN_TILES = set("o")
PLATFORM_TILES = set("SQ?X")
GROUND_TILES = set("X")
SEMANTIC_TILES = {
    "-": "-",
    "o": "coin",
    "S": "block",
    "Q": "block",
    "?": "block",
    "X": "solid",
    "<": "pipe",
    ">": "pipe",
    "[": "pipe",
    "]": "pipe",
    "B": "cannon",
    "b": "cannon",
    "E": "enemy",
}


@dataclass(frozen=True)
class Boundary:
    known_col: int
    gen_col: int
    direction: int


def _boundaries(mask: np.ndarray) -> list[Boundary]:
    out: list[Boundary] = []
    for c in range(mask.shape[1] - 1):
        left_known = bool(mask[:, c].any())
        right_known = bool(mask[:, c + 1].any())
        left_gen = bool((~mask)[:, c].any())
        right_gen = bool((~mask)[:, c + 1].any())
        if left_known and right_gen:
            out.append(Boundary(c, c + 1, 1))
        if left_gen and right_known:
            out.append(Boundary(c + 1, c, -1))
    return out


def _same_region(mask: np.ndarray, c: int, known: bool) -> bool:
    return bool(mask[:, c].any()) if known else bool((~mask)[:, c].any())


def _run_len(rows: list[str], r: int, c: int, direction: int, tiles: set[str], mask: np.ndarray, known: bool) -> int:
    width = mask.shape[1]
    n = 0
    while 0 <= c < width and _same_region(mask, c, known) and rows[r][c] in tiles:
        n += 1
        c += direction
    return n


def _first_cols(boundary: Boundary, width: int, span: int = 4) -> range:
    stop = boundary.gen_col + boundary.direction * span
    if boundary.direction > 0:
        return range(boundary.gen_col, min(width, stop))
    return range(boundary.gen_col, max(-1, stop), -1)


def _surface_height(rows: list[str], c: int) -> int | None:
    for r in range(MAP_HEIGHT - 1, 0, -1):
        if rows[r][c] in GROUND_TILES and rows[r - 1][c] == "-":
            return r
    return None


def _kl(p: Counter[str], q: Counter[str], eps: float = 1e-6) -> float:
    keys = set(p) | set(q)
    p_total = sum(p.values()) + eps * len(keys)
    q_total = sum(q.values()) + eps * len(keys)
    return float(
        sum(
            ((p[k] + eps) / p_total) * math.log(((p[k] + eps) / p_total) / ((q[k] + eps) / q_total))
            for k in keys
        )
    )


def continuity_metrics(level: list[str], known_mask: np.ndarray) -> dict[str, float]:
    rows = normalize_level(level, width=known_mask.shape[1])
    width = known_mask.shape[1]
    boundaries = _boundaries(known_mask)

    coin_hits = coin_total = 0
    platform_hits = platform_total = 0
    ground_hits = ground_total = 0
    ref_sem: Counter[str] = Counter()
    gen_sem: Counter[str] = Counter()

    for boundary in boundaries:
        for r in range(MAP_HEIGHT):
            ref_sem[SEMANTIC_TILES.get(rows[r][boundary.known_col], "other")] += 1
            gen_sem[SEMANTIC_TILES.get(rows[r][boundary.gen_col], "other")] += 1

            coin_run = _run_len(rows, r, boundary.known_col, -boundary.direction, COIN_TILES, known_mask, known=True)
            if coin_run >= 2:
                coin_total += 1
                if any(rows[r][c] in COIN_TILES for c in _first_cols(boundary, width, span=4)):
                    coin_hits += 1

            platform_run = _run_len(rows, r, boundary.known_col, -boundary.direction, PLATFORM_TILES, known_mask, known=True)
            if platform_run >= 2:
                platform_total += 1
                if any(rows[r][c] in PLATFORM_TILES for c in _first_cols(boundary, width, span=3)):
                    platform_hits += 1

        known_h = _surface_height(rows, boundary.known_col)
        if known_h is not None:
            ground_total += 1
            gen_heights = [
                _surface_height(rows, c)
                for c in _first_cols(boundary, width, span=4)
            ]
            gen_heights = [h for h in gen_heights if h is not None]
            if gen_heights and min(abs(h - known_h) for h in gen_heights) <= 1:
                ground_hits += 1

    coin_rate = 1.0 if coin_total == 0 else coin_hits / coin_total
    platform_rate = 1.0 if platform_total == 0 else platform_hits / platform_total
    ground_rate = 1.0 if ground_total == 0 else ground_hits / ground_total
    boundary_kl = _kl(gen_sem, ref_sem) if ref_sem and gen_sem else 0.0
    continuity_score = (coin_rate + platform_rate + ground_rate + max(0.0, 1.0 - min(boundary_kl, 1.0))) / 4.0

    return {
        "coin_run_continuation_rate": float(coin_rate),
        "coin_run_opportunities": float(coin_total),
        "platform_row_continuity": float(platform_rate),
        "platform_row_opportunities": float(platform_total),
        "ground_height_continuity": float(ground_rate),
        "ground_height_opportunities": float(ground_total),
        "boundary_pattern_kl": float(boundary_kl),
        "continuity_score": float(continuity_score),
    }


def add_boundary_continuity_echoes(level: list[str], known_mask: np.ndarray, max_echoes: int = 3) -> list[str]:
    """Create a conservative variant that extends clear boundary patterns."""
    rows = [list(row) for row in normalize_level(level, width=known_mask.shape[1])]
    source = normalize_level(level, width=known_mask.shape[1])
    width = known_mask.shape[1]
    echoes = 0
    for boundary in _boundaries(known_mask):
        for r in range(MAP_HEIGHT):
            if echoes >= max_echoes:
                return ["".join(row) for row in rows]
            coin_run = _run_len(source, r, boundary.known_col, -boundary.direction, COIN_TILES, known_mask, known=True)
            if coin_run >= 2:
                span = min(3, coin_run)
                for c in _first_cols(boundary, width, span=span):
                    if not known_mask[r, c] and rows[r][c] == "-":
                        rows[r][c] = "o"
                echoes += 1
                continue

            platform_run = _run_len(source, r, boundary.known_col, -boundary.direction, PLATFORM_TILES, known_mask, known=True)
            if platform_run >= 3:
                tile = source[r][boundary.known_col]
                for c in _first_cols(boundary, width, span=min(2, platform_run)):
                    if not known_mask[r, c] and rows[r][c] == "-":
                        rows[r][c] = tile
                echoes += 1
    return ["".join(row) for row in rows]
