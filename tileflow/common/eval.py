"""Lightweight TileFlow v1 evaluation harness."""

from __future__ import annotations

import math
import time
from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np

from .data import MAP_HEIGHT, VOCAB, normalize_level
from .fill_api import FillModel, assert_known_preserved
from .style import style_metrics
from .continuity import continuity_metrics

SOLID = set("XSQ?<>[]Bb")
PIPE_TOP = set("<>")
PIPE_BODY = set("[]")
FLOATING_ALLOWED_SOLID = set("SQ?oE")
RARE_TILES = set("Q?E<>[]Bbo")


def _patterns(level: list[str], n: int = 2) -> Counter[tuple[str, ...]]:
    rows = normalize_level(level, width=len(level[0]))
    h, w = MAP_HEIGHT, len(rows[0])
    out: Counter[tuple[str, ...]] = Counter()
    for r in range(h - n + 1):
        for c in range(w - n + 1):
            out[tuple(rows[rr][cc] for rr in range(r, r + n) for cc in range(c, c + n))] += 1
    return out


def _kl(p_counts: Counter, q_counts: Counter, eps: float = 1e-6) -> float:
    keys = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values()) + eps * len(keys)
    q_total = sum(q_counts.values()) + eps * len(keys)
    return float(
        sum(
            ((p_counts[k] + eps) / p_total) * math.log(((p_counts[k] + eps) / p_total) / ((q_counts[k] + eps) / q_total))
            for k in keys
        )
    )


def tile_pattern_kl(gen_list: Iterable[list[str]], real_list: Iterable[list[str]], n: int = 2) -> float:
    gen_counts: Counter = Counter()
    real_counts: Counter = Counter()
    for level in gen_list:
        gen_counts.update(_patterns(level, n=n))
    for level in real_list:
        real_counts.update(_patterns(level, n=n))
    return _kl(gen_counts, real_counts)


def _terrain_states(level: list[str], mask: np.ndarray) -> list[str]:
    rows = normalize_level(level, width=mask.shape[1])
    states: list[str] = []
    for c in range(mask.shape[1]):
        if bool(mask[:, c].any()):
            continue
        bottom_solid = rows[MAP_HEIGHT - 1][c] in SOLID or rows[MAP_HEIGHT - 2][c] in SOLID
        raised_solid = any(rows[r][c] in SOLID for r in range(6, MAP_HEIGHT - 2))
        mid_platform = any(rows[r][c] in "XSQ?" for r in range(3, MAP_HEIGHT - 3))
        structure = any(rows[r][c] in "SQ?E<>[]Bbo" for r in range(MAP_HEIGHT))
        if structure and bottom_solid:
            states.append("structure")
        elif mid_platform and not bottom_solid:
            states.append("mid_platform")
        elif not bottom_solid:
            states.append("gap")
        elif raised_solid:
            states.append("raised")
        else:
            states.append("ground")
    return states


def _sequence_ngrams(values: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(values) < n:
        return Counter()
    return Counter(tuple(values[i : i + n]) for i in range(len(values) - n + 1))


def terrain_state_ngram_kl(
    gen_list: Iterable[list[str]],
    real_list: Iterable[list[str]],
    mask: np.ndarray,
    n: int = 3,
) -> float:
    gen_counts: Counter = Counter()
    real_counts: Counter = Counter()
    for level in gen_list:
        gen_counts.update(_sequence_ngrams(_terrain_states(level, mask), n=n))
    for level in real_list:
        real_counts.update(_sequence_ngrams(_terrain_states(level, mask), n=n))
    return _kl(gen_counts, real_counts)


def diversity(gen_list: list[list[str]]) -> float:
    if len(gen_list) < 2:
        return 0.0
    distances = []
    for a, b in combinations(gen_list, 2):
        a = normalize_level(a, width=len(a[0]))
        b = normalize_level(b, width=len(a[0]))
        total = MAP_HEIGHT * len(a[0])
        diff = sum(a[r][c] != b[r][c] for r in range(MAP_HEIGHT) for c in range(len(a[0])))
        distances.append(diff / total)
    return float(np.mean(distances))


def masked_hamming(a: list[str], b: list[str], mask: np.ndarray) -> float:
    a = normalize_level(a, width=mask.shape[1])
    b = normalize_level(b, width=mask.shape[1])
    unknown = ~mask
    total = int(unknown.sum())
    if total == 0:
        return 0.0
    diff = sum(a[r][c] != b[r][c] for r in range(MAP_HEIGHT) for c in range(mask.shape[1]) if unknown[r, c])
    return diff / total


def masked_diversity(gen_list: list[list[str]], mask: np.ndarray) -> float:
    if len(gen_list) < 2:
        return 0.0
    return float(np.mean([masked_hamming(a, b, mask) for a, b in combinations(gen_list, 2)]))


def playable_masked_diversity(gen_list: list[list[str]], mask: np.ndarray) -> float:
    playable = [level for level in gen_list if playability(level)["completable"]]
    return masked_diversity(playable, mask)


def seam_transition_kl(level: list[str], mask: np.ndarray, real_list: Iterable[list[str]] | None = None) -> dict[str, float]:
    rows = normalize_level(level, width=mask.shape[1])
    real_rows = list(real_list or [rows])
    ref: Counter[tuple[str, str]] = Counter()
    for real in real_rows:
        real = normalize_level(real, width=mask.shape[1])
        for r in range(MAP_HEIGHT):
            for c in range(mask.shape[1] - 1):
                ref[(real[r][c], real[r][c + 1])] += 1

    left: Counter[tuple[str, str]] = Counter()
    right: Counter[tuple[str, str]] = Counter()
    for r in range(MAP_HEIGHT):
        for c in range(mask.shape[1] - 1):
            pair = (rows[r][c], rows[r][c + 1])
            if mask[r, c] and not mask[r, c + 1]:
                left[pair] += 1
            if not mask[r, c] and mask[r, c + 1]:
                right[pair] += 1
    return {
        "left": _kl(left, ref) if left else 0.0,
        "right": _kl(right, ref) if right else 0.0,
    }


def structural_violations(level: list[str]) -> dict[str, float | int]:
    rows = normalize_level(level, width=len(level[0]))
    width = len(rows[0])
    pipe = floating = floor = 0
    for r in range(MAP_HEIGHT):
        for c in range(width):
            ch = rows[r][c]
            if ch == "<" and (c + 1 >= width or rows[r][c + 1] != ">"):
                pipe += 1
            if ch == "[" and (c + 1 >= width or rows[r][c + 1] != "]"):
                pipe += 1
            if ch in PIPE_TOP and r + 1 < MAP_HEIGHT and rows[r + 1][c] not in PIPE_BODY | SOLID:
                pipe += 1
            if ch in SOLID and r + 1 < MAP_HEIGHT and rows[r + 1][c] == "-" and ch not in FLOATING_ALLOWED_SOLID:
                floating += 1

    max_gap = 0
    run = 0
    for c in range(width):
        grounded = rows[MAP_HEIGHT - 1][c] in SOLID or rows[MAP_HEIGHT - 2][c] in SOLID
        run = 0 if grounded else run + 1
        max_gap = max(max_gap, run)
    if max_gap > 4:
        floor = max_gap - 4

    total = pipe + floating + floor
    return {"pipe": pipe, "floor": floor, "floating": floating, "per_col": total / max(width, 1)}


def playability_gap_run(level: list[str]) -> dict[str, float | bool]:
    rows = normalize_level(level, width=len(level[0]))
    width = len(rows[0])
    progress_col = width - 1
    run = 0
    for c in range(width):
        grounded = rows[MAP_HEIGHT - 1][c] in SOLID or rows[MAP_HEIGHT - 2][c] in SOLID
        run = 0 if grounded else run + 1
        if run > 4:
            progress_col = c
            break
    progress = progress_col / max(width - 1, 1)
    return {"completable": progress >= 1.0, "progress": float(progress)}


def _surface_nodes(rows: list[str]) -> dict[int, list[int]]:
    width = len(rows[0])
    nodes: dict[int, list[int]] = {}
    for c in range(width):
        surfaces = []
        for r in range(1, MAP_HEIGHT):
            if rows[r][c] in SOLID and rows[r - 1][c] == "-":
                surfaces.append(r - 1)
        if surfaces:
            nodes[c] = sorted(set(surfaces))
    return nodes


def playability_jump_reachability(
    level: list[str],
    max_jump_dx: int = 4,
    max_jump_up: int = 4,
    max_safe_drop: int = 6,
) -> dict[str, float | bool]:
    """A lightweight A*-like forward reachability check.

    External Java A* agents are not bundled here. This local scorer keeps the
    same output contract while modeling the key SMB
    constraint we need for reranking: can the player chain landings across
    gaps, platforms, and height changes?
    """
    rows = normalize_level(level, width=len(level[0]))
    width = len(rows[0])
    nodes = _surface_nodes(rows)
    if not nodes:
        return {"completable": False, "progress": 0.0, "gap_progress": 0.0}

    start_cols = [c for c in range(min(width, max_jump_dx + 1)) if c in nodes]
    if not start_cols:
        start_cols = [min(nodes)]
    frontier = [(c, y) for c in start_cols for y in nodes[c]]
    seen = set(frontier)
    best_col = max(c for c, _ in frontier)

    while frontier:
        c, y = frontier.pop(0)
        best_col = max(best_col, c)
        for nc in range(c + 1, min(width, c + max_jump_dx + 1)):
            for ny in nodes.get(nc, []):
                jump_up = y - ny
                drop = ny - y
                if jump_up > max_jump_up or drop > max_safe_drop:
                    continue
                node = (nc, ny)
                if node in seen:
                    continue
                seen.add(node)
                frontier.append(node)

    gap = playability_gap_run(rows)
    progress = best_col / max(width - 1, 1)
    return {
        "completable": progress >= 1.0,
        "progress": float(progress),
        "gap_progress": float(gap["progress"]),
    }


def playability(level: list[str]) -> dict[str, float | bool]:
    return playability_jump_reachability(level)


def context_consistency(filled: list[str], original: list[str], mask: np.ndarray) -> dict[str, float]:
    filled = normalize_level(filled, width=mask.shape[1])
    original = normalize_level(original, width=mask.shape[1])
    unknown = ~mask
    total = int(unknown.sum())
    if total == 0:
        return {"tile_acc": 1.0, "valid_rate": 1.0}
    correct = sum(filled[r][c] == original[r][c] for r in range(MAP_HEIGHT) for c in range(mask.shape[1]) if unknown[r, c])
    struct = structural_violations(filled)
    return {"tile_acc": correct / total, "valid_rate": float(struct["per_col"] == 0.0)}


def rare_tile_scores(filled: list[str], original: list[str], mask: np.ndarray) -> dict[str, float]:
    filled = normalize_level(filled, width=mask.shape[1])
    original = normalize_level(original, width=mask.shape[1])
    unknown = ~mask
    tp = fp = fn = 0
    recalls = []
    for tile in sorted(RARE_TILES):
        tile_tp = tile_fn = 0
        for r in range(MAP_HEIGHT):
            for c in range(mask.shape[1]):
                if not unknown[r, c]:
                    continue
                pred_is = filled[r][c] == tile
                true_is = original[r][c] == tile
                if pred_is and true_is:
                    tp += 1
                    tile_tp += 1
                elif pred_is and not true_is:
                    fp += 1
                elif true_is:
                    fn += 1
                    tile_fn += 1
        if tile_tp + tile_fn > 0:
            recalls.append(tile_tp / (tile_tp + tile_fn))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "rare_precision": precision,
        "rare_recall": recall,
        "rare_f1": f1,
        "object_recall": float(np.mean(recalls)) if recalls else 0.0,
    }


def swap_right_context(base: list[str], donor: list[str], mask: np.ndarray) -> list[str]:
    base = normalize_level(base, width=mask.shape[1])
    donor = normalize_level(donor, width=mask.shape[1])
    unknown_cols = np.where((~mask).any(axis=0))[0]
    if len(unknown_cols) == 0:
        return base
    right_start = int(unknown_cols.max()) + 1
    rows = []
    for r in range(MAP_HEIGHT):
        rows.append(base[r][:right_start] + donor[r][right_start:])
    return rows


def right_context_sensitivity(
    model: FillModel,
    contexts: list[list[str]],
    mask: np.ndarray,
    N: int = 2,
    max_contexts: int = 4,
) -> dict[str, float]:
    """Measure whether completions respond to the right-side known context.

    We compare outputs for the same left context with its original right side
    versus a right side swapped from another eval level. To avoid confusing
    stochastic sampling with actual conditioning, `right_context_effect`
    subtracts same-context masked diversity.
    """
    contexts = contexts[:max_contexts]
    if len(contexts) < 2:
        return {"right_context_sensitivity": 0.0, "self_masked_diversity": 0.0, "right_context_effect": 0.0}

    self_distances = []
    swap_distances = []
    for i, base in enumerate(contexts):
        original_outputs = [model.fill(base, mask) for _ in range(N)]
        self_distances.extend(masked_hamming(a, b, mask) for a, b in combinations(original_outputs, 2))
        for j, donor in enumerate(contexts):
            if i == j:
                continue
            swapped = swap_right_context(base, donor, mask)
            swapped_outputs = [model.fill(swapped, mask) for _ in range(N)]
            for original in original_outputs:
                for changed in swapped_outputs:
                    swap_distances.append(masked_hamming(original, changed, mask))

    self_div = float(np.mean(self_distances)) if self_distances else 0.0
    sensitivity = float(np.mean(swap_distances)) if swap_distances else 0.0
    return {
        "right_context_sensitivity": sensitivity,
        "self_masked_diversity": self_div,
        "right_context_effect": max(0.0, sensitivity - self_div),
    }


def efficiency(model: FillModel, example_level: list[str] | None = None, example_mask: np.ndarray | None = None) -> dict[str, float | int]:
    params = 0
    module = getattr(model, "model", None)
    if module is not None and not hasattr(module, "parameters"):
        module = getattr(module, "model", module)
    if module is not None and hasattr(module, "parameters"):
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    latency = 0.0
    if example_level is not None and example_mask is not None:
        start = time.perf_counter()
        model.fill(example_level, example_mask)
        latency = (time.perf_counter() - start) * 1000
    return {"params_M": params / 1_000_000, "latency_ms": latency, "steps": int(getattr(model, "sample_steps", 1))}


def evaluate(model: FillModel, mask_set: list[tuple[str, np.ndarray]], real_contexts: list[list[str]], N: int = 8) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for mask_name, mask in mask_set:
        filled_levels = []
        play = []
        struct = []
        consistency = []
        rare = []
        seams = []
        for i, context in enumerate(real_contexts):
            for _ in range(N):
                filled = model.fill(context, mask)
                assert_known_preserved(context, filled, mask)
                filled_levels.append(filled)
                play.append(playability(filled))
                struct.append(structural_violations(filled))
                consistency.append(context_consistency(filled, context, mask))
                rare.append(rare_tile_scores(filled, context, mask))
                seams.append(seam_transition_kl(filled, mask, real_contexts))
        report[mask_name] = {
            "completable_rate": float(np.mean([p["completable"] for p in play])),
            "progress_mean": float(np.mean([p["progress"] for p in play])),
            "tpk_kl_2x2": tile_pattern_kl(filled_levels, real_contexts, n=2),
            "tpk_kl_3x3": tile_pattern_kl(filled_levels, real_contexts, n=3),
            "tpk_kl_4x4": tile_pattern_kl(filled_levels, real_contexts, n=4),
            "terrain_state_ngram_kl": terrain_state_ngram_kl(filled_levels, real_contexts, mask, n=3),
            "diversity": diversity(filled_levels),
            "masked_diversity": masked_diversity(filled_levels, mask),
            "playable_masked_diversity": playable_masked_diversity(filled_levels, mask),
            "jump_gap_progress": float(np.mean([p.get("gap_progress", p["progress"]) for p in play])),
            "seam_kl_left": float(np.mean([s["left"] for s in seams])),
            "seam_kl_right": float(np.mean([s["right"] for s in seams])),
            "struct_viol_per_col": float(np.mean([s["per_col"] for s in struct])),
            "context_tile_acc": float(np.mean([c["tile_acc"] for c in consistency])),
            "context_valid_rate": float(np.mean([c["valid_rate"] for c in consistency])),
            "rare_precision": float(np.mean([r["rare_precision"] for r in rare])),
            "rare_recall": float(np.mean([r["rare_recall"] for r in rare])),
            "rare_f1": float(np.mean([r["rare_f1"] for r in rare])),
            "object_recall": float(np.mean([r["object_recall"] for r in rare])),
        }
        style_rows = [style_metrics(filled, mask) for filled in filled_levels]
        for key in [
            "complete_structure_viol_per_col",
            "pipe_valid_rate",
            "pipe_pair_error_rate",
            "cannon_valid_rate",
            "ground_void_ratio_gap",
            "obstacle_density_gap",
            "style_class_match_rate",
            "descriptor_distance",
        ]:
            report[mask_name][key] = float(np.mean([row[key] for row in style_rows]))
        continuity_rows = [continuity_metrics(filled, mask) for filled in filled_levels]
        continuity_score = float(np.mean([row["continuity_score"] for row in continuity_rows]))
        for key in [
            "coin_run_continuation_rate",
            "platform_row_continuity",
            "ground_height_continuity",
            "boundary_pattern_kl",
            "continuity_score",
        ]:
            report[mask_name][key] = float(np.mean([row[key] for row in continuity_rows]))
        report[mask_name]["continuity_diversity_tradeoff"] = float(
            report[mask_name]["masked_diversity"] * (0.5 + 0.5 * continuity_score)
        )
        if mask_name == "inpaint_center":
            report[mask_name].update(right_context_sensitivity(model, real_contexts, mask, N=min(N, 2)))
    return report
