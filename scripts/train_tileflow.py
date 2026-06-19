"""Train and evaluate TileFlow categorical center-expansion prototypes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.common.data import DEFAULT_W, MAP_HEIGHT, SEED, VOCAB, load_level_windows, make_level_splits_with_eval_files
from tileflow.common.eval import (
    context_consistency,
    masked_hamming,
    playability,
    structural_violations,
    terrain_state_ngram_kl,
    tile_pattern_kl,
)
from tileflow.common.fill_api import assert_known_preserved
from tileflow.common.masks import center_expand
from tileflow.common.style import SOLID_TILES, SUPPORTABLE_TILES, style_descriptor, style_metrics
from tileflow.models.categorical_flow import (
    CONTINUATION_CONTEXT_RUN,
    CONTINUATION_LANDING_SPAN,
    CONTINUATION_NOISE,
    CONTINUATION_OTHER,
    CategoricalFlowNet,
    TERRAIN_AIR,
    TERRAIN_BAD_TOOTH,
    TERRAIN_GAP,
    TERRAIN_LANDING,
    TERRAIN_STABLE_GROUND,
    MOTIF_HIGH,
    MOTIF_LOW,
    MOTIF_MEDIUM,
    SKELETON_GAP,
    SKELETON_GROUND_RUN,
    SKELETON_HEIGHT_1,
    SKELETON_HEIGHT_2,
    SKELETON_HEIGHT_3,
    SKELETON_HEIGHT_4_PLUS,
    SKELETON_HEIGHT_MID_PLATFORM,
    SKELETON_HEIGHT_NONE,
    SKELETON_LANDING_ISLAND,
    SKELETON_MID_PLATFORM,
    SKELETON_RAISED_GROUND,
    SKELETON_STRUCTURE_ZONE,
    TileFlowConfig,
    TileFlowModel,
    checkpoint_payload,
    idx_to_level,
    level_to_idx,
    make_model_input,
)

FIXED_EVAL_FILES = ("mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt")
SOLID_CHARS = SOLID_TILES
SURFACE_ROWS = range(6, 14)
UTILITY_EMPTY = 0
UTILITY_ROUTE_SURFACE = 1
UTILITY_PLATFORM_BODY = 2
UTILITY_PLATFORM_EDGE = 3
UTILITY_SUPPORTED_ENTITY = 4
UTILITY_GAP_VOID = 5


def config_for_version(version: str, width: int) -> TileFlowConfig:
    if version == "v0.1":
        return TileFlowConfig(version=version, width=width, hidden=48, dilations=(1, 1, 1, 1), class_weighted_loss=False)
    if version == "v0.2":
        return TileFlowConfig(version=version, width=width, hidden=48, dilations=(1, 1, 1, 1), class_weighted_loss=True)
    if version == "v0.3":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 1, 2, 4),
            class_weighted_loss=False,
            position_channels=True,
        )
    if version == "v0.9":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
        )
    if version == "v0.12":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            structure_heads=True,
        )
    if version == "v0.13":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            structure_heads=True,
            utility_heads=True,
        )
    if version == "v0.14":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            structure_heads=True,
            utility_heads=True,
            context_channels=False,
            strict_utility_labels=True,
        )
    if version == "v0.15":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=64,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            structure_heads=True,
            utility_heads=True,
            context_channels=False,
            strict_utility_labels=True,
            terrain_heads=True,
            continuation_heads=True,
        )
    if version == "v1.0":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=72,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            context_channels=True,
            skeleton_heads=True,
            skeleton_conditioning=True,
            stochastic_decode=True,
        )
    if version == "v1.1":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=72,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            context_channels=True,
            structure_heads=True,
            skeleton_heads=True,
            skeleton_conditioning=True,
            stochastic_decode=True,
            support_conditioning=True,
            support_logit_bias=0.65,
        )
    if version == "v1.2":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=72,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            context_channels=True,
            structure_heads=True,
            skeleton_heads=True,
            skeleton_conditioning=True,
            stochastic_decode=True,
            support_conditioning=True,
            support_logit_bias=0.65,
            time_conditioning=True,
            dfm_target="x",
            dfm_schedule="linear",
            dfm_source="air",
            sample_steps=4,
        )
    if version == "v1.3":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=72,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            context_channels=True,
            structure_heads=True,
            skeleton_heads=True,
            skeleton_conditioning=True,
            stochastic_decode=True,
            support_conditioning=True,
            support_logit_bias=0.65,
            activation="gelu",
            time_conditioning=True,
            dfm_target="x",
            dfm_schedule="cosine",
            dfm_source="train_prior",
            sample_steps=4,
            gap_heavy_sample_steps=0,
            candidate_samples=2,
            adaptive_support_bias=False,
        )
    if version == "v1.4":
        return TileFlowConfig(
            version=version,
            width=width,
            hidden=72,
            dilations=(1, 2, 4, 8, 1, 2),
            class_weighted_loss=False,
            position_channels=True,
            context_channels=True,
            structure_heads=True,
            skeleton_heads=True,
            skeleton_conditioning=True,
            stochastic_decode=True,
            support_conditioning=True,
            support_logit_bias=0.65,
            activation="gelu",
            time_conditioning=True,
            dfm_target="x",
            dfm_schedule="cosine",
            dfm_source="train_prior",
            sample_steps=4,
            gap_heavy_sample_steps=0,
            candidate_samples=2,
            adaptive_support_bias=False,
        )
    raise ValueError(f"Unsupported TileFlow version: {version}")


def make_dev_split(train_files: list[str], val_count: int, seed: int) -> dict[str, list[str]]:
    if val_count <= 0:
        return {"train": sorted(train_files), "dev": []}
    if len(train_files) <= val_count:
        raise ValueError(f"val_count={val_count} leaves no training files from {len(train_files)} candidates.")
    rng = random.Random(seed)
    shuffled = list(train_files)
    rng.shuffle(shuffled)
    dev = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return {"train": train, "dev": dev}


class WindowDataset(Dataset):
    def __init__(self, levels: list[list[str]], mask: np.ndarray, config: TileFlowConfig) -> None:
        self.idx = [level_to_idx(level, width=mask.shape[1]) for level in levels]
        self.mask = mask
        self.inputs = [
            make_model_input(idx, mask, position_channels=config.position_channels, context_channels=config.context_channels)
            for idx in self.idx
        ]
        self.support_labels = [structure_support_labels(idx) for idx in self.idx]
        self.landing_labels = [structure_landing_labels(idx) for idx in self.idx]
        self.utility_labels = [structure_utility_labels(idx, strict_singletons=config.strict_utility_labels) for idx in self.idx]
        self.terrain_labels = [terrain_rhythm_labels(idx) for idx in self.idx]
        self.continuation_labels = [continuation_context_labels(idx, mask) for idx in self.idx]
        self.skeleton_labels = [skeleton_column_labels(idx, mask) for idx in self.idx]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        skeleton_state, skeleton_height, motif_budget = self.skeleton_labels[index]
        return (
            self.inputs[index],
            torch.as_tensor(self.idx[index], dtype=torch.long),
            torch.as_tensor(self.support_labels[index], dtype=torch.long),
            torch.as_tensor(self.landing_labels[index], dtype=torch.long),
            torch.as_tensor(self.utility_labels[index], dtype=torch.long),
            torch.as_tensor(self.terrain_labels[index], dtype=torch.long),
            torch.as_tensor(self.continuation_labels[index], dtype=torch.long),
            torch.as_tensor(skeleton_state, dtype=torch.long),
            torch.as_tensor(skeleton_height, dtype=torch.long),
            torch.as_tensor(motif_budget, dtype=torch.long),
        )


def structure_support_labels(idx: np.ndarray) -> np.ndarray:
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    supportable_ids = {VOCAB.index(ch) for ch in SUPPORTABLE_TILES | {"E"} if ch in VOCAB}
    air_id = VOCAB.index("-")
    labels = np.zeros_like(idx, dtype=np.int64)
    for r in range(idx.shape[0]):
        solid_run = 0
        run_start = 0
        for c in range(idx.shape[1] + 1):
            is_solid = c < idx.shape[1] and int(idx[r, c]) in solid_ids
            if is_solid:
                if solid_run == 0:
                    run_start = c
                solid_run += 1
                continue
            if solid_run:
                for rc in range(run_start, c):
                    above_air = r == 0 or int(idx[r - 1, rc]) == air_id
                    below_object = r > 0 and int(idx[r - 1, rc]) in supportable_ids
                    if below_object:
                        labels[r, rc] = 3
                    elif r in SURFACE_ROWS and solid_run >= 2 and above_air:
                        labels[r, rc] = 2
                    else:
                        labels[r, rc] = 1
                solid_run = 0
    return labels


def structure_landing_labels(idx: np.ndarray) -> np.ndarray:
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    grounded = [
        int(idx[idx.shape[0] - 1, c]) in solid_ids or int(idx[idx.shape[0] - 2, c]) in solid_ids
        for c in range(idx.shape[1])
    ]
    labels = np.zeros(idx.shape[1], dtype=np.int64)
    for c, is_grounded in enumerate(grounded):
        labels[c] = 2 if is_grounded else 0
    run: list[int] = []
    for c, is_grounded in enumerate(grounded + [True]):
        if not is_grounded:
            run.append(c)
            continue
        if run and len(run) <= 4:
            left = run[0] - 1
            right = run[-1] + 1
            if 0 <= left < idx.shape[1] and grounded[left]:
                labels[left] = 1
            if 0 <= right < idx.shape[1] and grounded[right]:
                labels[right] = 1
        run = []
    return labels


def structure_utility_labels(idx: np.ndarray, strict_singletons: bool = False) -> np.ndarray:
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    platform_ids = {VOCAB.index(ch) for ch in "XSQ?" if ch in VOCAB}
    entity_ids = {VOCAB.index(ch) for ch in "E<>[]Bb" if ch in VOCAB}
    air_id = VOCAB.index("-")
    labels = np.zeros_like(idx, dtype=np.int64)

    ground_solid = [
        int(idx[idx.shape[0] - 1, c]) in solid_ids or int(idx[idx.shape[0] - 2, c]) in solid_ids
        for c in range(idx.shape[1])
    ]
    for c, grounded in enumerate(ground_solid):
        if not grounded:
            labels[idx.shape[0] - 2 :, c] = UTILITY_GAP_VOID

    for r in range(idx.shape[0]):
        c = 0
        while c < idx.shape[1]:
            if int(idx[r, c]) not in platform_ids:
                c += 1
                continue
            start = c
            while c < idx.shape[1] and int(idx[r, c]) in platform_ids:
                c += 1
            end = c
            run_len = end - start
            for cc in range(start, end):
                above_air = r == 0 or int(idx[r - 1, cc]) == air_id
                if not above_air:
                    continue
                below_supported = r + 1 >= idx.shape[0] or int(idx[r + 1, cc]) in solid_ids
                isolated_midair = run_len == 1 and r in range(3, idx.shape[0] - 2) and not below_supported
                if strict_singletons and isolated_midair:
                    continue
                if cc == start or cc == end - 1:
                    if strict_singletons and run_len == 1 and not below_supported:
                        continue
                    labels[r, cc] = UTILITY_PLATFORM_EDGE
                elif r in range(4, 10) and run_len >= 2:
                    labels[r, cc] = UTILITY_PLATFORM_BODY
                else:
                    labels[r, cc] = UTILITY_ROUTE_SURFACE

    for r in range(idx.shape[0]):
        for c in range(idx.shape[1]):
            if int(idx[r, c]) not in entity_ids:
                continue
            below_supported = r + 1 >= idx.shape[0] or int(idx[r + 1, c]) in solid_ids
            paired_pipe = int(idx[r, c]) in {VOCAB.index(ch) for ch in "<>[]" if ch in VOCAB}
            if below_supported or paired_pipe:
                labels[r, c] = UTILITY_SUPPORTED_ENTITY

    return labels


def bottom_stack_heights(idx: np.ndarray) -> list[int]:
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    heights: list[int] = []
    for c in range(idx.shape[1]):
        height = 0
        for r in range(idx.shape[0] - 1, -1, -1):
            if int(idx[r, c]) not in solid_ids:
                break
            height += 1
        heights.append(height)
    return heights


def terrain_rhythm_labels(idx: np.ndarray) -> np.ndarray:
    """Soft labels for bottom rhythm; bad teeth are discouraged, not forbidden."""
    labels = np.full(idx.shape, TERRAIN_AIR, dtype=np.int64)
    heights = bottom_stack_heights(idx)
    grounded = [height > 0 for height in heights]

    for c, height in enumerate(heights):
        if height <= 0:
            labels[idx.shape[0] - 2 :, c] = TERRAIN_GAP
            continue
        start = idx.shape[0] - height
        labels[start:, c] = TERRAIN_STABLE_GROUND
        left_gap = c == 0 or not grounded[c - 1]
        right_gap = c + 1 >= idx.shape[1] or not grounded[c + 1]
        if left_gap or right_gap:
            labels[start:, c] = TERRAIN_LANDING

    for c in range(1, idx.shape[1] - 1):
        left_h = heights[c - 1]
        current_h = heights[c]
        right_h = heights[c + 1]
        if current_h > 0 and left_h == 0 and right_h == 0:
            labels[idx.shape[0] - current_h :, c] = TERRAIN_BAD_TOOTH
            continue
        if current_h == 0 and left_h > 0 and right_h > 0:
            labels[idx.shape[0] - 2 :, c] = TERRAIN_BAD_TOOTH
            continue
        if current_h > 0 and left_h > 0 and right_h > 0:
            is_spike = current_h >= max(left_h, right_h) + 2
            is_dent = current_h <= min(left_h, right_h) - 2
            if is_spike or is_dent:
                labels[idx.shape[0] - current_h :, c] = TERRAIN_BAD_TOOTH

    return labels


def _unknown_regions(mask: np.ndarray) -> list[list[int]]:
    regions: list[list[int]] = []
    current: list[int] = []
    for c in range(mask.shape[1]):
        if bool(mask[:, c].any()):
            if current:
                regions.append(current)
                current = []
            continue
        current.append(c)
    if current:
        regions.append(current)
    return regions


def _center_platform_rows(idx: np.ndarray, mask: np.ndarray) -> list[int]:
    known_cols = [c for c in range(mask.shape[1]) if bool(mask[:, c].any())]
    if not known_cols:
        return []
    platform_ids = {VOCAB.index(ch) for ch in "XSQ?" if ch in VOCAB}
    scored: list[tuple[int, float]] = []
    for r in range(3, idx.shape[0]):
        density = sum(int(idx[r, c]) in platform_ids for c in known_cols) / len(known_cols)
        if density >= 0.08:
            scored.append((r, density))
    return [row for row, _ in sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)[:4]]


def continuation_context_labels(idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels = np.full(idx.shape, CONTINUATION_OTHER, dtype=np.int64)
    platform_ids = {VOCAB.index(ch) for ch in "XSQ?" if ch in VOCAB}
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    air_id = VOCAB.index("-")
    center_rows = _center_platform_rows(idx, mask)
    heights = bottom_stack_heights(idx)

    for cols in _unknown_regions(mask):
        for r in range(3, idx.shape[0]):
            c_i = 0
            while c_i < len(cols):
                c = cols[c_i]
                if int(idx[r, c]) not in platform_ids:
                    c_i += 1
                    continue
                start_i = c_i
                while c_i < len(cols) and int(idx[r, cols[c_i]]) in platform_ids:
                    c_i += 1
                run_cols = cols[start_i:c_i]
                if len(run_cols) >= 2 and any(abs(r - row) <= 1 for row in center_rows):
                    for rc in run_cols:
                        labels[r, rc] = CONTINUATION_CONTEXT_RUN

        run: list[int] = []
        for c in cols + [-1]:
            is_grounded = c != -1 and heights[c] > 0
            if is_grounded:
                run.append(c)
                continue
            if len(run) >= 2:
                adjacent_gap = (run[0] - 1 >= 0 and heights[run[0] - 1] == 0) or (
                    run[-1] + 1 < idx.shape[1] and heights[run[-1] + 1] == 0
                )
                if adjacent_gap or len(run) <= 8:
                    for rc in run:
                        labels[idx.shape[0] - heights[rc] :, rc] = CONTINUATION_LANDING_SPAN
            run = []

    for r in range(2, idx.shape[0] - 1):
        for c in range(idx.shape[1]):
            if mask[r, c] or int(idx[r, c]) == air_id:
                continue
            tile = int(idx[r, c])
            if tile in solid_ids:
                horizontal = (c > 0 and int(idx[r, c - 1]) in solid_ids) or (
                    c + 1 < idx.shape[1] and int(idx[r, c + 1]) in solid_ids
                )
                supported = int(idx[r + 1, c]) in solid_ids
                context_aligned = any(abs(r - row) <= 1 for row in center_rows)
                if not horizontal and not supported and not context_aligned:
                    labels[r, c] = CONTINUATION_NOISE
            elif tile != air_id:
                supported = int(idx[r + 1, c]) in solid_ids
                context_aligned = any(abs(r - row) <= 1 for row in center_rows)
                if not supported and not context_aligned:
                    labels[r, c] = CONTINUATION_NOISE

    return labels


def _solid_run_lengths(values: list[bool]) -> dict[int, int]:
    run_lengths: dict[int, int] = {}
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        if values[start]:
            for i in range(start, end):
                run_lengths[i] = end - start
        start = end
    return run_lengths


def skeleton_column_labels(idx: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-level v1.0 structure labels derived from soft Mario grammar."""
    width = idx.shape[1]
    solid_ids = {VOCAB.index(ch) for ch in SOLID_CHARS if ch in VOCAB}
    platform_ids = {VOCAB.index(ch) for ch in "XSQ?" if ch in VOCAB}
    obstacle_ids = {VOCAB.index(ch) for ch in "SQ?E<>[]Bbo" if ch in VOCAB}
    heights = bottom_stack_heights(idx)
    grounded = [height > 0 for height in heights]
    ground_run_lengths = _solid_run_lengths(grounded)

    states = np.full(width, SKELETON_GAP, dtype=np.int64)
    surface_heights = np.full(width, SKELETON_HEIGHT_NONE, dtype=np.int64)
    motif_budget = np.full(width, MOTIF_LOW, dtype=np.int64)

    for c in range(width):
        column_obstacles = sum(int(idx[r, c]) in obstacle_ids for r in range(idx.shape[0]))
        local_obstacles = 0
        for cc in range(max(0, c - 1), min(width, c + 2)):
            local_obstacles += sum(int(idx[r, cc]) in obstacle_ids for r in range(idx.shape[0]))
        mid_platform = any(int(idx[r, c]) in platform_ids for r in range(3, idx.shape[0] - 3))
        height = heights[c]

        if height <= 0:
            state = SKELETON_MID_PLATFORM if mid_platform else SKELETON_GAP
            height_label = SKELETON_HEIGHT_MID_PLATFORM if mid_platform else SKELETON_HEIGHT_NONE
        else:
            run_len = ground_run_lengths.get(c, 1)
            adjacent_gap = (c > 0 and heights[c - 1] == 0) or (c + 1 < width and heights[c + 1] == 0)
            if column_obstacles > 0:
                state = SKELETON_STRUCTURE_ZONE
            elif mid_platform and height <= 1:
                state = SKELETON_MID_PLATFORM
            elif adjacent_gap or run_len <= 3:
                state = SKELETON_LANDING_ISLAND
            elif height >= 3:
                state = SKELETON_RAISED_GROUND
            else:
                state = SKELETON_GROUND_RUN

            if mid_platform and height <= 1:
                height_label = SKELETON_HEIGHT_MID_PLATFORM
            elif height == 1:
                height_label = SKELETON_HEIGHT_1
            elif height == 2:
                height_label = SKELETON_HEIGHT_2
            elif height == 3:
                height_label = SKELETON_HEIGHT_3
            else:
                height_label = SKELETON_HEIGHT_4_PLUS

        if local_obstacles >= 5:
            motif = MOTIF_HIGH
        elif local_obstacles >= 2:
            motif = MOTIF_MEDIUM
        else:
            motif = MOTIF_LOW

        states[c] = state
        surface_heights[c] = height_label
        motif_budget[c] = motif

    return states, surface_heights, motif_budget


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(dataset: WindowDataset, mask: np.ndarray) -> torch.Tensor:
    counts = np.ones(len(VOCAB), dtype=np.float64)
    unknown = ~mask
    for idx in dataset.idx:
        values, value_counts = np.unique(idx[unknown], return_counts=True)
        counts[values] += value_counts
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def source_probs(dataset: WindowDataset, mask: np.ndarray) -> tuple[float, ...]:
    counts = np.ones(len(VOCAB), dtype=np.float64)
    unknown = ~mask
    for idx in dataset.idx:
        values, value_counts = np.unique(idx[unknown], return_counts=True)
        counts[values] += value_counts
    probs = counts / counts.sum()
    return tuple(float(value) for value in probs)


def source_style_probs(dataset: WindowDataset, mask: np.ndarray) -> tuple[tuple[float, ...], ...]:
    style_index = {"gap-heavy": 0, "obstacle-heavy": 1, "plain/low-obstacle": 2}
    counts = np.ones((3, len(VOCAB)), dtype=np.float64)
    unknown = ~mask
    for idx in dataset.idx:
        style = style_descriptor(idx_to_level(idx), mask).style_class
        row = style_index.get(style, 2)
        values, value_counts = np.unique(idx[unknown], return_counts=True)
        counts[row, values] += value_counts
    probs = counts / counts.sum(axis=1, keepdims=True)
    return tuple(tuple(float(value) for value in row) for row in probs)


def dfm_alpha(t: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "cosine":
        return torch.sin(t * torch.pi / 2.0).pow(2)
    if schedule == "sqrt":
        return torch.sqrt(t.clamp_min(1e-4))
    if schedule == "linear":
        return t
    raise ValueError(f"Unsupported DFM schedule: {schedule}")


def make_dfm_training_batch(
    base_x: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: TileFlowConfig,
    source_prob: torch.Tensor | None,
    source_style_prob: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, height, width = target.shape
    device = target.device
    t = torch.rand((batch,), generator=generator, device=device).clamp(0.02, 0.98)
    alpha = dfm_alpha(t, config.dfm_schedule).view(batch, 1, 1)
    unknown = (~mask).unsqueeze(0).expand(batch, height, width)
    reveal = torch.rand(target.shape, generator=generator, device=device) < alpha

    if config.dfm_source == "uniform":
        source = torch.randint(0, len(VOCAB), target.shape, generator=generator, device=device)
    elif config.dfm_source == "train_prior" and source_prob is not None:
        source = torch.multinomial(source_prob, num_samples=batch * height * width, replacement=True, generator=generator)
        source = source.view(batch, height, width).to(device)
    elif config.dfm_source == "style_prior" and source_style_prob is not None:
        if not config.context_channels:
            raise ValueError("style_prior source requires context channels.")
        context_base = len(VOCAB) + 2 + (2 if config.position_channels else 0)
        gap_heavy = base_x[:, context_base + 12, 0, 0] >= 0.5
        obstacle_heavy = base_x[:, context_base + 13, 0, 0] >= 0.5
        style_ids = torch.where(gap_heavy, 0, torch.where(obstacle_heavy, 1, 2)).to(torch.long)
        source_rows = []
        for style_id in style_ids.tolist():
            probs = source_style_prob[int(style_id)]
            sampled = torch.multinomial(probs, num_samples=height * width, replacement=True, generator=generator)
            source_rows.append(sampled.view(height, width))
        source = torch.stack(source_rows, dim=0).to(device)
    elif config.dfm_source == "gap_style_prior" and source_prob is not None and source_style_prob is not None:
        if not config.context_channels:
            raise ValueError("gap_style_prior source requires context channels.")
        context_base = len(VOCAB) + 2 + (2 if config.position_channels else 0)
        gap_heavy = base_x[:, context_base + 12, 0, 0] >= 0.5
        source_rows = []
        for is_gap in gap_heavy.tolist():
            probs = source_style_prob[0] if bool(is_gap) else source_prob
            sampled = torch.multinomial(probs, num_samples=height * width, replacement=True, generator=generator)
            source_rows.append(sampled.view(height, width))
        source = torch.stack(source_rows, dim=0).to(device)
    else:
        source = torch.full_like(target, VOCAB.index("-"))

    corrupted = target.clone()
    corrupted[unknown & ~reveal] = source[unknown & ~reveal]
    onehot = nn.functional.one_hot(corrupted, num_classes=len(VOCAB)).permute(0, 3, 1, 2).float()

    dfm_x = base_x.clone()
    dfm_x[:, : len(VOCAB)] = onehot
    t_plane = t.view(batch, 1, 1, 1).expand(batch, 1, height, width)
    dfm_x = torch.cat([dfm_x, t_plane, 1.0 - t_plane], dim=1)
    return dfm_x, source, reveal, t


def unknown_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    per_cell = nn.functional.cross_entropy(logits, target, weight=weight, reduction="none")
    unknown = (~mask).unsqueeze(0).expand_as(per_cell)
    return per_cell[unknown].mean()


def unknown_aux_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    per_cell = nn.functional.cross_entropy(logits, target, weight=weight, reduction="none")
    unknown = (~mask).unsqueeze(0).expand_as(per_cell)
    return per_cell[unknown].mean()


def unknown_column_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    pooled = logits.mean(dim=2)
    per_col = nn.functional.cross_entropy(pooled, target, weight=weight, reduction="none")
    unknown_cols = (~mask.any(dim=0)).unsqueeze(0).expand_as(per_col)
    return per_col[unknown_cols].mean()


def decoder_support_group_loss(logits: torch.Tensor, support_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = [
        [VOCAB.index("-")],
        [VOCAB.index(ch) for ch in "XSQ?<>[]Bb" if ch in VOCAB],
        [VOCAB.index(ch) for ch in "XSQ?" if ch in VOCAB],
        [VOCAB.index(ch) for ch in "E<>[]Bb" if ch in VOCAB],
    ]
    group_logits = torch.cat([torch.logsumexp(logits[:, ids], dim=1, keepdim=True) for ids in groups], dim=1)
    return unknown_aux_ce_loss(group_logits, support_target, mask, weight=None)


def dfm_tile_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    reveal: torch.Tensor,
    support_target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
    config: TileFlowConfig,
) -> torch.Tensor:
    if config.dfm_target == "noise":
        mixed_target = target.clone()
        unknown = (~mask).unsqueeze(0).expand_as(target)
        mixed_target[unknown & ~reveal] = source[unknown & ~reveal]
        return unknown_ce_loss(logits, mixed_target, mask, weight)
    loss = unknown_ce_loss(logits, target, mask, weight)
    if config.dfm_target == "support":
        loss = loss + 0.18 * decoder_support_group_loss(logits, support_target, mask)
    elif config.dfm_target != "x":
        raise ValueError(f"Unsupported DFM target: {config.dfm_target}")
    return loss


def generated_utility_metrics(level: list[str], mask: np.ndarray) -> dict[str, float]:
    rows = [list(row) for row in level]
    width = mask.shape[1]
    unknown_cols = [c for c in range(width) if not bool(mask[:, c].any())]
    if not unknown_cols:
        return {
            "utility_surface_col_rate": 1.0,
            "utility_empty_col_rate": 0.0,
            "utility_long_gap_rate": 0.0,
            "utility_score": 1.0,
        }
    solid = set(SOLID_CHARS)
    surface_cols = 0
    empty_cols = 0
    long_gap_cols = 0
    gap_run: list[int] = []
    for c in unknown_cols + [-1]:
        if c == -1:
            grounded = True
            has_content = True
            has_surface = False
        else:
            has_content = any(rows[r][c] != "-" for r in range(len(rows)))
            has_surface = any(rows[r][c] in solid and r > 0 and rows[r - 1][c] == "-" for r in range(1, len(rows)))
            grounded = rows[-1][c] in solid or rows[-2][c] in solid
        empty_cols += int(not has_content)
        surface_cols += int(has_surface)
        if not grounded and c != -1:
            gap_run.append(c)
            continue
        if len(gap_run) > 4:
            long_gap_cols += len(gap_run)
        gap_run = []
    surface_rate = surface_cols / len(unknown_cols)
    empty_rate = empty_cols / len(unknown_cols)
    long_gap_rate = long_gap_cols / len(unknown_cols)
    utility_score = max(0.0, min(1.0, surface_rate - 0.45 * empty_rate - 0.55 * long_gap_rate))
    return {
        "utility_surface_col_rate": float(surface_rate),
        "utility_empty_col_rate": float(empty_rate),
        "utility_long_gap_rate": float(long_gap_rate),
        "utility_score": float(utility_score),
    }


def generated_coherence_metrics(level: list[str], mask: np.ndarray) -> dict[str, float]:
    rows = [list(row) for row in level]
    width = mask.shape[1]
    solid = set(SOLID_CHARS)
    supportable = set("E<>[]Bb")
    singleton = total_solid = unsupported = total_supportable = short_runs = total_runs = 0
    for r in range(MAP_HEIGHT):
        run_len = 0
        run_generated = False
        for c in range(width + 1):
            in_bounds = c < width
            is_unknown = in_bounds and not mask[r, c]
            is_solid = in_bounds and rows[r][c] in solid
            if is_unknown and is_solid:
                run_len += 1
                run_generated = True
            else:
                if run_generated:
                    total_runs += 1
                    if run_len == 1:
                        short_runs += 1
                run_len = 0
                run_generated = False
            if not is_unknown:
                continue
            ch = rows[r][c]
            if ch in solid:
                total_solid += 1
                horizontal = (c > 0 and rows[r][c - 1] in solid) or (c + 1 < width and rows[r][c + 1] in solid)
                supported = r + 1 >= MAP_HEIGHT or rows[r + 1][c] in solid
                if not horizontal and not supported:
                    singleton += 1
            if ch in supportable:
                total_supportable += 1
                if r + 1 < MAP_HEIGHT and rows[r + 1][c] not in solid:
                    unsupported += 1
    return {
        "coherence_singleton_rate": float(singleton / max(total_solid, 1)),
        "coherence_short_run_rate": float(short_runs / max(total_runs, 1)),
        "coherence_unsupported_rate": float(unsupported / max(total_supportable, 1)),
    }


def generated_terrain_metrics(level: list[str], mask: np.ndarray) -> dict[str, float]:
    rows = [list(row) for row in level]
    width = mask.shape[1]
    solid = set(SOLID_CHARS)
    unknown_cols = [c for c in range(width) if not bool(mask[:, c].any())]
    if not unknown_cols:
        return {
            "terrain_tooth_rate": 0.0,
            "terrain_one_col_hole_rate": 0.0,
            "terrain_one_col_island_rate": 0.0,
            "terrain_height_flicker_rate": 0.0,
            "terrain_ground_run_mean": 0.0,
            "terrain_gap_run_mean": 0.0,
        }

    heights: list[int] = []
    for c in range(width):
        height = 0
        for r in range(MAP_HEIGHT - 1, -1, -1):
            if rows[r][c] not in solid:
                break
            height += 1
        heights.append(height)

    holes = islands = flickers = 0
    for c in unknown_cols:
        left_h = heights[c - 1] if c > 0 else heights[c]
        current_h = heights[c]
        right_h = heights[c + 1] if c + 1 < width else heights[c]
        if current_h == 0 and left_h > 0 and right_h > 0:
            holes += 1
        if current_h > 0 and left_h == 0 and right_h == 0:
            islands += 1
        if c > 0 and c + 1 < width and current_h > 0 and left_h > 0 and right_h > 0:
            is_spike = current_h >= max(left_h, right_h) + 2
            is_dent = current_h <= min(left_h, right_h) - 2
            flickers += int(is_spike or is_dent)

    ground_runs: list[int] = []
    gap_runs: list[int] = []
    for cols in _unknown_regions(mask):
        run_len = 0
        run_ground = None
        for c in cols + [-1]:
            is_ground = c != -1 and heights[c] > 0
            if run_ground is None:
                run_ground = is_ground
                run_len = 1 if c != -1 else 0
                continue
            if c != -1 and is_ground == run_ground:
                run_len += 1
                continue
            if run_len:
                if run_ground:
                    ground_runs.append(run_len)
                else:
                    gap_runs.append(run_len)
            run_ground = is_ground if c != -1 else None
            run_len = 1 if c != -1 else 0

    denom = max(len(unknown_cols), 1)
    tooth_rate = min(1.0, (holes + islands + flickers) / denom)
    return {
        "terrain_tooth_rate": float(tooth_rate),
        "terrain_one_col_hole_rate": float(holes / denom),
        "terrain_one_col_island_rate": float(islands / denom),
        "terrain_height_flicker_rate": float(flickers / denom),
        "terrain_ground_run_mean": float(np.mean(ground_runs) if ground_runs else 0.0),
        "terrain_gap_run_mean": float(np.mean(gap_runs) if gap_runs else 0.0),
    }


@torch.no_grad()
def reconstruction_metrics(fill_model: TileFlowModel, eval_levels: list[list[str]], mask: np.ndarray) -> dict[str, float]:
    consistency = []
    play = []
    struct = []
    utility = []
    coherence = []
    terrain = []
    style = []
    filled_levels = []
    intra_context_distances = []
    for level in eval_levels:
        filled = fill_model.fill(level, mask)
        if getattr(fill_model.config, "stochastic_decode", False):
            alternate = fill_model.fill(level, mask)
            intra_context_distances.append(masked_hamming(filled, alternate, mask))
        assert_known_preserved(level, filled, mask)
        filled_levels.append(filled)
        consistency.append(context_consistency(filled, level, mask))
        play.append(playability(filled))
        struct.append(structural_violations(filled))
        utility.append(generated_utility_metrics(filled, mask))
        coherence.append(generated_coherence_metrics(filled, mask))
        terrain.append(generated_terrain_metrics(filled, mask))
        style.append(style_metrics(filled, mask))
    return {
        "eval_unknown_tile_acc": float(np.mean([row["tile_acc"] for row in consistency])),
        "eval_context_valid_rate": float(np.mean([row["valid_rate"] for row in consistency])),
        "eval_completable_rate": float(np.mean([row["completable"] for row in play])),
        "eval_progress_mean": float(np.mean([row["progress"] for row in play])),
        "eval_jump_gap_progress": float(np.mean([row.get("gap_progress", row["progress"]) for row in play])),
        "eval_tpk_kl_3x3": tile_pattern_kl(filled_levels, eval_levels, n=3),
        "eval_tpk_kl_4x4": tile_pattern_kl(filled_levels, eval_levels, n=4),
        "eval_terrain_state_ngram_kl": terrain_state_ngram_kl(filled_levels, eval_levels, mask, n=3),
        "eval_intra_context_diversity": float(np.mean(intra_context_distances)) if intra_context_distances else 0.0,
        "eval_struct_viol_per_col": float(np.mean([row["per_col"] for row in struct])),
        "eval_utility_surface_col_rate": float(np.mean([row["utility_surface_col_rate"] for row in utility])),
        "eval_utility_empty_col_rate": float(np.mean([row["utility_empty_col_rate"] for row in utility])),
        "eval_utility_long_gap_rate": float(np.mean([row["utility_long_gap_rate"] for row in utility])),
        "eval_utility_score": float(np.mean([row["utility_score"] for row in utility])),
        "eval_coherence_singleton_rate": float(np.mean([row["coherence_singleton_rate"] for row in coherence])),
        "eval_coherence_short_run_rate": float(np.mean([row["coherence_short_run_rate"] for row in coherence])),
        "eval_coherence_unsupported_rate": float(np.mean([row["coherence_unsupported_rate"] for row in coherence])),
        "eval_terrain_tooth_rate": float(np.mean([row["terrain_tooth_rate"] for row in terrain])),
        "eval_terrain_one_col_hole_rate": float(np.mean([row["terrain_one_col_hole_rate"] for row in terrain])),
        "eval_terrain_one_col_island_rate": float(np.mean([row["terrain_one_col_island_rate"] for row in terrain])),
        "eval_terrain_height_flicker_rate": float(np.mean([row["terrain_height_flicker_rate"] for row in terrain])),
        "eval_terrain_ground_run_mean": float(np.mean([row["terrain_ground_run_mean"] for row in terrain])),
        "eval_terrain_gap_run_mean": float(np.mean([row["terrain_gap_run_mean"] for row in terrain])),
        "eval_descriptor_distance": float(np.mean([row["descriptor_distance"] for row in style])),
        "eval_descriptor_adherence": float(np.mean([1.0 / (1.0 + row["descriptor_distance"]) for row in style])),
        "eval_style_class_match_rate": float(np.mean([row["style_class_match_rate"] for row in style])),
        "eval_ground_void_ratio_gap": float(np.mean([row["ground_void_ratio_gap"] for row in style])),
        "eval_pipe_pair_error_rate": float(np.mean([row["pipe_pair_error_rate"] for row in style])),
        "eval_overlong_block_run_rate": float(np.mean([row["overlong_block_run_rate"] for row in style])),
        "eval_bulky_mass_rate": float(np.mean([row["bulky_mass_rate"] for row in style])),
        "eval_sky_mass_rate": float(np.mean([row["sky_mass_rate"] for row in style])),
        "eval_sparse_context_mismatch": float(np.mean([row["sparse_context_mismatch"] for row in style])),
    }


SKELETON_STATE_NAMES = {
    SKELETON_GAP: "gap",
    SKELETON_GROUND_RUN: "ground_run",
    SKELETON_RAISED_GROUND: "raised_ground",
    SKELETON_LANDING_ISLAND: "landing_island",
    SKELETON_MID_PLATFORM: "mid_platform",
    SKELETON_STRUCTURE_ZONE: "structure_zone",
}
SKELETON_HEIGHT_NAMES = {
    SKELETON_HEIGHT_NONE: "none",
    SKELETON_HEIGHT_1: "height_1",
    SKELETON_HEIGHT_2: "height_2",
    SKELETON_HEIGHT_3: "height_3",
    SKELETON_HEIGHT_4_PLUS: "height_4_plus",
    SKELETON_HEIGHT_MID_PLATFORM: "mid_platform",
}
MOTIF_NAMES = {
    MOTIF_LOW: "low",
    MOTIF_MEDIUM: "medium",
    MOTIF_HIGH: "high",
}


def _normalized_counts(counter: Counter, names: dict[int, str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {name: 0.0 for name in names.values()}
    return {name: float(counter.get(index, 0) / total) for index, name in names.items()}


def _counter_kl(p: Counter, q: Counter, names: dict[int, str], eps: float = 1e-6) -> float:
    keys = set(names)
    p_total = sum(p.values()) + eps * len(keys)
    q_total = sum(q.values()) + eps * len(keys)
    return float(
        sum(
            ((p[k] + eps) / p_total) * np.log(((p[k] + eps) / p_total) / ((q[k] + eps) / q_total))
            for k in keys
        )
    )


def _skeleton_summary(real: Counter, predicted: Counter, sampled: Counter, names: dict[int, str]) -> dict:
    return {
        "real": _normalized_counts(real, names),
        "predicted": _normalized_counts(predicted, names),
        "sampled": _normalized_counts(sampled, names),
        "predicted_kl": _counter_kl(real, predicted, names),
        "sampled_kl": _counter_kl(real, sampled, names),
    }


@torch.no_grad()
def skeleton_distribution_audit(
    model: CategoricalFlowNet,
    records,
    mask: np.ndarray,
    config: TileFlowConfig,
    device: torch.device,
    seed: int,
) -> dict:
    if not config.skeleton_heads:
        return {}
    unknown_cols = np.where((~mask).any(axis=0))[0]
    grouped: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    rng = torch.Generator(device=rng_device)
    rng.manual_seed(seed + 303)
    model.eval()

    def add_counts(group: str, key: str, values: np.ndarray) -> None:
        grouped[group][key].update(int(value) for value in values[unknown_cols])

    for record in records:
        idx = level_to_idx(record.level, width=mask.shape[1])
        real_state, real_height, real_motif = skeleton_column_labels(idx, mask)
        x = make_model_input(
            idx,
            mask,
            device=device,
            position_channels=config.position_channels,
            context_channels=config.context_channels,
            time_value=1.0 if config.time_conditioning else None,
        ).unsqueeze(0)
        outputs = model.forward_all(x)
        sampled = model.forward_sampled_skeleton(x, generator=rng)
        predicted_state = outputs["skeleton_state"].argmax(dim=1).squeeze().detach().cpu().numpy()
        predicted_height = outputs["skeleton_height"].argmax(dim=1).squeeze().detach().cpu().numpy()
        predicted_motif = outputs["motif_budget"].argmax(dim=1).squeeze().detach().cpu().numpy()
        sampled_state = sampled["sampled_skeleton_state"].squeeze(0).detach().cpu().numpy()
        sampled_height = sampled["sampled_skeleton_height"].squeeze(0).detach().cpu().numpy()
        sampled_motif = sampled["sampled_motif_budget"].squeeze(0).detach().cpu().numpy()

        for group in ("all", record.source):
            add_counts(group, "real_state", real_state)
            add_counts(group, "predicted_state", predicted_state)
            add_counts(group, "sampled_state", sampled_state)
            add_counts(group, "real_height", real_height)
            add_counts(group, "predicted_height", predicted_height)
            add_counts(group, "sampled_height", sampled_height)
            add_counts(group, "real_motif", real_motif)
            add_counts(group, "predicted_motif", predicted_motif)
            add_counts(group, "sampled_motif", sampled_motif)

    report = {}
    for group, counters in sorted(grouped.items()):
        state_summary = _skeleton_summary(
            counters["real_state"],
            counters["predicted_state"],
            counters["sampled_state"],
            SKELETON_STATE_NAMES,
        )
        height_summary = _skeleton_summary(
            counters["real_height"],
            counters["predicted_height"],
            counters["sampled_height"],
            SKELETON_HEIGHT_NAMES,
        )
        motif_summary = _skeleton_summary(
            counters["real_motif"],
            counters["predicted_motif"],
            counters["sampled_motif"],
            MOTIF_NAMES,
        )
        report[group] = {
            "state": state_summary,
            "height": height_summary,
            "motif": motif_summary,
            "mean_sampled_kl": float(
                np.mean(
                    [
                        state_summary["sampled_kl"],
                        height_summary["sampled_kl"],
                        motif_summary["sampled_kl"],
                    ]
                )
            ),
        }
    return report


def selection_score(metrics: dict[str, float], version: str) -> float:
    if version == "v1.4":
        return (
            0.30 * metrics["eval_progress_mean"]
            + 0.24 * metrics["eval_completable_rate"]
            + 0.10 * metrics["eval_utility_score"]
            + 0.12 * metrics["eval_jump_gap_progress"]
            + 0.10 * metrics["eval_intra_context_diversity"]
            + 0.10 * metrics["eval_descriptor_adherence"]
            + 0.06 * metrics["eval_style_class_match_rate"]
            + 0.06 * metrics["eval_context_valid_rate"]
            - 0.38 * metrics["eval_struct_viol_per_col"]
            - 0.18 * metrics["eval_pipe_pair_error_rate"]
            - 0.16 * metrics["eval_coherence_singleton_rate"]
            - 0.14 * metrics["eval_coherence_unsupported_rate"]
            - 0.08 * metrics["eval_utility_long_gap_rate"]
            - 0.05 * metrics["eval_terrain_tooth_rate"]
            - 0.08 * metrics["eval_overlong_block_run_rate"]
            - 0.08 * metrics["eval_bulky_mass_rate"]
            - 0.05 * metrics["eval_sky_mass_rate"]
            - 0.06 * metrics["eval_sparse_context_mismatch"]
            - 0.03 * metrics["eval_tpk_kl_3x3"]
            - 0.02 * metrics["eval_tpk_kl_4x4"]
            - 0.03 * metrics["eval_terrain_state_ngram_kl"]
        )
    if version == "v1.3":
        return (
            0.30 * metrics["eval_progress_mean"]
            + 0.22 * metrics["eval_completable_rate"]
            + 0.12 * metrics["eval_utility_score"]
            + 0.12 * metrics["eval_jump_gap_progress"]
            + 0.08 * metrics["eval_intra_context_diversity"]
            + 0.10 * metrics["eval_descriptor_adherence"]
            + 0.05 * metrics["eval_style_class_match_rate"]
            + 0.06 * metrics["eval_context_valid_rate"]
            - 0.36 * metrics["eval_struct_viol_per_col"]
            - 0.16 * metrics["eval_coherence_singleton_rate"]
            - 0.14 * metrics["eval_coherence_unsupported_rate"]
            - 0.08 * metrics["eval_utility_long_gap_rate"]
            - 0.04 * metrics["eval_terrain_tooth_rate"]
            - 0.06 * metrics["eval_overlong_block_run_rate"]
            - 0.06 * metrics["eval_bulky_mass_rate"]
            - 0.04 * metrics["eval_sky_mass_rate"]
            - 0.05 * metrics["eval_sparse_context_mismatch"]
            - 0.02 * metrics["eval_tpk_kl_3x3"]
            - 0.01 * metrics["eval_tpk_kl_4x4"]
            - 0.02 * metrics["eval_terrain_state_ngram_kl"]
        )
    if version == "v1.2":
        return (
            0.30 * metrics["eval_progress_mean"]
            + 0.22 * metrics["eval_completable_rate"]
            + 0.15 * metrics["eval_utility_score"]
            + 0.12 * metrics["eval_jump_gap_progress"]
            + 0.08 * metrics["eval_intra_context_diversity"]
            + 0.08 * metrics["eval_descriptor_adherence"]
            - 0.36 * metrics["eval_struct_viol_per_col"]
            - 0.16 * metrics["eval_coherence_singleton_rate"]
            - 0.14 * metrics["eval_coherence_unsupported_rate"]
            - 0.08 * metrics["eval_utility_long_gap_rate"]
            - 0.04 * metrics["eval_terrain_tooth_rate"]
            - 0.02 * metrics["eval_tpk_kl_3x3"]
            - 0.01 * metrics["eval_tpk_kl_4x4"]
            - 0.02 * metrics["eval_terrain_state_ngram_kl"]
        )
    if version == "v1.1":
        return (
            0.26 * metrics["eval_progress_mean"]
            + 0.22 * metrics["eval_completable_rate"]
            + 0.18 * metrics["eval_utility_score"]
            + 0.14 * metrics["eval_descriptor_adherence"]
            + 0.10 * metrics["eval_jump_gap_progress"]
            + 0.08 * metrics["eval_intra_context_diversity"]
            - 0.34 * metrics["eval_struct_viol_per_col"]
            - 0.14 * metrics["eval_coherence_singleton_rate"]
            - 0.14 * metrics["eval_coherence_unsupported_rate"]
            - 0.08 * metrics["eval_utility_long_gap_rate"]
            - 0.02 * metrics["eval_tpk_kl_3x3"]
            - 0.01 * metrics["eval_tpk_kl_4x4"]
            - 0.02 * metrics["eval_terrain_state_ngram_kl"]
            - 0.06 * metrics["eval_terrain_tooth_rate"]
        )
    if version == "v1.0":
        return (
            0.24 * metrics["eval_progress_mean"]
            + 0.18 * metrics["eval_completable_rate"]
            + 0.18 * metrics["eval_utility_score"]
            + 0.16 * metrics["eval_descriptor_adherence"]
            + 0.12 * metrics["eval_style_class_match_rate"]
            + 0.10 * metrics["eval_jump_gap_progress"]
            + 0.08 * metrics["eval_intra_context_diversity"]
            - 0.24 * metrics["eval_struct_viol_per_col"]
            - 0.02 * metrics["eval_tpk_kl_3x3"]
            - 0.01 * metrics["eval_tpk_kl_4x4"]
            - 0.02 * metrics["eval_terrain_state_ngram_kl"]
            - 0.10 * metrics["eval_utility_long_gap_rate"]
            - 0.08 * metrics["eval_coherence_singleton_rate"]
            - 0.08 * metrics["eval_terrain_tooth_rate"]
        )
    if version == "v0.15":
        return (
            0.28 * metrics["eval_progress_mean"]
            + 0.18 * metrics["eval_completable_rate"]
            + 0.18 * metrics["eval_utility_score"]
            + 0.12 * metrics["eval_style_class_match_rate"]
            - 0.22 * metrics["eval_struct_viol_per_col"]
            - 0.12 * metrics["eval_utility_long_gap_rate"]
            - 0.10 * metrics["eval_coherence_singleton_rate"]
            - 0.16 * metrics["eval_terrain_tooth_rate"]
            - 0.05 * metrics["eval_descriptor_distance"]
        )
    if version == "v0.14":
        return (
            0.30 * metrics["eval_progress_mean"]
            + 0.18 * metrics["eval_completable_rate"]
            + 0.18 * metrics["eval_utility_score"]
            + 0.14 * metrics["eval_style_class_match_rate"]
            - 0.24 * metrics["eval_struct_viol_per_col"]
            - 0.12 * metrics["eval_utility_long_gap_rate"]
            - 0.10 * metrics["eval_coherence_singleton_rate"]
            - 0.05 * metrics["eval_descriptor_distance"]
        )
    if version == "v0.13":
        return (
            0.42 * metrics["eval_progress_mean"]
            + 0.24 * metrics["eval_completable_rate"]
            + 0.26 * metrics["eval_utility_score"]
            + 0.08 * metrics["eval_utility_surface_col_rate"]
            - 0.28 * metrics["eval_struct_viol_per_col"]
            - 0.16 * metrics["eval_utility_long_gap_rate"]
        )
    if version in {"v0.9", "v0.12"}:
        return (
            metrics["eval_unknown_tile_acc"]
            + 0.16 * metrics["eval_completable_rate"]
            + 0.12 * metrics["eval_progress_mean"]
            - 0.22 * metrics["eval_struct_viol_per_col"]
        )
    return metrics["eval_unknown_tile_acc"]


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    mask = center_expand(args.width)
    split = make_level_splits_with_eval_files(args.data_dir, FIXED_EVAL_FILES)
    dev_split = make_dev_split(
        split["train"],
        args.val_count if args.version in {"v0.13", "v0.14", "v0.15", "v1.0", "v1.1", "v1.2", "v1.3", "v1.4"} else 0,
        args.seed,
    )
    train_windows = load_level_windows(args.data_dir, width=args.width, stride=args.stride, file_names=dev_split["train"])
    dev_windows = load_level_windows(args.data_dir, width=args.width, stride=args.stride, file_names=dev_split["dev"])
    eval_windows = load_level_windows(args.data_dir, width=args.width, stride=args.stride, file_names=split["eval"])
    train_levels = [record.level for record in train_windows]
    dev_levels = [record.level for record in dev_windows]
    eval_levels = [record.level for record in eval_windows]

    config = config_for_version(args.version, args.width)
    if args.version in {"v1.2", "v1.3", "v1.4"}:
        config = replace(
            config,
            activation=args.activation or config.activation,
            dfm_target=args.dfm_target or config.dfm_target,
            dfm_schedule=args.dfm_schedule or config.dfm_schedule,
            dfm_source=args.dfm_source or config.dfm_source,
            sample_steps=args.sample_steps if args.sample_steps is not None else config.sample_steps,
            support_logit_bias=args.support_logit_bias
            if args.support_logit_bias is not None
            else config.support_logit_bias,
            adaptive_support_bias=False if args.disable_adaptive_support_bias else config.adaptive_support_bias,
        )
    dataset = WindowDataset(train_levels, mask, config)
    if args.version in {"v1.2", "v1.3", "v1.4"} and config.dfm_source in {"train_prior", "gap_style_prior"}:
        config = replace(config, dfm_source_probs=source_probs(dataset, mask))
    if args.version in {"v1.3", "v1.4"} and config.dfm_source in {"style_prior", "gap_style_prior"}:
        config = replace(config, dfm_source_style_probs=source_style_probs(dataset, mask))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CategoricalFlowNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    weights = class_weights(dataset, mask).to(device) if config.class_weighted_loss else None
    support_weights = torch.as_tensor([0.20, 0.45, 1.0, 1.1], dtype=torch.float32, device=device)
    landing_weights = torch.as_tensor([0.55, 1.15, 0.40], dtype=torch.float32, device=device)
    source_prob_t = (
        torch.as_tensor(config.dfm_source_probs, dtype=torch.float32, device=device)
        if config.dfm_source_probs is not None
        else None
    )
    source_style_prob_t = (
        torch.as_tensor(config.dfm_source_style_probs, dtype=torch.float32, device=device)
        if config.dfm_source_style_probs is not None
        else None
    )

    history = []
    best_score = -float("inf")
    best_metrics = {"eval_unknown_tile_acc": -1.0}
    best_epoch = 0
    best_state = None
    phase_best = {
        "ground_truth": {"score": -float("inf"), "epoch": 0, "metrics": {}},
        "sampled": {"score": -float("inf"), "epoch": 0, "metrics": {}},
    }
    ground_truth_skeleton_epochs = 0
    if config.skeleton_conditioning:
        ground_truth_skeleton_epochs = (
            args.ground_truth_skeleton_epochs
            if args.ground_truth_skeleton_epochs > 0
            else max(1, args.epochs // 2)
        )
    skeleton_rng = torch.Generator(device=device)
    skeleton_rng.manual_seed(args.seed + 101)
    dfm_rng = torch.Generator(device=device)
    dfm_rng.manual_seed(args.seed + 202)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (
            x,
            target,
            support_target,
            landing_target,
            utility_target,
            terrain_target,
            continuation_target,
            skeleton_state_target,
            skeleton_height_target,
            motif_budget_target,
        ) in loader:
            x = x.to(device)
            target = target.to(device)
            support_target = support_target.to(device)
            landing_target = landing_target.to(device)
            utility_target = utility_target.to(device)
            terrain_target = terrain_target.to(device)
            continuation_target = continuation_target.to(device)
            skeleton_state_target = skeleton_state_target.to(device)
            skeleton_height_target = skeleton_height_target.to(device)
            motif_budget_target = motif_budget_target.to(device)
            source_target = target
            reveal_mask = torch.ones_like(target, dtype=torch.bool)
            if config.time_conditioning:
                x, source_target, reveal_mask, _ = make_dfm_training_batch(
                    x,
                    target,
                    mask_t,
                    config,
                    source_prob_t,
                    source_style_prob_t,
                    dfm_rng,
                )
            optimizer.zero_grad(set_to_none=True)
            if (
                config.structure_heads
                or config.utility_heads
                or config.terrain_heads
                or config.continuation_heads
                or config.skeleton_heads
            ):
                use_ground_truth_skeleton = config.skeleton_conditioning and epoch <= ground_truth_skeleton_epochs
                if config.skeleton_conditioning and not use_ground_truth_skeleton:
                    outputs = model.forward_sampled_skeleton(x, generator=skeleton_rng)
                else:
                    skeleton_targets = (
                        (skeleton_state_target, skeleton_height_target, motif_budget_target)
                        if config.skeleton_conditioning
                        else None
                    )
                    outputs = model.forward_all(x, skeleton_targets=skeleton_targets)
                if config.time_conditioning:
                    loss = dfm_tile_loss(
                        outputs["tile"],
                        target,
                        source_target,
                        reveal_mask,
                        support_target,
                        mask_t,
                        weights,
                        config,
                    )
                else:
                    loss = unknown_ce_loss(outputs["tile"], target, mask_t, weights)
                if config.skeleton_heads:
                    state_weights = torch.as_tensor([0.85, 0.55, 0.95, 1.05, 1.05, 1.15], dtype=torch.float32, device=device)
                    height_weights = torch.as_tensor([0.80, 0.85, 0.72, 0.95, 1.05, 1.10], dtype=torch.float32, device=device)
                    motif_weights = torch.as_tensor([0.60, 1.00, 1.35], dtype=torch.float32, device=device)
                    loss = loss + 0.34 * unknown_column_ce_loss(
                        outputs["skeleton_state"],
                        skeleton_state_target,
                        mask_t,
                        state_weights,
                    )
                    loss = loss + 0.22 * unknown_column_ce_loss(
                        outputs["skeleton_height"],
                        skeleton_height_target,
                        mask_t,
                        height_weights,
                    )
                    loss = loss + 0.16 * unknown_column_ce_loss(
                        outputs["motif_budget"],
                        motif_budget_target,
                        mask_t,
                        motif_weights,
                    )
                if config.structure_heads:
                    loss = loss + 0.20 * unknown_aux_ce_loss(outputs["support"], support_target, mask_t, support_weights)
                    loss = loss + 0.15 * unknown_column_ce_loss(outputs["landing"], landing_target, mask_t, landing_weights)
                    if config.support_conditioning:
                        loss = loss + 0.22 * decoder_support_group_loss(outputs["tile"], support_target, mask_t)
                if config.utility_heads:
                    utility_weights = torch.as_tensor([0.18, 0.85, 1.05, 1.10, 1.20, 0.55], dtype=torch.float32, device=device)
                    loss = loss + 0.28 * unknown_aux_ce_loss(outputs["utility"], utility_target, mask_t, utility_weights)
                if config.terrain_heads:
                    terrain_weights = torch.as_tensor([0.14, 0.62, 0.48, 0.88, 1.15], dtype=torch.float32, device=device)
                    loss = loss + 0.18 * unknown_aux_ce_loss(outputs["terrain"], terrain_target, mask_t, terrain_weights)
                if config.continuation_heads:
                    continuation_weights = torch.as_tensor([0.16, 0.90, 0.82, 1.10], dtype=torch.float32, device=device)
                    loss = loss + 0.20 * unknown_aux_ce_loss(
                        outputs["continuation"], continuation_target, mask_t, continuation_weights
                    )
            else:
                loss = unknown_ce_loss(model(x), target, mask_t, weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0:
            fill_model = TileFlowModel(model, config, device=device)
            monitor_levels = dev_levels if dev_levels else eval_levels
            eval_row = reconstruction_metrics(fill_model, monitor_levels, mask)
            benchmark_row = reconstruction_metrics(fill_model, eval_levels, mask)
            score = selection_score(eval_row, args.version)
            skeleton_mode = (
                "ground_truth" if config.skeleton_conditioning and epoch <= ground_truth_skeleton_epochs else "sampled"
            )
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_score": score,
                "skeleton_training_mode": skeleton_mode,
                "monitor_split": "dev" if dev_levels else "eval",
                **{f"monitor_{key.removeprefix('eval_')}": value for key, value in eval_row.items()},
                **{f"fixed_{key.removeprefix('eval_')}": value for key, value in benchmark_row.items()},
            })
            if skeleton_mode in phase_best and score > phase_best[skeleton_mode]["score"]:
                phase_best[skeleton_mode] = {"score": score, "epoch": epoch, "metrics": dict(eval_row)}
            can_select = not (args.version in {"v1.1", "v1.2", "v1.3", "v1.4"} and skeleton_mode != "sampled")
            if can_select and score > best_score:
                best_score = score
                best_metrics = dict(eval_row)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f"{args.version} epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"eval_acc={eval_row['eval_unknown_tile_acc']:.4f} "
                f"progress={eval_row['eval_progress_mean']:.4f} "
                f"utility={eval_row['eval_utility_score']:.4f} "
                f"tooth={eval_row['eval_terrain_tooth_rate']:.4f} "
                f"score={score:.4f}"
            )

    if best_state is None:
        raise RuntimeError(
            "No selectable checkpoint was produced. For v1.1+ sampled-skeleton versions, ensure training reaches sampled phase."
        )
    model.load_state_dict(best_state)
    fill_model = TileFlowModel(model, config, device=device)
    final_metrics = reconstruction_metrics(fill_model, eval_levels, mask)
    dev_metrics = reconstruction_metrics(fill_model, dev_levels, mask) if dev_levels else {}
    skeleton_audit = skeleton_distribution_audit(model, eval_windows, mask, config, device, args.seed)
    artifact_name = f"tileflow_{args.version}"
    output_dir = args.output_dir / artifact_name
    checkpoint_dir = args.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"tileflow_{args.version}.pt"

    metrics = {
        "version": args.version,
        "config": asdict(config),
        "train_window_count": len(train_windows),
        "dev_window_count": len(dev_windows),
        "eval_window_count": len(eval_windows),
        "train_files": list(dev_split["train"]),
        "dev_files": list(dev_split["dev"]),
        "eval_files": list(FIXED_EVAL_FILES),
        "epochs": args.epochs,
        "ground_truth_skeleton_epochs": ground_truth_skeleton_epochs,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "phase_best": phase_best,
        "history": history,
        "best": best_metrics,
        "dev": dev_metrics,
        "final": final_metrics,
        "skeleton_audit": skeleton_audit,
    }
    torch.save(checkpoint_payload(model, config, metrics), checkpoint_path)
    metrics["checkpoint"] = str(checkpoint_path)

    if not args.skip_benchmark:
        from tileflow.benchmarks.suite import BenchmarkConfig, run_fill_benchmark

        benchmark_dir = args.benchmark_dir / artifact_name
        report = run_fill_benchmark(
            fill_model,
            BenchmarkConfig(
                method=artifact_name,
                output_dir=benchmark_dir,
                data_dir=args.data_dir,
                width=args.width,
                stride=args.stride,
                n=args.benchmark_n,
                seed=args.seed,
                render=args.render_benchmark,
            ),
        )
        metrics["benchmark"] = report["main_benchmark"]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "splits.json").write_text(
        json.dumps({"train": dev_split["train"], "dev": dev_split["dev"], "eval": list(FIXED_EVAL_FILES)}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'metrics.json'}")
    print(f"wrote {checkpoint_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=[
            "v0.1",
            "v0.2",
            "v0.3",
            "v0.9",
            "v0.12",
            "v0.13",
            "v0.14",
            "v0.15",
            "v1.0",
            "v1.1",
            "v1.2",
            "v1.3",
            "v1.4",
        ],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tileflow"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--ground-truth-skeleton-epochs", type=int, default=0)
    parser.add_argument("--dfm-target", choices=["x", "noise", "support"], default=None)
    parser.add_argument("--dfm-schedule", choices=["linear", "cosine", "sqrt"], default=None)
    parser.add_argument("--dfm-source", choices=["air", "uniform", "train_prior", "style_prior", "gap_style_prior"], default=None)
    parser.add_argument("--activation", choices=["gelu", "silu"], default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--support-logit-bias", type=float, default=None)
    parser.add_argument("--disable-adaptive-support-bias", action="store_true")
    parser.add_argument("--benchmark-n", type=int, default=1)
    parser.add_argument("--render-benchmark", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
