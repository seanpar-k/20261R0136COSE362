"""Context-style and structure-completion metrics for TileFlow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tileflow.common.data import MAP_HEIGHT, normalize_level

SOLID_TILES = set("XSQ?<>[]Bb")
PIPE_TOP = set("<>")
PIPE_BODY = set("[]")
CANNON_TILES = set("Bb")
BLOCK_TILES = set("SQ?")
ENEMY_TILES = set("E")
OBSTACLE_TILES = PIPE_TOP | PIPE_BODY | CANNON_TILES | BLOCK_TILES | ENEMY_TILES
SUPPORTABLE_TILES = PIPE_TOP | PIPE_BODY | CANNON_TILES | ENEMY_TILES


@dataclass(frozen=True)
class StyleDescriptor:
    ground_void_ratio: float
    gap_count: float
    mean_gap_length: float
    longest_gap: float
    longest_ground_run: float
    sky_solid_density: float
    mid_solid_density: float
    low_solid_density: float
    obstacle_density: float
    pipe_density: float
    cannon_density: float
    block_density: float
    enemy_density: float
    structural_density: float
    object_support_ratio: float
    style_class: str


@dataclass(frozen=True)
class StructureReport:
    partial_structure_count: int
    complete_structure_viol_per_col: float
    pipe_valid_rate: float
    pipe_pair_error_rate: float
    cannon_valid_rate: float
    unsupported_obstacle_rate: float
    boundary_structure_violations: int


def _selected_cols(region: np.ndarray) -> list[int]:
    return [c for c in range(region.shape[1]) if bool(region[:, c].any())]


def _runs(values: list[bool], target: bool) -> list[int]:
    runs: list[int] = []
    run = 0
    for value in values:
        if value == target:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def _density(rows: list[str], region: np.ndarray, row_range: range, tiles: set[str]) -> float:
    total = 0
    count = 0
    for r in row_range:
        for c in range(region.shape[1]):
            if not region[r, c]:
                continue
            total += 1
            if rows[r][c] in tiles:
                count += 1
    return count / max(total, 1)


def _supported(rows: list[str], r: int, c: int) -> bool:
    if r + 1 >= MAP_HEIGHT:
        return True
    return rows[r + 1][c] in SOLID_TILES


def style_descriptor(level: list[str], region: np.ndarray) -> StyleDescriptor:
    rows = normalize_level(level, width=region.shape[1])
    cols = _selected_cols(region)
    grounded = []
    for c in cols:
        ground_known = region[MAP_HEIGHT - 1, c] or region[MAP_HEIGHT - 2, c]
        ground_solid = rows[MAP_HEIGHT - 1][c] in SOLID_TILES or rows[MAP_HEIGHT - 2][c] in SOLID_TILES
        grounded.append(bool(ground_known and ground_solid))

    gap_runs = _runs(grounded, False)
    ground_runs = _runs(grounded, True)
    total_cells = int(region.sum())
    obstacle_count = 0
    pipe_count = cannon_count = block_count = enemy_count = 0
    supported = supportable = 0
    vertical_blockers = floating = narrow_landings = 0

    for c in cols:
        solid_run = 0
        for r in range(MAP_HEIGHT):
            if not region[r, c]:
                solid_run = 0
                continue
            ch = rows[r][c]
            if ch in OBSTACLE_TILES:
                obstacle_count += 1
            if ch in PIPE_TOP | PIPE_BODY:
                pipe_count += 1
            if ch in CANNON_TILES:
                cannon_count += 1
            if ch in BLOCK_TILES:
                block_count += 1
            if ch in ENEMY_TILES:
                enemy_count += 1
            if ch in SUPPORTABLE_TILES:
                supportable += 1
                if _supported(rows, r, c):
                    supported += 1
            if ch in SOLID_TILES:
                solid_run += 1
                if r + 1 < MAP_HEIGHT and region[r + 1, c] and rows[r + 1][c] == "-":
                    floating += 1
            else:
                solid_run = 0
            if solid_run >= 3:
                vertical_blockers += 1

    for r in range(6, MAP_HEIGHT):
        run = 0
        for c in cols:
            if region[r, c] and rows[r][c] in SOLID_TILES:
                run += 1
            else:
                if run == 1:
                    narrow_landings += 1
                run = 0
        if run == 1:
            narrow_landings += 1

    ground_void_ratio = 1.0 - (sum(grounded) / max(len(grounded), 1))
    structural_density = (vertical_blockers + floating + narrow_landings + sum(max(gap - 4, 0) for gap in gap_runs)) / max(len(cols), 1)
    obstacle_density = obstacle_count / max(total_cells, 1)
    if ground_void_ratio >= 0.35 or max(gap_runs or [0]) >= 5:
        style_class = "gap-heavy"
    elif obstacle_density >= 0.035 or structural_density >= 0.30:
        style_class = "obstacle-heavy"
    else:
        style_class = "plain/low-obstacle"

    return StyleDescriptor(
        ground_void_ratio=ground_void_ratio,
        gap_count=float(len(gap_runs)),
        mean_gap_length=float(np.mean(gap_runs)) if gap_runs else 0.0,
        longest_gap=float(max(gap_runs or [0])),
        longest_ground_run=float(max(ground_runs or [0])),
        sky_solid_density=_density(rows, region, range(0, 6), SOLID_TILES),
        mid_solid_density=_density(rows, region, range(6, 11), SOLID_TILES),
        low_solid_density=_density(rows, region, range(11, MAP_HEIGHT), SOLID_TILES),
        obstacle_density=obstacle_density,
        pipe_density=pipe_count / max(total_cells, 1),
        cannon_density=cannon_count / max(total_cells, 1),
        block_density=block_count / max(total_cells, 1),
        enemy_density=enemy_count / max(total_cells, 1),
        structural_density=structural_density,
        object_support_ratio=supported / max(supportable, 1),
        style_class=style_class,
    )


def descriptor_distance(known: StyleDescriptor, generated: StyleDescriptor) -> dict[str, float | bool]:
    ground_gap = abs(generated.ground_void_ratio - known.ground_void_ratio)
    obstacle_gap = abs(generated.obstacle_density - known.obstacle_density)
    pipe_gap = abs(generated.pipe_density - known.pipe_density)
    cannon_gap = abs(generated.cannon_density - known.cannon_density)
    block_gap = abs(generated.block_density - known.block_density)
    structural_gap = abs(generated.structural_density - known.structural_density)
    band_gap = (
        abs(generated.sky_solid_density - known.sky_solid_density)
        + abs(generated.mid_solid_density - known.mid_solid_density)
        + abs(generated.low_solid_density - known.low_solid_density)
    ) / 3.0
    support_gap = abs(generated.object_support_ratio - known.object_support_ratio)
    style_match = generated.style_class == known.style_class
    distance = (
        2.0 * ground_gap
        + 2.0 * obstacle_gap
        + pipe_gap
        + cannon_gap
        + block_gap
        + 1.5 * structural_gap
        + band_gap
        + 0.5 * support_gap
        + (0.0 if style_match else 0.75)
    )
    return {
        "descriptor_distance": float(distance),
        "ground_void_ratio_gap": float(ground_gap),
        "obstacle_density_gap": float(obstacle_gap),
        "pipe_density_gap": float(pipe_gap),
        "cannon_density_gap": float(cannon_gap),
        "block_density_gap": float(block_gap),
        "structural_density_gap": float(structural_gap),
        "style_class_match": bool(style_match),
    }


def structure_report(level: list[str], region: np.ndarray) -> StructureReport:
    rows = normalize_level(level, width=region.shape[1])
    width = region.shape[1]
    violations = 0
    pipe_total = pipe_valid = 0
    pipe_pair_checks = pipe_pair_errors = 0
    cannon_total = cannon_valid = 0
    unsupported = supportable = 0
    boundary_violations = 0

    for r in range(MAP_HEIGHT):
        for c in range(width):
            if not region[r, c]:
                continue
            ch = rows[r][c]
            if ch == "<":
                pipe_pair_checks += 1
                pipe_pair_errors += int(c + 1 >= width or rows[r][c + 1] != ">")
            elif ch == ">":
                pipe_pair_checks += 1
                pipe_pair_errors += int(c - 1 < 0 or rows[r][c - 1] != "<")
            elif ch == "[":
                pipe_pair_checks += 1
                pipe_pair_errors += int(c + 1 >= width or rows[r][c + 1] != "]")
            elif ch == "]":
                pipe_pair_checks += 1
                pipe_pair_errors += int(c - 1 < 0 or rows[r][c - 1] != "[")
            if ch in SUPPORTABLE_TILES:
                supportable += 1
                if _supported(rows, r, c):
                    unsupported += 0
                else:
                    unsupported += 1
            if ch == "<":
                pipe_total += 1
                valid = c + 1 < width and rows[r][c + 1] == ">"
                if r + 1 < MAP_HEIGHT and rows[r + 1][c] == "[":
                    valid = valid and c + 1 < width and rows[r + 1][c + 1] == "]"
                pipe_valid += int(valid)
                violations += 0 if valid else 1
            elif ch == ">":
                valid = c - 1 >= 0 and rows[r][c - 1] == "<"
                violations += 0 if valid else 1
            elif ch == "[":
                pipe_total += 1
                valid = c + 1 < width and rows[r][c + 1] == "]"
                if r - 1 >= 0 and rows[r - 1][c] in PIPE_TOP:
                    valid = valid and c + 1 < width and rows[r - 1][c + 1] == ">"
                if r + 1 < MAP_HEIGHT and rows[r + 1][c] in PIPE_BODY:
                    valid = valid and c + 1 < width and rows[r + 1][c + 1] == "]"
                pipe_valid += int(valid)
                violations += 0 if valid else 1
            elif ch == "]":
                valid = c - 1 >= 0 and rows[r][c - 1] == "["
                violations += 0 if valid else 1
            elif ch in CANNON_TILES:
                cannon_total += 1
                valid = r + 1 >= MAP_HEIGHT or rows[r + 1][c] in SOLID_TILES
                if ch == "B" and r + 1 < MAP_HEIGHT and rows[r + 1][c] == "b":
                    valid = True
                cannon_valid += int(valid)
                violations += 0 if valid else 1

            for dc in (-1, 1):
                nc = c + dc
                if 0 <= nc < width and not region[r, nc]:
                    if ch in "<[" and rows[r][nc] not in ">]":
                        boundary_violations += 1
                    if ch in ">]" and rows[r][nc] not in "<[":
                        boundary_violations += 1

    violations += unsupported + boundary_violations
    return StructureReport(
        partial_structure_count=int(violations),
        complete_structure_viol_per_col=violations / max(len(_selected_cols(region)), 1),
        pipe_valid_rate=1.0 if pipe_total == 0 else pipe_valid / pipe_total,
        pipe_pair_error_rate=0.0 if pipe_pair_checks == 0 else pipe_pair_errors / pipe_pair_checks,
        cannon_valid_rate=1.0 if cannon_total == 0 else cannon_valid / cannon_total,
        unsupported_obstacle_rate=unsupported / max(supportable, 1),
        boundary_structure_violations=int(boundary_violations),
    )


def run_mass_report(level: list[str], known_mask: np.ndarray) -> dict[str, float]:
    rows = normalize_level(level, width=known_mask.shape[1])
    generated = ~known_mask
    cols = _selected_cols(generated)
    if not cols:
        return {
            "overlong_block_run_rate": 0.0,
            "bulky_mass_rate": 0.0,
            "sky_mass_rate": 0.0,
        }

    overlong_cells = 0
    bulky_cells = 0
    sky_mass_cells = 0
    block_or_solid = SOLID_TILES
    for r in range(0, MAP_HEIGHT - 2):
        run = 0
        run_generated = False
        for c in cols + [-1]:
            in_bounds = c != -1
            is_solid = in_bounds and generated[r, c] and rows[r][c] in block_or_solid
            if is_solid:
                run += 1
                run_generated = True
                continue
            if run_generated and run > 10:
                overlong_cells += run - 10
            run = 0
            run_generated = False

    for r in range(0, MAP_HEIGHT - 3):
        for c in cols:
            if c + 1 >= known_mask.shape[1] or not generated[r, c] or not generated[r, c + 1]:
                continue
            if not generated[r + 1, c] or not generated[r + 1, c + 1]:
                continue
            if all(rows[rr][cc] in SOLID_TILES for rr in (r, r + 1) for cc in (c, c + 1)):
                bulky_cells += 1
                if r <= 5:
                    sky_mass_cells += 1

    denom = max(len(cols), 1)
    return {
        "overlong_block_run_rate": float(overlong_cells / denom),
        "bulky_mass_rate": float(bulky_cells / denom),
        "sky_mass_rate": float(sky_mass_cells / denom),
    }


def style_metrics(level: list[str], known_mask: np.ndarray) -> dict[str, float]:
    known = style_descriptor(level, known_mask)
    generated = style_descriptor(level, ~known_mask)
    distance = descriptor_distance(known, generated)
    structure = structure_report(level, ~known_mask)
    run_mass = run_mass_report(level, known_mask)
    out = {
        **{f"known_{k}": v for k, v in asdict(known).items() if isinstance(v, (int, float))},
        **{f"generated_{k}": v for k, v in asdict(generated).items() if isinstance(v, (int, float))},
        **{k: float(v) for k, v in distance.items() if isinstance(v, (int, float, bool))},
        **{k: float(v) for k, v in asdict(structure).items()},
        **run_mass,
    }
    out["style_class_match_rate"] = float(distance["style_class_match"])
    out["sparse_context_mismatch"] = float(distance["ground_void_ratio_gap"] if known.style_class == "gap-heavy" else 0.0)
    return out
