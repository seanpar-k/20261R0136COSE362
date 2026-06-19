"""Shared benchmark scaffold for center-conditioned TileFlow evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tileflow.common.continuity import continuity_metrics
from tileflow.common.data import DEFAULT_W, SEED, LevelWindow, load_level_windows
from tileflow.common.eval import (
    context_consistency,
    diversity,
    masked_diversity,
    playable_masked_diversity,
    playability,
    rare_tile_scores,
    seam_transition_kl,
    structural_violations,
    terrain_state_ngram_kl,
    tile_pattern_kl,
)
from tileflow.common.fill_api import FillModel, assert_known_preserved
from tileflow.common.masks import center_expand
from tileflow.common.render import render_level
from tileflow.common.style import style_metrics

FIXED_EVAL_FILES = ["mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt"]
MAIN_BENCHMARK_METRICS = [
    "completable_rate",
    "tpk_kl_2x2",
    "playable_masked_diversity",
    "struct_viol_per_col",
    "continuity_score",
    "descriptor_distance",
]
MAIN_BENCHMARK_COLUMNS = [
    "method",
    "known_preserved",
    *MAIN_BENCHMARK_METRICS,
]


@dataclass(frozen=True)
class BenchmarkConfig:
    method: str
    output_dir: Path = Path("results/benchmarks/random_fill")
    data_dir: Path = Path("data")
    eval_files: tuple[str, ...] = tuple(FIXED_EVAL_FILES)
    mask_name: str = "center_expand"
    width: int = DEFAULT_W
    stride: int = 10
    n: int = 8
    seed: int = SEED
    render: bool = True


def _write_level(path: Path, level: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(level) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _split_metric_groups(method: str, metrics: dict[str, float], known_preserved: bool) -> tuple[dict, dict]:
    main = {
        "method": method,
        "known_preserved": bool(known_preserved),
    }
    for key in MAIN_BENCHMARK_METRICS:
        main[key] = metrics[key]
    diagnostics = {key: value for key, value in metrics.items() if key not in MAIN_BENCHMARK_METRICS}
    return main, diagnostics


def format_benchmark_table(rows: list[dict]) -> str:
    headers = [
        "Method",
        "Known preserved",
        "Completable",
        "Fidelity KL",
        "Playable diversity",
        "Structure errors",
        "Boundary continuity",
        "Style distance",
    ]
    keys = MAIN_BENCHMARK_COLUMNS

    def fmt(key: str, value) -> str:
        if key == "known_preserved":
            return "pass" if value else "fail"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(key, row[key]) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def _score_center_expand(
    windows: list[LevelWindow],
    generated_batches: list[list[list[str]]],
    mask: np.ndarray,
) -> dict[str, float]:
    contexts = [record.level for record in windows]
    filled_levels = [level for batch in generated_batches for level in batch]
    context_for_level = [context for context, batch in zip(contexts, generated_batches) for _ in batch]

    play = [playability(level) for level in filled_levels]
    struct = [structural_violations(level) for level in filled_levels]
    consistency = [
        context_consistency(filled, original, mask)
        for filled, original in zip(filled_levels, context_for_level)
    ]
    rare = [
        rare_tile_scores(filled, original, mask)
        for filled, original in zip(filled_levels, context_for_level)
    ]
    seams = [seam_transition_kl(level, mask, contexts) for level in filled_levels]

    report = {
        "completable_rate": float(np.mean([p["completable"] for p in play])),
        "progress_mean": float(np.mean([p["progress"] for p in play])),
        "tpk_kl_2x2": tile_pattern_kl(filled_levels, contexts, n=2),
        "tpk_kl_3x3": tile_pattern_kl(filled_levels, contexts, n=3),
        "tpk_kl_4x4": tile_pattern_kl(filled_levels, contexts, n=4),
        "terrain_state_ngram_kl": terrain_state_ngram_kl(filled_levels, contexts, mask, n=3),
        "diversity": diversity(filled_levels),
        "masked_diversity": masked_diversity(filled_levels, mask),
        "intra_context_diversity": float(
            np.mean([masked_diversity(batch, mask) for batch in generated_batches if len(batch) >= 2])
            if any(len(batch) >= 2 for batch in generated_batches)
            else 0.0
        ),
        "playable_masked_diversity": playable_masked_diversity(filled_levels, mask),
        "jump_gap_progress": float(np.mean([p.get("gap_progress", p["progress"]) for p in play])),
        "seam_kl_left": float(np.mean([s["left"] for s in seams])),
        "seam_kl_right": float(np.mean([s["right"] for s in seams])),
        "struct_viol_per_col": float(np.mean([s["per_col"] for s in struct])),
        "context_tile_acc": float(np.mean([c["tile_acc"] for c in consistency])),
        "context_valid_rate": float(np.mean([c["valid_rate"] for c in consistency])),
        "known_preservation_rate": 1.0,
        "rare_precision": float(np.mean([r["rare_precision"] for r in rare])),
        "rare_recall": float(np.mean([r["rare_recall"] for r in rare])),
        "rare_f1": float(np.mean([r["rare_f1"] for r in rare])),
        "object_recall": float(np.mean([r["object_recall"] for r in rare])),
    }

    style_rows = [style_metrics(level, mask) for level in filled_levels]
    for key in [
        "complete_structure_viol_per_col",
        "pipe_valid_rate",
        "pipe_pair_error_rate",
        "cannon_valid_rate",
        "ground_void_ratio_gap",
        "obstacle_density_gap",
        "style_class_match_rate",
        "descriptor_distance",
        "overlong_block_run_rate",
        "bulky_mass_rate",
        "sky_mass_rate",
        "sparse_context_mismatch",
    ]:
        report[key] = float(np.mean([row[key] for row in style_rows]))

    continuity_rows = [continuity_metrics(level, mask) for level in filled_levels]
    continuity_score = float(np.mean([row["continuity_score"] for row in continuity_rows]))
    for key in [
        "coin_run_continuation_rate",
        "platform_row_continuity",
        "ground_height_continuity",
        "boundary_pattern_kl",
        "continuity_score",
    ]:
        report[key] = float(np.mean([row[key] for row in continuity_rows]))
    report["continuity_diversity_tradeoff"] = float(report["masked_diversity"] * (0.5 + 0.5 * continuity_score))
    return report


def _load_windows(config: BenchmarkConfig) -> list[LevelWindow]:
    return load_level_windows(
        config.data_dir,
        width=config.width,
        stride=config.stride,
        file_names=config.eval_files,
    )


def run_fill_benchmark(model: FillModel, config: BenchmarkConfig) -> dict:
    if config.mask_name != "center_expand":
        raise ValueError("The shared benchmark scaffold currently supports center_expand only.")

    mask = center_expand(config.width)
    windows = _load_windows(config)
    generated_batches: list[list[list[str]]] = []

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for record in windows:
        window_dir = config.output_dir / record.source / f"window_{record.window_index:03d}"
        _write_level(window_dir / "context.txt", record.level)
        if config.render:
            render_level(record.level, window_dir / "context.png", title=f"{record.source} / context")

        sample_dir = window_dir / config.mask_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        condition = {
            "source": record.source,
            "window_index": record.window_index,
            "mask": config.mask_name,
            "width": config.width,
            "known_true": True,
            "known_cols": np.where(mask.any(axis=0))[0].astype(int).tolist(),
            "unknown_cols": np.where((~mask).any(axis=0))[0].astype(int).tolist(),
        }
        (sample_dir / "condition.json").write_text(json.dumps(condition, indent=2), encoding="utf-8")

        batch = []
        for sample_index in range(config.n):
            filled = model.fill(record.level, mask)
            assert_known_preserved(record.level, filled, mask)
            batch.append(filled)
            sample_path = sample_dir / f"sample_{sample_index:03d}.txt"
            _write_level(sample_path, filled)
            if config.render:
                render_level(
                    filled,
                    sample_path.with_suffix(".png"),
                    title=f"{record.source} / {config.mask_name} / sample_{sample_index:03d}",
                    mask=mask,
                    show_overlay=True,
                )
        generated_batches.append(batch)

    metrics = _score_center_expand(windows, generated_batches, mask)
    by_source = {}
    for source in sorted({record.source for record in windows}):
        source_windows = [record for record in windows if record.source == source]
        source_batches = [batch for record, batch in zip(windows, generated_batches) if record.source == source]
        source_metrics = _score_center_expand(source_windows, source_batches, mask)
        source_main, source_diagnostics = _split_metric_groups(config.method, source_metrics, known_preserved=True)
        by_source[source] = {
            "main": source_main,
            "diagnostics": source_diagnostics,
            "all_metrics": source_metrics,
        }
    main_metrics, diagnostics = _split_metric_groups(config.method, metrics, known_preserved=True)
    report = {
        "_benchmark": {
            **{key: _jsonable(value) for key, value in asdict(config).items()},
            "eval_window_count": len(windows),
            "sample_count": len(windows) * config.n,
            "main_benchmark_columns": MAIN_BENCHMARK_COLUMNS,
        },
        "main_benchmark": main_metrics,
        config.mask_name: {
            "main": main_metrics,
            "diagnostics": diagnostics,
            "all_metrics": metrics,
        },
        "by_source": by_source,
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (config.output_dir / "benchmark_table.md").write_text(
        format_benchmark_table([main_metrics]),
        encoding="utf-8",
    )
    return report


def fixed_eval_files() -> Iterable[str]:
    return tuple(FIXED_EVAL_FILES)
