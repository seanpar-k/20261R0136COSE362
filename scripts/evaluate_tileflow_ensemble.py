"""Evaluate a TileFlow logits ensemble as a versioned model artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.benchmarks.suite import BenchmarkConfig, run_fill_benchmark
from tileflow.common.data import DEFAULT_W, SEED, load_level_windows, make_level_splits_with_eval_files
from tileflow.common.eval import context_consistency, playability, structural_violations
from tileflow.common.fill_api import assert_known_preserved
from tileflow.common.masks import center_expand
from tileflow.models import (
    load_tileflow_ensemble,
    load_tileflow_stochastic_guided_ensemble,
    load_tileflow_style_guided_ensemble,
)

FIXED_EVAL_FILES = ("mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt")


def reconstruction_metrics(model, eval_levels: list[list[str]], mask: np.ndarray) -> dict[str, float]:
    consistency = []
    play = []
    struct = []
    for level in eval_levels:
        filled = model.fill(level, mask)
        assert_known_preserved(level, filled, mask)
        consistency.append(context_consistency(filled, level, mask))
        play.append(playability(filled))
        struct.append(structural_violations(filled))
    return {
        "eval_unknown_tile_acc": float(np.mean([row["tile_acc"] for row in consistency])),
        "eval_completable_rate": float(np.mean([row["completable"] for row in play])),
        "eval_progress_mean": float(np.mean([row["progress"] for row in play])),
        "eval_struct_viol_per_col": float(np.mean([row["per_col"] for row in struct])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0.3")
    parser.add_argument("--paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--difficulty", choices=["easy", "neutral", "hard"], default="neutral")
    parser.add_argument("--structure-coupling", action="store_true")
    parser.add_argument("--coupling-strength", type=float, default=2.8)
    parser.add_argument("--learned-structure-bias", action="store_true")
    parser.add_argument("--learned-bias-strength", type=float, default=1.0)
    parser.add_argument("--utility-guidance", action="store_true")
    parser.add_argument("--utility-guidance-strength", type=float, default=1.0)
    parser.add_argument("--coherence-guidance", action="store_true")
    parser.add_argument("--coherence-guidance-strength", type=float, default=1.0)
    parser.add_argument("--context-guidance", action="store_true")
    parser.add_argument("--context-guidance-strength", type=float, default=1.0)
    parser.add_argument("--terrain-guidance", action="store_true")
    parser.add_argument("--terrain-guidance-strength", type=float, default=1.0)
    parser.add_argument("--continuation-guidance", action="store_true")
    parser.add_argument("--continuation-guidance-strength", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tileflow"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--benchmark-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    name = f"tileflow_{args.version}"
    if args.stochastic:
        model = load_tileflow_stochastic_guided_ensemble(
            args.paths,
            args.weights,
            device=args.device,
            name=name,
            strength=args.strength,
            temperature=args.temperature,
            top_k=args.top_k,
            difficulty=args.difficulty,
            seed=args.seed,
            structure_coupling=args.structure_coupling,
            coupling_strength=args.coupling_strength,
            learned_structure_bias=args.learned_structure_bias,
            learned_bias_strength=args.learned_bias_strength,
            utility_guidance=args.utility_guidance,
            utility_guidance_strength=args.utility_guidance_strength,
            coherence_guidance=args.coherence_guidance,
            coherence_guidance_strength=args.coherence_guidance_strength,
            context_guidance=args.context_guidance,
            context_guidance_strength=args.context_guidance_strength,
            terrain_guidance=args.terrain_guidance,
            terrain_guidance_strength=args.terrain_guidance_strength,
            continuation_guidance=args.continuation_guidance,
            continuation_guidance_strength=args.continuation_guidance_strength,
        )
    elif args.guided:
        model = load_tileflow_style_guided_ensemble(
            args.paths,
            args.weights,
            device=args.device,
            name=name,
            strength=args.strength,
        )
    else:
        model = load_tileflow_ensemble(args.paths, args.weights, device=args.device, name=name)
    mask = center_expand(args.width)
    split = make_level_splits_with_eval_files(args.data_dir, FIXED_EVAL_FILES)
    eval_windows = load_level_windows(args.data_dir, width=args.width, stride=args.stride, file_names=split["eval"])
    eval_levels = [record.level for record in eval_windows]
    final_metrics = reconstruction_metrics(model, eval_levels, mask)

    benchmark_dir = args.benchmark_dir / name
    report = run_fill_benchmark(
        model,
        BenchmarkConfig(
            method=name,
            output_dir=benchmark_dir,
            data_dir=args.data_dir,
            width=args.width,
            stride=args.stride,
            n=args.benchmark_n,
            seed=args.seed,
            render=False,
        ),
    )

    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_metrics_path = output_dir / "metrics.json"
    training_metrics_path = output_dir / "training_metrics.json"
    if existing_metrics_path.exists() and not training_metrics_path.exists():
        training_metrics_path.write_text(existing_metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path = output_dir / "ensemble_manifest.json"
    manifest = {
        "version": args.version,
        "model_type": "stochastic_guided_ensemble" if args.stochastic else "logit_ensemble",
        "guided": bool(args.guided or args.stochastic),
        "stochastic": bool(args.stochastic),
        "guidance_strength": args.strength if (args.guided or args.stochastic) else 0.0,
        "temperature": args.temperature if args.stochastic else 0.0,
        "top_k": args.top_k if args.stochastic else 0,
        "difficulty": args.difficulty if args.stochastic else "none",
        "structure_coupling": bool(args.structure_coupling) if args.stochastic else False,
        "coupling_strength": args.coupling_strength if args.stochastic and args.structure_coupling else 0.0,
        "learned_structure_bias": bool(args.learned_structure_bias) if args.stochastic else False,
        "learned_bias_strength": args.learned_bias_strength if args.stochastic and args.learned_structure_bias else 0.0,
        "utility_guidance": bool(args.utility_guidance) if args.stochastic else False,
        "utility_guidance_strength": args.utility_guidance_strength if args.stochastic and args.utility_guidance else 0.0,
        "coherence_guidance": bool(args.coherence_guidance) if args.stochastic else False,
        "coherence_guidance_strength": args.coherence_guidance_strength if args.stochastic and args.coherence_guidance else 0.0,
        "context_guidance": bool(args.context_guidance) if args.stochastic else False,
        "context_guidance_strength": args.context_guidance_strength if args.stochastic and args.context_guidance else 0.0,
        "terrain_guidance": bool(args.terrain_guidance) if args.stochastic else False,
        "terrain_guidance_strength": args.terrain_guidance_strength if args.stochastic and args.terrain_guidance else 0.0,
        "continuation_guidance": bool(args.continuation_guidance) if args.stochastic else False,
        "continuation_guidance_strength": (
            args.continuation_guidance_strength if args.stochastic and args.continuation_guidance else 0.0
        ),
        "members": [str(path) for path in args.paths],
        "weights": args.weights,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics = {
        **manifest,
        "eval_window_count": len(eval_windows),
        "eval_files": list(FIXED_EVAL_FILES),
        "final": final_metrics,
        "benchmark": report["main_benchmark"],
        "manifest": str(manifest_path),
    }
    splits_path = output_dir / "splits.json"
    if splits_path.exists():
        metrics["splits"] = json.loads(splits_path.read_text(encoding="utf-8"))
    if training_metrics_path.exists():
        metrics["training_metrics"] = str(training_metrics_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'metrics.json'}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
