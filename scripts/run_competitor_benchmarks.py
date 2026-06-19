"""Run center-expand benchmarks for MarioGPT, MarioDiffusion, and TileFlow.

The external repositories are kept under ``external/`` for provenance:

- ``external/mario-gpt``: GPT-style left-to-right Mario level generation.
- ``external/MarioDiffusion``: text-conditioned Mario diffusion generation.

Their public APIs target open-ended text-to-level generation, while this
project's benchmark requires masked center-conditioned expansion. For a fair
same-input comparison, this script uses local adapters that preserve the
original method families:

- MarioGPT: a causal autoregressive column model trained on the fixed train
  split.
- MarioDiffusion: the archived D3PM-lite masked inpainting model trained on the
  fixed train split.
- TileFlow: the active categorical flow model trained/evaluated by
  ``scripts/train_tileflow.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tileflow-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tileflow-cache")

ROOT = Path(__file__).resolve().parents[1]
OLD_MARIODIFF = ROOT / "old" / "MarioDiff"
for path in (ROOT, OLD_MARIODIFF):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mariodiff.models.mariogpt_ar import MarioGPTCausalAR
from mariodiff.models.v1_8 import MarioDiffV18D3PM, train_v1_8_d3pm

from scripts.train_tileflow import train as train_tileflow
from tileflow.benchmarks.suite import (
    BenchmarkConfig,
    FIXED_EVAL_FILES,
    format_benchmark_table,
    run_fill_benchmark,
)
from tileflow.common.data import DEFAULT_W, SEED, load_level_windows, make_level_splits_with_eval_files
from tileflow.models.categorical_flow import load_tileflow_model


def parse_methods(value: str) -> list[str]:
    if value == "all":
        return ["mariogpt", "mariodiffusion", "tileflow"]
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"mariogpt", "mariodiffusion", "tileflow"}
    bad = sorted(set(methods) - allowed)
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Allowed: {sorted(allowed)}")
    return methods


def train_levels(data_dir: Path, width: int, stride: int) -> tuple[list[str], list[list[str]]]:
    split = make_level_splits_with_eval_files(data_dir, FIXED_EVAL_FILES)
    windows = load_level_windows(data_dir, width=width, stride=stride, file_names=split["train"])
    return split["train"], [record.level for record in windows]


def benchmark_config(args: argparse.Namespace, method: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        method=method,
        output_dir=args.benchmark_dir / method,
        data_dir=args.data_dir,
        width=args.width,
        stride=args.stride,
        n=args.n,
        seed=args.seed,
        render=args.render,
    )


def run_mariogpt(args: argparse.Namespace, train_data: list[list[str]]) -> dict:
    model = MarioGPTCausalAR(train_levels=train_data, width=args.width, seed=args.seed)
    report = run_fill_benchmark(model, benchmark_config(args, "mariogpt_causal_ar"))
    report["_adapter"] = {
        "source_repo": "https://github.com/shyamsn97/mario-gpt",
        "local_source": "old/MarioDiff/mariodiff/models/mariogpt_ar.py",
        "note": "Causal left-to-right adapter for TileFlow center-expand masks.",
    }
    (args.benchmark_dir / "mariogpt_causal_ar" / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def run_mariodiffusion(args: argparse.Namespace, train_data: list[list[str]]) -> dict:
    checkpoint_path = args.checkpoint_dir / "mariodiffusion_v1_8_adapter.pt"
    summary_path = checkpoint_path.with_suffix(".summary.json")
    if checkpoint_path.exists() and not args.force_retrain:
        model = MarioDiffV18D3PM(
            checkpoint_path=checkpoint_path,
            width=args.width,
            T=args.mariodiff_T,
            sample_steps=args.mariodiff_sample_steps,
            base=args.mariodiff_base,
            temperature=args.mariodiff_temperature,
            top_k=args.mariodiff_top_k,
            sample_final=args.mariodiff_sample_final,
            sample_intermediate=args.mariodiff_sample_intermediate,
            device=args.device,
        )
    else:
        model, summary = train_v1_8_d3pm(
            train_data,
            checkpoint_path=checkpoint_path,
            width=args.width,
            epochs=args.mariodiff_epochs,
            batch_size=args.mariodiff_batch_size,
            lr=args.mariodiff_lr,
            T=args.mariodiff_T,
            base=args.mariodiff_base,
            sample_steps=args.mariodiff_sample_steps,
            temperature=args.mariodiff_temperature,
            top_k=args.mariodiff_top_k,
            sample_final=args.mariodiff_sample_final,
            sample_intermediate=args.mariodiff_sample_intermediate,
            seed=args.seed,
            device=args.device,
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report = run_fill_benchmark(model, benchmark_config(args, "mariodiffusion_d3pm_lite"))
    report["_adapter"] = {
        "source_repo": "https://github.com/schrum2/MarioDiffusion",
        "local_source": "old/MarioDiff/mariodiff/models/v1_8.py",
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
    }
    (args.benchmark_dir / "mariodiffusion_d3pm_lite" / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def run_tileflow(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint_dir / f"tileflow_{args.tileflow_version}.pt"
    method = f"tileflow_{args.tileflow_version}"
    if checkpoint_path.exists() and not args.force_retrain:
        model = load_tileflow_model(checkpoint_path, device=args.device)
        return run_fill_benchmark(model, benchmark_config(args, method))

    tileflow_args = SimpleNamespace(
        version=args.tileflow_version,
        data_dir=args.data_dir,
        output_dir=args.tileflow_output_dir,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_dir=args.benchmark_dir,
        width=args.width,
        stride=args.stride,
        epochs=args.tileflow_epochs,
        batch_size=args.tileflow_batch_size,
        lr=args.tileflow_lr,
        weight_decay=args.tileflow_weight_decay,
        seed=args.seed,
        eval_every=max(1, args.tileflow_eval_every),
        benchmark_n=args.n,
        render_benchmark=args.render,
        device=args.device,
    )
    metrics = train_tileflow(tileflow_args)
    return {"main_benchmark": metrics["benchmark"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all", help="all or comma-separated: mariogpt,mariodiffusion,tileflow")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--tileflow-output-dir", type=Path, default=Path("results/tileflow"))
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--mariodiff-epochs", type=int, default=2)
    parser.add_argument("--mariodiff-batch-size", type=int, default=8)
    parser.add_argument("--mariodiff-lr", type=float, default=2e-4)
    parser.add_argument("--mariodiff-T", type=int, default=200)
    parser.add_argument("--mariodiff-base", type=int, default=16)
    parser.add_argument("--mariodiff-sample-steps", type=int, default=12)
    parser.add_argument("--mariodiff-temperature", type=float, default=1.0)
    parser.add_argument("--mariodiff-top-k", type=int, default=0)
    parser.add_argument("--mariodiff-sample-final", action="store_true")
    parser.add_argument("--mariodiff-sample-intermediate", action="store_true")

    parser.add_argument("--tileflow-version", default="v0.3", choices=["v0.1", "v0.2", "v0.3"])
    parser.add_argument("--tileflow-epochs", type=int, default=30)
    parser.add_argument("--tileflow-batch-size", type=int, default=16)
    parser.add_argument("--tileflow-lr", type=float, default=3e-3)
    parser.add_argument("--tileflow-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tileflow-eval-every", type=int, default=10)
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    train_files, train_data = train_levels(args.data_dir, args.width, args.stride)
    rows: list[dict] = []

    print(f"fixed_eval_files={list(FIXED_EVAL_FILES)}")
    print(f"train_file_count={len(train_files)} train_window_count={len(train_data)}")

    if "mariogpt" in methods:
        report = run_mariogpt(args, train_data)
        rows.append(report["main_benchmark"])
    if "mariodiffusion" in methods:
        report = run_mariodiffusion(args, train_data)
        rows.append(report["main_benchmark"])
    if "tileflow" in methods:
        report = run_tileflow(args)
        rows.append(report["main_benchmark"])

    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.benchmark_dir / "competitor_benchmark_table.md"
    table = format_benchmark_table(rows)
    combined_path.write_text(table, encoding="utf-8")
    print(f"wrote {combined_path}")
    print(table, end="")


if __name__ == "__main__":
    main()
