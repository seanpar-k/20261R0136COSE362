"""Run final center-expand benchmarks with full external baselines.

This runner is intentionally separate from ``run_competitor_benchmarks.py``,
which uses archived local proxy/light adapters. The external baselines here call
the original repositories under ``external/`` and their Hugging Face model
paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf-cache" / "transformers"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tileflow-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tileflow-cache")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_generated_maps import render_benchmark_image_sets
from tileflow.benchmarks.external_full import MarioDiffusionFullAdapter, MarioGPTFullAdapter
from tileflow.benchmarks.random_fill import RandomFillBenchmark
from tileflow.benchmarks.suite import BenchmarkConfig, format_benchmark_table, run_fill_benchmark
from tileflow.common.data import DEFAULT_W, SEED
from tileflow.models.categorical_flow import load_tileflow_model


def _parse_methods(value: str) -> list[str]:
    if value == "all":
        return ["random_fill", "mariogpt_full", "mariodiffusion_full", "tileflow"]
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"random_fill", "mariogpt_full", "mariodiffusion_full", "tileflow"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def _config(args: argparse.Namespace, method: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        method=method,
        output_dir=args.output_dir / method,
        data_dir=args.data_dir,
        width=args.width,
        stride=args.stride,
        n=args.n,
        seed=args.seed,
        render=args.render,
    )


def _write_report(args: argparse.Namespace, rows: list[dict], failures: dict[str, str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = format_benchmark_table(rows) if rows else "| Method | Status |\n| --- | --- |\n"
    (args.output_dir / "benchmark_table.md").write_text(table, encoding="utf-8")
    report = {
        "scope": "final center_expand benchmark set",
        "tileflow_version": args.tileflow_version,
        "eval_set": ["mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt"],
        "mask": "center_expand",
        "width": args.width,
        "stride": args.stride,
        "samples_per_window": args.n,
        "external_baseline_policy": {
            "mariogpt_full": "original external/mario-gpt MarioLM using HF model path, projected into center_expand by preserving known center cells",
            "mariodiffusion_full": "original external/MarioDiffusion TextConditionalDDPMPipeline using HF model path, projected into center_expand by preserving known center cells",
            "limitation": "Original public APIs are text/open-ended generators, not exact center-inpainting APIs. The benchmark adapter preserves the same center mask after generation and records this limitation.",
        },
        "rows": rows,
        "failures": failures,
    }
    (args.output_dir / "final_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all", help="all or comma-separated methods")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/final"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--tileflow-version", default="v1.2")
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--visuals", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--mariogpt-model-path", default="shyamsn97/Mario-GPT2-700-context-length")
    parser.add_argument("--mariogpt-temperature", type=float, default=2.0)
    parser.add_argument("--mariodiffusion-model-path", default="schrum2/MarioDiffusion-MLM-regular0")
    parser.add_argument("--mariodiffusion-steps", type=int, default=30)
    parser.add_argument("--mariodiffusion-guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    rows: list[dict] = []
    failures: dict[str, str] = {}

    def run(method: str, model) -> None:
        report = run_fill_benchmark(model, _config(args, method))
        rows.append(report["main_benchmark"])

    for method in methods:
        try:
            if method == "random_fill":
                run(method, RandomFillBenchmark(seed=args.seed))
            elif method == "mariogpt_full":
                run(
                    method,
                    MarioGPTFullAdapter(
                        model_path=args.mariogpt_model_path,
                        width=args.width,
                        temperature=args.mariogpt_temperature,
                        device=args.device,
                    ),
                )
            elif method == "mariodiffusion_full":
                run(
                    method,
                    MarioDiffusionFullAdapter(
                        model_path=args.mariodiffusion_model_path,
                        width=args.width,
                        num_inference_steps=args.mariodiffusion_steps,
                        guidance_scale=args.mariodiffusion_guidance_scale,
                        device=args.device,
                        seed=args.seed,
                    ),
                )
            elif method == "tileflow":
                checkpoint_path = args.checkpoint_dir / f"tileflow_{args.tileflow_version}.pt"
                model = load_tileflow_model(checkpoint_path, device=args.device)
                run(f"tileflow_{args.tileflow_version}", model)
        except Exception as exc:
            failures[method] = f"{type(exc).__name__}: {exc}"
            if not args.continue_on_failure:
                _write_report(args, rows, failures)
                raise

    _write_report(args, rows, failures)
    if args.visuals and rows:
        render_benchmark_image_sets(
            args.output_dir,
            width=args.width,
            mask_name="center_expand",
            overlay=True,
            visual_root=args.output_dir / "visuals",
        )

    print(f"wrote {args.output_dir / 'final_benchmark_report.json'}")
    print(f"wrote {args.output_dir / 'benchmark_table.md'}")
    if rows:
        print(format_benchmark_table(rows), end="")
    if failures:
        print("failures:")
        for method, error in failures.items():
            print(f"- {method}: {error}")


if __name__ == "__main__":
    main()
