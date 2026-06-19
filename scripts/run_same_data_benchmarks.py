"""Run controlled same-data center-expand benchmarks.

This runner is for the main comparison setting: external baseline code is used,
but published Hugging Face pretrained baseline weights are not used. MarioGPT
and MarioDiffusion checkpoints must be local artifacts trained from the same
TileFlow training split.
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


DISPLAY_NAMES = {
    "mariogpt_same_data": "MarioGPT same-data small",
    "mariodiffusion_same_data": "MarioDiffusion same-data small",
    "tileflow": "TileFlow",
}


def _parse_methods(value: str) -> list[str]:
    if value == "all":
        return ["random_fill", "mariogpt_same_data", "mariodiffusion_same_data", "tileflow"]
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"random_fill", "mariogpt_same_data", "mariodiffusion_same_data", "tileflow"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def _require_local_dir(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required for same-data benchmarking.")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{label} must be a local same-data checkpoint path, not a Hugging Face repo id: {value}"
        )
    return str(path.resolve())


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
        "scope": "main same-data no-pretraining center_expand benchmark",
        "training_budget": "20 epochs",
        "eval_set": ["mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt"],
        "mask": "center_expand",
        "width": args.width,
        "stride": args.stride,
        "samples_per_window": args.n,
        "baseline_policy": {
            "pretrained_hf_weights": "disallowed for MarioGPT and MarioDiffusion in this runner",
            "mariogpt_same_data": "original external/mario-gpt model code with a required local same-data LM checkpoint; tokenizer path may be supplied separately",
            "mariodiffusion_same_data": "original external/MarioDiffusion pipeline code with a required local same-data diffusion checkpoint directory",
            "projection": "External open-ended generators are projected into the center_expand contract by preserving the known center cells after generation.",
        },
        "paths": {
            "mariogpt_model_path": args.mariogpt_model_path,
            "mariogpt_tokenizer_path": args.mariogpt_tokenizer_path,
            "mariodiffusion_model_path": args.mariodiffusion_model_path,
            "tileflow_checkpoint": str(args.tileflow_checkpoint),
        },
        "rows": rows,
        "failures": failures,
    }
    (args.output_dir / "same_data_benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all", help="all or comma-separated methods")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/main_benchmark"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--tileflow-checkpoint", type=Path, default=Path("results/checkpoints/tileflow.pt"))
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--visuals", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--mariogpt-model-path", default=None)
    parser.add_argument("--mariogpt-tokenizer-path", default=None)
    parser.add_argument("--mariogpt-temperature", type=float, default=2.0)
    parser.add_argument("--mariodiffusion-model-path", default=None)
    parser.add_argument("--mariodiffusion-steps", type=int, default=30)
    parser.add_argument("--mariodiffusion-guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    rows: list[dict] = []
    failures: dict[str, str] = {}

    def run(method: str, model) -> None:
        report = run_fill_benchmark(model, _config(args, method))
        row = dict(report["main_benchmark"])
        row["method"] = DISPLAY_NAMES.get(method, method)
        rows.append(row)

    for method in methods:
        try:
            if method == "random_fill":
                run(method, RandomFillBenchmark(seed=args.seed))
            elif method == "mariogpt_same_data":
                model_path = _require_local_dir(args.mariogpt_model_path, "--mariogpt-model-path")
                tokenizer_path = _require_local_dir(
                    args.mariogpt_tokenizer_path or args.mariogpt_model_path,
                    "--mariogpt-tokenizer-path",
                )
                run(
                    method,
                    MarioGPTFullAdapter(
                        model_path=model_path,
                        tokenizer_path=tokenizer_path,
                        width=args.width,
                        temperature=args.mariogpt_temperature,
                        device=args.device,
                        count_prompter=True,
                    ),
                )
            elif method == "mariodiffusion_same_data":
                model_path = _require_local_dir(
                    args.mariodiffusion_model_path,
                    "--mariodiffusion-model-path",
                )
                run(
                    method,
                    MarioDiffusionFullAdapter(
                        model_path=model_path,
                        width=args.width,
                        num_inference_steps=args.mariodiffusion_steps,
                        guidance_scale=args.mariodiffusion_guidance_scale,
                        device=args.device,
                        seed=args.seed,
                    ),
                )
            elif method == "tileflow":
                model = load_tileflow_model(args.tileflow_checkpoint, device=args.device)
                run(method, model)
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

    print(f"wrote {args.output_dir / 'same_data_benchmark_report.json'}")
    print(f"wrote {args.output_dir / 'benchmark_table.md'}")
    if rows:
        print(format_benchmark_table(rows), end="")
    if failures:
        print("failures:")
        for method, error in failures.items():
            print(f"- {method}: {error}")


if __name__ == "__main__":
    main()
