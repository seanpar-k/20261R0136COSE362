"""Run TileFlow benchmark methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.benchmarks.random_fill import RandomFillBenchmark
from tileflow.benchmarks.suite import BenchmarkConfig, format_benchmark_table, run_fill_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["random_fill"], default="random_fill")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"results/benchmarks/{args.method}")
    config = BenchmarkConfig(
        method=args.method,
        output_dir=output_dir,
        data_dir=Path(args.data_dir),
        n=args.n,
        seed=args.seed,
        width=args.width,
        stride=args.stride,
        render=not args.no_render,
    )

    if args.method == "random_fill":
        model = RandomFillBenchmark(seed=args.seed)
    else:
        raise ValueError(f"Unsupported benchmark method: {args.method}")

    report = run_fill_benchmark(model, config)
    metrics_path = output_dir / "metrics.json"
    table_path = output_dir / "benchmark_table.md"
    sample_count = report["_benchmark"]["sample_count"]
    print(f"wrote {metrics_path}")
    print(f"wrote {table_path}")
    print(f"sample_count={sample_count}")
    print(format_benchmark_table([report["main_benchmark"]]), end="")


if __name__ == "__main__":
    main()
