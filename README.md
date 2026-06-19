# TileFlow

TileFlow is a center-conditioned Mario-style tile-map expansion project. Given
a known center region, the model generates plausible left/right map context
while preserving the center tiles exactly.

## What Is Included

- `tileflow/`: core package, metrics, rendering, and model code
- `scripts/`: training, rendering, and benchmark entrypoints
- `tests/`: smoke and contract tests
- `data/`: Mario-style text levels used by the project
- `results/`: generated checkpoints, benchmark outputs, and visual samples

Local archives, virtual environments, Hugging Face caches, and external
baseline checkouts are intentionally ignored by Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify

```bash
PYTHONPYCACHEPREFIX=/tmp/tileflow_pycache python -m pytest -q
```

Expected current result:

```text
10 passed
```

## Run Main Benchmark

The committed TileFlow checkpoint is `results/checkpoints/tileflow.pt`.
The MarioDiffusion-style same-data baseline checkpoint is not committed; prepare
that local baseline under `results/baselines/` before rerunning all methods.

```bash
python scripts/run_same_data_benchmarks.py \
  --methods all \
  --checkpoint-dir results/checkpoints \
  --tileflow-checkpoint results/checkpoints/tileflow.pt \
  --mariodiffusion-model-path results/baselines/same_data_v1.4/mariodiffusion_colab20 \
  --output-dir results/benchmarks/main_benchmark \
  --n 4 \
  --device cpu \
  --visuals
```

## Project Benchmark

The submitted benchmark uses a same-data, no-pretraining, `center_expand`
setting with a 20-epoch training budget. It includes TileFlow,
MarioGPT-style Causal AR, MarioDiffusion-style Same-Data, and Random Fill.
It excludes HF-pretrained supplementary runs, full-paper-scale comparisons,
retry/smoke outputs, and baseline checkpoints.

Summary: `results/benchmarks/main_benchmark/benchmark_summary.md`
Visual samples in `results/benchmarks/main_benchmark/visuals/main/` are
rule-selected examples from the same benchmark run.

## External Baselines

The submitted benchmark uses a MarioDiffusion-style same-data local checkpoint.
External repositories, downloaded model weights, retry runs, and full
paper-scale experiments are not committed here. Diagnostic MarioGPT external
runs were attempted locally but are not included in the submitted benchmark
because the reliable project comparison uses `MarioGPT-style Causal AR`.

## Notes

- The project goal is stochastic center-conditioned generation, not hidden
  target-tile reconstruction.
