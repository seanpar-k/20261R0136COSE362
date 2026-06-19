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

## Run Benchmark

```bash
python scripts/run_final_benchmarks.py \
  --methods tileflow \
  --checkpoint-dir results/checkpoints \
  --output-dir results/benchmarks/final_tileflow \
  --n 4 \
  --device cpu \
  --visuals
```

## External Baselines

The final benchmark session used external MarioDiffusion and MarioGPT
repositories locally, but those repositories and downloaded model weights are
not committed here. To reproduce full external baselines, clone/download them
separately and point the benchmark scripts to those local checkouts.

## Notes

- The project goal is stochastic center-conditioned generation, not hidden
  target-tile reconstruction.
