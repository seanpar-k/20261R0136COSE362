# TileFlow

TileFlow is a center-conditioned Mario-style tile-map expansion project. Given
a known center region, the model generates plausible left/right map context
while preserving the center tiles exactly.

## What Is Included

- `tileflow/`: core package, metrics, rendering, and model code
- `scripts/`: training, rendering, and benchmark entrypoints
- `tests/`: smoke and contract tests
- `data/`: Mario-style text levels used by the project
- `results/`: final submission artifacts
  - `results/checkpoints/tileflow_v1.4.pt`
  - `results/benchmarks/tileflow_v1.4/`
  - `results/benchmarks/final_tileflow_v1.4/`
  - `results/benchmarks/final_full_available/`
  - `results/visuals/tileflow_v1.4/`

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

## Run TileFlow v1.4 Benchmark

```bash
python scripts/run_final_benchmarks.py \
  --methods tileflow \
  --tileflow-version v1.4 \
  --checkpoint-dir results/checkpoints \
  --output-dir results/benchmarks/final_tileflow_v1.4 \
  --n 4 \
  --device cpu \
  --visuals
```

## Final TileFlow v1.4 Result

| Method | Known preserved | Completable | Fidelity KL | Playable diversity | Structure errors | Boundary continuity | Style distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tileflow_v1.4 | pass | 0.7578 | 0.1059 | 0.1758 | 0.1473 | 0.8695 | 1.4446 |

## External Baselines

The final benchmark session used external MarioDiffusion and MarioGPT
repositories locally, but those repositories and downloaded model weights are
not committed here. To reproduce full external baselines, clone/download them
separately according to `results/benchmarks/final_benchmark_diagnostic.md` and
the notes in `LOG.md`.

## Notes

- The active submitted TileFlow artifact is `v1.4`.
- Historical probes and old model outputs were moved to local archive paths
  under `old/`, which is ignored by Git.
- The project goal is stochastic center-conditioned generation, not hidden
  target-tile reconstruction.
