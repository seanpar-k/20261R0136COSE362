# TileFlow Agent Guide

TileFlow is the active project. It targets center-conditioned expansion of
Mario-style 2D tile maps with constraint-guided categorical/discrete flow.

Last completed prototype artifact: **TileFlow v1.4**.
Active prototype target: **post-v1.4 benchmark integration / next-step review**.

## Project Frame

- Task: expand a designer-provided center map region left/right.
- Dataset: Mario-style VGLC text levels in `data/`.
- Primary method: categorical/discrete flow over tile states.
- Main quality target: plausible stochastic map generation that preserves
  center context, style continuity, structure, and reachability.
- Do not optimize or judge primarily by hidden target-tile matching.

Avoid claiming novelty from generic PCG, diffusion, inpainting, or flow matching
alone. The defensible claim is constraint-guided center-conditioned 2D tile-map
expansion.

## Naming

- **MarioDiff**: archived user-developed earlier model under `old/MarioDiff/`.
  Do not use it as a current implementation reference unless explicitly asked.
- **MarioDiffusion**: external diffusion baseline and valid benchmark
  comparison target.

## Layout

- `tileflow/`: active package and shared logic
- `scripts/`: executable entrypoints only
- `data/`: source level text files
- `results/`: confirmed outputs and benchmark artifacts
- `old/`: archived legacy material
- `LOG.md`: full history, experiment record, and decision trail
- `PLAN.md`: final goal, stable research framing, and next-step implementation
  details
- `TASK.md`: next execution tasks only; keep items that are required by the
  current objective but not yet implemented
- `MARIO_GRAMMAR.md`: soft Mario terrain/tile grammar for model-side
  supervision, diagnostics, and visual review; do not treat it as a hard
  postprocessing rulebook

## Artifact Hygiene

Keep only selected version artifacts in canonical form:

- `results/tileflow/tileflow_vX.Y/`
- `results/benchmarks/tileflow_vX.Y/`
- `results/visuals/tileflow_vX.Y/`
- `results/checkpoints/tileflow_vX.Y.pt`

Temporary probes and sweeps may be run, but after assessment remove or archive
non-selected outputs such as `_probe`, `_tmp`, `_mix`, `_direct`, `_learned`,
`_hybrid`, `_logits`, or parameter sweep names like `s0p...`. Record decisions
in `LOG.md`.

## Contracts

- Map height: 14 rows
- Default width: 80 columns
- Vocabulary: `- X S Q E ? < > [ ] B b o`
- Fixed eval files: `mario-1-2.txt`, `mario-4-1.txt`, `mario-6-3.txt`
- Fill contract: `mask=True` cells are known and must be preserved exactly.

Benchmark methods:

1. random fill
2. MarioGPT
3. MarioDiffusion
4. TileFlow

For v0.13 and later, use a small file-level dev/val split from the non-benchmark
training files for checkpoint selection and utility-metric monitoring. Keep
`mario-1-2.txt`, `mario-4-1.txt`, and `mario-6-3.txt` as fixed benchmark files,
not as the main tuning set.

Main benchmark columns:

- known-cell preservation
- `completable_rate`
- `tpk_kl_2x2`
- `playable_masked_diversity`
- `struct_viol_per_col`
- `continuity_score`
- `descriptor_distance`

Visual comparisons must be rule-selected, not hand-picked.

## Verification

After structural code changes, run:

```bash
.venv/bin/python -m pytest -q
```
