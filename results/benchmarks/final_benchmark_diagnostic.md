# Final Benchmark Diagnostic

Date: 2026-06-19

## Scope

This session is benchmark-only. TileFlow model/version updates are out of
scope. TileFlow is compared as the canonical `v1.2` artifact unless a newer
artifact is explicitly selected later.

## Current Benchmark Diagnosis

The existing competitor runner is not using full external MarioGPT or
MarioDiffusion models:

| Method | Current path | Diagnosis |
| --- | --- | --- |
| MarioGPT | `scripts/run_competitor_benchmarks.py` imports `old/MarioDiff/mariodiff/models/mariogpt_ar.py` | Local causal AR adapter/proxy trained on the TileFlow train split, not the original MarioGPT HF model. |
| MarioDiffusion | `scripts/run_competitor_benchmarks.py` imports `old/MarioDiff/mariodiff/models/v1_8.py` | Archived user-developed D3PM-lite adapter/proxy, not external MarioDiffusion. |
| External MarioDiffusion | `external/MarioDiffusion/` | Valid external baseline repo exists. It is distinct from archived `old/MarioDiff/`. |
| External MarioGPT | `external/mario-gpt/` | Valid external baseline repo exists. |

Existing result directories confirm the proxy/light state:

- `results/benchmarks/mariogpt_causal_ar/`
- `results/benchmarks/mariodiffusion_d3pm_lite/`

## Full Baseline Availability

| Baseline | Original repo/API | Pretrained path | Local checkout | Current env status | Can run now? |
| --- | --- | --- | --- | --- | --- |
| MarioGPT | `external/mario-gpt`, `MarioLM()` | `shyamsn97/Mario-GPT2-700-context-length` | Present | Dependencies installed; HF weights downloaded to `.hf-cache`; CPU full eval is impractically slow and MPS is unsupported on this OS | Load yes, full benchmark not yet |
| MarioDiffusion | `external/MarioDiffusion`, `TextConditionalDDPMPipeline` via `get_pipeline()` | `schrum2/MarioDiffusion-MLM-regular0` recommended by repo README | Present | Dependencies installed; HF weights downloaded to `.hf-cache`; Python 3.9 annotation compatibility patched in external repo | Yes |
| TileFlow | `tileflow.models.categorical_flow.load_tileflow_model` | `results/checkpoints/tileflow_v1.2.pt` | Present | Available | Yes |

## Implemented Benchmark Runner

Added:

- `tileflow/benchmarks/external_full.py`
- `scripts/run_final_benchmarks.py`

The new runner is separate from the older proxy runner. It calls the original
external repositories only:

- `mariogpt_full`: `external/mario-gpt` + HF MarioGPT model path
- `mariodiffusion_full`: `external/MarioDiffusion` + HF MarioDiffusion model path
- `tileflow`: canonical TileFlow checkpoint, default `v1.2`
- `random_fill`: shared lower-bound baseline

Important limitation:

The public MarioGPT and MarioDiffusion APIs are text/open-ended generators, not
exact center-inpainting APIs. The full adapters therefore generate from a
deterministic caption extracted from the known center region, then preserve the
known center cells exactly under the same `center_expand` mask. This limitation
is recorded in `final_benchmark_report.json`.

## Materials Status

Resolved:

1. Python dependencies for the external baselines were installed without
   replacing the working `torch==2.8.0` install:
   - `transformers`
   - `diffusers`
   - `huggingface_hub`
   - `accelerate`
   - `safetensors`
   - `datasets`
   - `timm`
   - `scipy`
2. Pretrained weights were downloaded to the project-local `.hf-cache`:
   - `shyamsn97/Mario-GPT2-700-context-length`
   - `schrum2/MarioDiffusion-MLM-regular0`
3. MarioDiffusion model choice used:
   - default recommendation from repo README: `schrum2/MarioDiffusion-MLM-regular0`
4. The benchmark remains center-expansion focused. For external models without
   exact center-inpainting APIs, adapters use the known center to derive a
   caption and then enforce known-cell preservation under the shared mask.

Still needed:

1. A practical MarioGPT full benchmark compute path. The model loads, but CPU
   token-by-token generation over all eval windows is too slow for this machine;
   `--device mps` failed because the OS does not support the MPS backend.

## Executed Results

The initial diagnostic run before dependency installation was written to:

- `results/benchmarks/final_diagnostic/final_benchmark_report.json`

Available-method benchmark with `n=4`, same eval set, same `center_expand`, same
metric scaffold, and rule-selected PNG visuals:

- `results/benchmarks/final_available/final_benchmark_report.json`
- `results/benchmarks/final_available/benchmark_table.md`
- `results/benchmarks/final_available/visuals/visual_selection.json`
- `results/benchmarks/final_available/visuals/README.md`

Full-available benchmark including external MarioDiffusion full model:

- `results/benchmarks/final_full_available/final_benchmark_report.json`
- `results/benchmarks/final_full_available/benchmark_table.md`
- `results/benchmarks/final_full_available/visuals/visual_selection.json`
- `results/benchmarks/final_full_available/visuals/README.md`

Full-available quantitative table:

| Method | Known preserved | Completable | Fidelity KL | Playable diversity | Structure errors | Boundary continuity | Style distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_fill | pass | 0.0156 | 9.3438 | 0.9196 | 2.1900 | 0.3807 | 12.2919 |
| mariodiffusion_full | pass | 0.6094 | 0.0645 | 0.0859 | 0.1377 | 0.7608 | 1.7390 |
| tileflow_v1.2 | pass | 0.7109 | 0.1198 | 0.1867 | 0.1544 | 0.8752 | 1.6753 |

These are still partial relative to the requested final comparison because
`mariogpt_full` has not completed on available hardware.
