# TileFlow Plan

## Goal

Build TileFlow: a center-conditioned Mario-style tile-map expansion model for
small or solo game developers who provide a seed region and want plausible
surrounding terrain, structures, and difficulty variation.

## Core Thesis

Discrete/categorical flow is a good fit because maps are categorical grids and
generation can be guided before tiles are committed. This is an advantage only
if the model learns or is guided by tile-to-tile structure coupling:

- supported platform runs
- paired pipes/cannons
- landing-safe gaps
- supported enemies and objects
- boundary style continuation

If generation degenerates into mostly independent per-cell sampling, the method
will not realize its advantage over diffusion-style baselines.

## Research Question

Can constraint-guided discrete flow expand a fixed center region into a larger
playable 2D tile map while preserving context, structural validity, style
continuity, and useful stochastic variation better than unconstrained or
post-filtered generation?

## Scope

In scope:

- Mario-style VGLC text levels
- fixed-height 14-row maps
- `center_expand` as the headline task
- compact quantitative metrics plus rule-selected PNG inspection
- optional player-behavior difficulty steering

Out of scope:

- full game-engine simulation
- broad text-to-level generation
- Genie-style interactive world modeling
- claiming novelty from diffusion, PCG, inpainting, or flow matching alone

## Evaluation

Benchmark methods:

- random fill
- MarioGPT
- MarioDiffusion
- TileFlow

Main benchmark columns:

| Column | Direction |
| --- | --- |
| Known preserved | must pass |
| `completable_rate` | higher |
| `tpk_kl_2x2` | lower |
| `playable_masked_diversity` | higher |
| `struct_viol_per_col` | lower |
| `continuity_score` | higher |
| `descriptor_distance` | lower |

Acceptance priority:

1. Generated samples must be useful Mario maps: traversable, structurally
   coherent, and not degenerate empty/floor-heavy expansions.
2. After map utility is acceptable, judge center-context style continuation,
   stochastic variation, and difficulty/structure matching.
3. Hidden target-tile reconstruction is diagnostic only and must not decide
   success.

Validation protocol for v0.13 and later:

- Keep the fixed benchmark files (`mario-1-2.txt`, `mario-4-1.txt`,
  `mario-6-3.txt`) separate for final comparison and visual feedback.
- Create a small file-level dev/val split from the remaining training files for
  checkpoint selection, hyperparameter decisions, and utility-metric monitoring.
- Prefer file-level splitting over window-level splitting because overlapping
  windows from the same level can leak structure between train and validation.
- Keep the dev/val split small so the limited training set is not weakened.
- Use dev/val to prevent false progress; it should not replace final benchmark
  comparison or human PNG inspection.

Qualitative visual comparison must use rule-selected representative images:

- one main median-quality sample per eval source and method
- appendix best/median/worst samples from the same canonical windows
- no manual cherry-picking

## v1.0 Redesign Plan

Purpose:

v1.0 replaces the v0.16 path. The problem is not one isolated symptom such as a
two-tile floor. TileFlow has not yet produced consistently useful Mario-style
center-conditioned expansions. Treat v0.15 as evidence that independent cell
sampling plus accumulated guidance and post-naturalization cannot carry the
task.

Current failure diagnosis:

- The main model predicts a full grid of tile logits, then many guidance knobs
  try to make the result look structured.
- Terrain, continuation, utility, coherence, and context guidance improve
  isolated symptoms but do not give the model a stable map-level plan.
- Visual quality remains below the project goal: samples often read as arbitrary
  generated maps rather than usable Mario levels.
- Hidden target-tile reconstruction is still only a diagnostic. v1.0 should not
  chase the original hidden side regions.

External research hints to keep:

- MarioGPT: Mario maps should be represented as continuation sequences, so
  TileFlow should learn column/run progression outward from the known center.
- MarioDiffusion: context descriptors and absence captions help controllable
  generation and collapse detection.
- TOAD-GAN: small-data Mario style is strongly local-pattern based, so
  multi-scale tile-pattern statistics must be part of validation.
- PCGRL / PCGNN: playability, novelty, and diversity need explicit measurement
  and model selection pressure.
- `MARIO_GRAMMAR.md`: keep it as a soft reference for skeleton labels,
  descriptors, and diagnostics. Do not turn it into hard map repair.

Architecture:

1. **Center descriptor extraction**
   - Extract style class, ground/gap rhythm, platform row density, obstacle
     density, block/enemy/pipe density, and absence features such as no pipes,
     no enemies, or mostly empty sky.
   - Feed descriptors to every v1.0 stage.

2. **Stochastic column-level terrain skeleton**
   - Generate left and right unknown regions outward from the known center
     boundaries.
   - Predict and sample compact column states:
     `gap`, `ground_run`, `raised_ground`, `landing/island`, `mid_platform`,
     and `structure_zone`.
   - Also predict surface height bins and local motif budget
     (`low`, `medium`, `high` obstacle density).
   - Keep explicit noise input so the same center can produce multiple valid
     expansions.

3. **Skeleton-conditioned tile decoder**
   - Keep categorical/discrete flow for tile generation.
   - Condition tile logits on the sampled or soft skeleton state instead of
     relying on independent per-cell logits.
   - Train first with ground-truth skeleton labels, then with sampled skeleton
     states.

4. **Metric-based selection, not final-map repair**
   - Use playability, jump/gap viability, structural validity, descriptor
     adherence, multi-scale pattern divergence, and intra-context diversity for
     checkpoint and sample-set selection.
   - Do not rewrite completed maps into valid-looking outputs as the primary
     quality mechanism.

Training direction:

- Build v1.0 descriptor and skeleton labels from real data using
  `MARIO_GRAMMAR.md` as a soft labeling guide.
- Use tile reconstruction as a training signal only, not as the success
  criterion.
- Keep a fixed small file-level dev/val split to choose checkpoints and catch
  collapse before benchmark rendering.
- Compare against v0.15 as the last accepted pre-redesign prototype.

Evaluation additions:

- Add `tpk_kl_3x3` and `tpk_kl_4x4`.
- Add terrain-state n-gram divergence.
- Add intra-context diversity across samples from the same center.
- Add descriptor adherence metrics for generated side regions.
- Add jump/gap viability diagnostics.
- Keep fixed visual comparison on `mario-1-2.txt`, `mario-4-1.txt`, and
  `mario-6-3.txt`.

Acceptance for v1.0:

- Known center cells are preserved exactly.
- Generated maps pass shape and vocabulary contracts.
- Median rule-selected PNGs must visually read as plausible Mario maps.
- Samples should preserve center context while still showing stochastic
  variation.
- Metrics must not improve mainly by becoming empty, flat, or over-repaired.
- v1.0 artifacts must be saved in canonical form:
  - `results/checkpoints/tileflow_v1.0.pt`
  - `results/tileflow/tileflow_v1.0/`
  - `results/benchmarks/tileflow_v1.0/`
  - `results/visuals/tileflow_v1.0/`
- Rejected probes must be removed or archived according to `AGENT.md`.

## v1.1 Recovery Plan

Purpose:

v1.1 should turn the v1.0 architecture into a usable training path. Do not add
another large architecture layer yet. v1.0 already added the correct high-level
idea, but the selected artifact regressed in completion and structure and did
not visually read as a Mario map. v1.1 should diagnose and repair that specific
failure inside the model and selection loop.

v1.0 lessons:

- Descriptor distance and diversity improved, so the v1.0 conditioning path is
  useful.
- Completion and structural validity regressed, so the decoder is not turning
  skeletons into supported, playable tile structure.
- The canonical best checkpoint came from epoch `1`, before sampled-skeleton
  training, so v1.0 did not truly validate the sampled skeleton path.
- Median PNGs show singleton/floating scatter, especially in `mario-4-1` and
  `mario-6-3`.

Implementation direction:

1. **Sampled-phase checkpoint gate**
   - Split v1.1 checkpoint selection into two phases.
   - Do not allow the canonical v1.1 checkpoint to be selected before the
     sampled-skeleton phase begins.
   - Report best ground-truth-phase and best sampled-phase checkpoints
     separately in metrics.

2. **Skeleton distribution audit**
   - Add an audit comparing real labels, predicted skeleton probabilities, and
     sampled skeleton states by source file.
   - Track column state distribution, surface height distribution, motif budget
     distribution, and terrain-state n-gram divergence.
   - Use this audit before changing model capacity. If sampled skeletons are
     wrong, fix skeleton calibration first; if skeletons are plausible but
     tiles scatter, fix decoder support first.

3. **Decoder structure support**
   - Add model-side support conditioning so generated `X/S/Q/?`, enemies,
     pipes, and cannons are encouraged to appear as supported runs or valid
     structures.
   - Penalize unsupported singleton solids and unsupported entities during
     training or selection.
   - Keep this as logits/loss/model conditioning, not completed-map repair.

4. **Stochasticity and noise**
   - Add a small explicit noise plane for v1.1 only if skeleton samples collapse
     or become too deterministic after support fixes.
   - Do not implement full autoregressive/outward generation in v1.1 unless the
     skeleton audit proves parallel skeleton prediction is the core blocker.

5. **Visual-first acceptance**
   - Compare v1.1 against v1.0 and v0.15.
   - v1.1 can be accepted only if median `mario-1-2`, `mario-4-1`, and
     `mario-6-3` visuals read more like playable Mario maps.
   - Metric improvement without visual support is not sufficient.

Acceptance for v1.1:

- Known center cells are preserved exactly.
- Canonical checkpoint is selected from the sampled-skeleton phase.
- Completable rate recovers meaningfully from v1.0 toward v0.15.
- Structural violations fall substantially from v1.0.
- Intra-context diversity remains above v0.15; do not recover playability by
  collapsing back to one flat template.
- Median visual samples reduce singleton/floating scatter and show supported
  ground/platform structure.
- v1.1 artifacts are saved in canonical form:
  - `results/checkpoints/tileflow_v1.1.pt`
  - `results/tileflow/tileflow_v1.1/`
  - `results/benchmarks/tileflow_v1.1/`
  - `results/visuals/tileflow_v1.1/`

## v1.1 Result

Decision:

- v1.1 is accepted as a recovery baseline, not as final-quality Mario map
  generation.
- The selected checkpoint is from the sampled-skeleton phase.
- Structure and pattern quality recover from v1.0 without collapsing to the
  flatter v0.15 behavior.
- `mario-6-3` remains the clearest unsolved case: sparse/jump-map continuation
  still produces scattered blocks and incomplete median samples.

Final v1.1 changes:

- Sampled-phase checkpoint gating.
- Skeleton distribution audit by source file.
- Decoder support loss/conditioning.
- Support-logit soft prior for generated tile logits.
- No explicit noise plane, because skeleton audit and diversity did not show
  collapse after support fixes.

## Pre-v1.2 DFM Literature Audit

Decision:

- Before implementing v1.2, audit the DFM training formulation itself. v1.1
  improved support and checkpoint selection, but it still behaves like a local
  categorical predictor with guidance. The next step should question the
  probability path, target parameterization, scheduler, and source distribution.

Relevant hints:

- Discrete Flow Matching suggests treating discrete generation through explicit
  probability paths, learned posteriors such as `x`-prediction or
  noise-prediction, and scheduler choices. For TileFlow, this means v1.2 should
  add a real time/noise corruption training path instead of only one-shot tile
  logits.
- Flow Matching with OT paths suggests straighter probability paths can train
  and sample more efficiently than diffusion-like paths. For TileFlow, use
  context-compatible source/target pairings rather than pairing noise with maps
  uniformly.
- Dirichlet/Fisher flow work warns that naive linear paths on the simplex can be
  pathological for categorical data. For TileFlow, do not assume a simple
  softmax interpolation over tile classes is the right categorical geometry.
- CTMC-based discrete denoising work suggests state-transition kernels and
  rate-based sampling are natural for discrete variables. For TileFlow, test
  masked/CTMC-style corruption and denoising before adding more decoder heads.
- Molecule DFM work warns that satisfying local validity constraints can still
  miss higher-order motifs. For TileFlow, keep multi-scale terrain and tile
  pattern metrics as acceptance checks, not just support validity.

Pre-v1.2 conclusion:

- v1.2 should first implement a minimal DFM-path experiment:
  time-conditioned corruption, path/scheduler variants, and denoiser target
  comparison.
- Architecture growth is secondary. Do not add broad new heads until the
  probability path and target parameterization have been tested.

## v1.2 Direction Plan

Purpose:

v1.2 should improve model-side generation by testing the DFM training path
before adding more architecture. The main problem is not simply that TileFlow
lacks enough heads; v1.1 still behaves too much like one-shot categorical
prediction with support guidance. Sparse/jump-map contexts such as `mario-6-3`
need a better discrete probability path, target parameterization, and
context-aware source/coupling.

Failure hypotheses:

- H1: one-shot unknown-cell prediction is causing local tile scatter because the
  model never learns intermediate map states.
- H2: uniform or context-blind noise/source states are mismatched to Mario
  level styles, especially sparse jump maps.
- H3: v1.1's support prior improves local validity but can suppress completion
  in sparse contexts.
- H4: activation choice may affect smooth time-conditioned dynamics, but it is
  secondary to path/target/scheduler design.

Implementation order:

1. **DFM path and target audit**
   - Add time-conditioned masked/categorical corruption for unknown cells.
   - Feed `t` through a small embedding or feature plane; preserve known center
     cells exactly at every `t`.
   - Compare `x`-prediction, noise-prediction, and support-aware denoising
     targets.
   - Compare simple linear, mask/unmask, and context-adaptive categorical
     schedulers.

2. **Minimal controlled ablations**
   - Keep model size close to v1.1 while testing path/target/scheduler.
   - Run each probe on the same benchmark protocol and visual selection.
   - Reject a probe if it only improves aggregate metrics while worsening
     `mario-6-3` visual coherence.

3. **Activation ablation**
   - Keep `GELU` as the baseline.
   - Compare `SiLU` only after the time-conditioned path works.
   - Consider `SwiGLU` or `GEGLU` only if the best path is underfitting and the
     extra parameters are justified.
   - Do not prioritize `ReLU`; it is less suitable for smooth flow/time
     conditioning.

4. **Context-adaptive skeleton/source decoding**
   - Make skeleton/run generation sensitive to source style and center rhythm.
   - Treat `mario-6-3`-like sparse maps as jump/island continuation, not as
     ordinary ground recovery.
   - Prefer column-run sequence consistency over additional per-cell heads.

5. **Adaptive support prior**
   - Keep the v1.1 support-logit prior, but consider making its strength
     descriptor-dependent.
   - Stronger support bias helps `mario-1-2` and `mario-4-1`, but can suppress
     completion in sparse levels.

6. **Visual-first acceptance**
   - Require `mario-6-3` median visual/progress improvement over v1.1.
   - Preserve v1.1's structure improvement over v1.0.
   - Do not accept v1.2 if it only improves metrics by becoming empty, flat, or
     over-constrained.

Stop rules:

- If time-conditioning does not improve `mario-6-3` progress or visual
  coherence, do not add more heads; revisit corruption/source coupling first.
- If `SiLU` does not beat `GELU` on the selected path, keep `GELU`.
- If descriptor-adaptive support improves structure but hurts completion beyond
  v1.1's baseline, reject it.

Acceptance for v1.2:

- Known center cells are preserved exactly.
- Canonical artifacts use only final v1.2 names and directories.
- Main benchmark completion is at least comparable to v1.1.
- Structure error remains no worse than v1.1.
- Diversity remains above v0.15.
- `mario-6-3` median visual/progress improves over v1.1 without making
  `mario-1-2` or `mario-4-1` visibly worse.

## v1.2 Result

Decision:

- v1.2 is accepted as the current best TileFlow artifact.
- The selected model is the time-conditioned DFM path with `x` prediction,
  cosine schedule, train-prior source, `GELU`, four sampling steps, and
  support-logit bias `0.65`.
- This choice was selected over support-aware, noise-prediction, `SiLU`, and
  weaker support-bias probes because it gave the best balance of completion,
  structure, diversity, and `mario-6-3` progress.

What improved:

- Main completable rate increased from the v1.1 baseline while preserving
  known center cells exactly.
- Multi-scale local pattern quality and playable diversity improved.
- `mario-6-3` median visual/progress improved enough to accept v1.2, although
  sparse-map structure remains the largest weakness.

Remaining limits:

- `mario-6-3` still has high structure errors and low context-valid rate.
- Some `mario-1-2` samples show overlong ceiling/block runs that read less like
  natural Mario map grammar.
- Descriptor/style distance worsened versus v1.1, so v1.2 should be treated as
  a stronger playability/DFM-path baseline rather than solved style matching.

Canonical artifacts:

- `results/checkpoints/tileflow_v1.2.pt`
- `results/tileflow/tileflow_v1.2/`
- `results/benchmarks/tileflow_v1.2/`
- `results/visuals/tileflow_v1.2/`

## v1.3 Direction Plan

Purpose:

v1.3 should keep v1.2 as the DFM-path baseline and target the remaining visual
grammar failures instead of adding broad capacity. The next improvement should
focus on context-adaptive source/support coupling and sparse-map structure.

Research traceability:

- MarioGPT hint: keep treating map generation as column/run continuation from
  the center, not independent side-region cell filling.
- MarioDiffusion hint: use center descriptors, absence features, and
  descriptor adherence to decide whether sparse contexts are being followed.
- TOAD-GAN hint: keep multi-scale tile-pattern and run/mass diagnostics because
  Mario style is local-pattern heavy.
- PCGRL / PCGNN hint: keep playability, novelty, and diversity in model
  selection so the model does not collapse into one safe template.
- DFM literature hint: do not add broad heads before testing source coupling
  and probability-path choices; v1.3 should refine the v1.2 DFM path rather
  than abandon it.

Implementation direction:

1. **Sparse-context specialization**
   - Detect jump/island contexts from center descriptors and skeleton labels.
   - Reduce ordinary ground-recovery bias for sparse maps such as `mario-6-3`.
   - Track context-valid rate by source as a first-class selection signal.

2. **Run/mass diagnostics**
   - Add diagnostics for overlong ceiling/block runs, bulky non-Mario massing,
     and unsupported sparse-map structures.
   - Use these diagnostics for checkpoint/sample selection, not final-map
     repair.

3. **Adaptive support/source coupling**
   - Test descriptor-conditioned support strength instead of one global
     support-logit bias.
   - Test source distributions conditioned on center style so sparse contexts
     do not inherit dense ground priors.

4. **Visual-first acceptance**
   - Accept v1.3 only if `mario-6-3` structure/context validity improves
     without losing v1.2 completion and diversity.
   - Reject probes that improve tables while making selected PNGs less like
     playable Mario levels.

## v1.3 Result

Decision:

- v1.3 is accepted as a conservative quality upgrade over v1.2.
- The selected change is metric-based candidate selection over two stochastic
  model samples. This selects among generated maps; it does not rewrite or
  repair completed maps.
- The final model keeps v1.2's DFM path: `x` prediction, cosine schedule,
  train-prior source, `GELU`, four sampling steps, and support-logit bias
  `0.65`.

Rejected v1.3 probes:

- `style_prior` source: improved some style diagnostics but reduced completion
  or produced bulky terrain masses.
- descriptor-adaptive support bias: did not preserve v1.2 visual quality.
- gap-heavy extra sampling steps: improved some sparse-map structure metrics
  but regressed 6-3 style/progress and visual rhythm.
- `gap_style_prior`: improved descriptor distance but produced visually bulky
  or massed structures in selected PNGs.

Final v1.3 benchmark:

| Version | Completable | KL | Diversity | Struct | Continuity | Style distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1.2 | 0.7109 | 0.1142 | 0.1880 | 0.1566 | 0.8618 | 1.6242 |
| v1.3 | 0.7578 | 0.1053 | 0.1705 | 0.1579 | 0.8809 | 1.4441 |

Remaining limits:

- v1.3 trades some playable masked diversity for better completion and style
  consistency.
- `mario-1-2` still has overlong block/ceiling runs in some samples.
- `mario-6-3` improves on completion/style distance but still does not fully
  capture the sparse athletic rhythm of the source level.

Canonical artifacts:

- `results/checkpoints/tileflow_v1.3.pt`
- `results/tileflow/tileflow_v1.3/`
- `results/benchmarks/tileflow_v1.3/`
- `results/visuals/tileflow_v1.3/`

## v1.4 Direction Plan

Purpose:

v1.4 should be the final model-side polish pass before benchmark-focused work.
It should keep v1.3's candidate-selection baseline and only accept changes that
maintain the current visual quality.

Implementation direction:

1. **Block-run restraint**
   - Penalize or select against unnatural overlong ceiling/block runs,
     especially in `mario-1-2`.
   - Keep this as model-side scoring or training pressure, not tile rewriting.

2. **Sparse-map plausibility**
   - Improve `mario-6-3` sparse/jump-map rhythm without adding bulky ground
     masses or solid walls.
   - Track source-specific 6-3 visuals before accepting any aggregate gain.

3. **Diversity preservation**
   - Candidate selection improves quality but can reduce diversity.
   - v1.4 should recover some diversity while keeping v1.3's completion and
     continuity gains.

## v1.4 Result

Decision:

- v1.4 is accepted as the current TileFlow artifact.
- The accepted change keeps the v1.3 DFM path and `candidate_samples=2`, then
  adds weak pipe-pair coupling in the logits and a pipe-aware candidate score.
- This is model-side generation/selection. It does not rewrite completed maps
  into valid-looking outputs.

Rejected v1.4 probes:

- `candidate_samples=3` with stronger penalties: reduced completion, KL, style
  distance, and 6-3 quality.
- 10-epoch v1.4 retrain: did not recover v1.3 visual/metric quality.
- v1.3 weights with v1.4 inference: improved structure/diversity, but the
  20-epoch v1.4 checkpoint was better overall under checkpoint reload.

Final checkpoint-reload benchmark:

| Method | Completable | KL | Diversity | Struct | Continuity | Style distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1.3 | 0.7578 | 0.1053 | 0.1705 | 0.1579 | 0.8809 | 1.4441 |
| v1.4 | 0.7578 | 0.1059 | 0.1758 | 0.1473 | 0.8695 | 1.4446 |

Full-available benchmark context:

| Method | Completable | KL | Diversity | Struct | Continuity | Style distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mariodiffusion_full | 0.6094 | 0.0645 | 0.0859 | 0.1377 | 0.7608 | 1.7390 |
| tileflow_v1.2 | 0.7109 | 0.1198 | 0.1867 | 0.1544 | 0.8752 | 1.6753 |
| tileflow_v1.4 | 0.7578 | 0.1059 | 0.1758 | 0.1473 | 0.8695 | 1.4446 |

Remaining limits:

- `mario-1-2` still shows upper block/ceiling mass in selected visuals.
- v1.4 moves structure closer to MarioDiffusion full while preserving better
  completion, continuity, diversity, and style distance, but KL is still not as
  low as MarioDiffusion full.
- `mario-6-3` is stable relative to v1.3 but still not a fully convincing
  sparse athletic map.

Canonical artifacts:

- `results/checkpoints/tileflow_v1.4.pt`
- `results/tileflow/tileflow_v1.4/`
- `results/benchmarks/tileflow_v1.4/`
- `results/benchmarks/final_tileflow_v1.4/`
- `results/visuals/tileflow_v1.4/`

## Result Story

Separate two claims:

1. **Benchmark validation:** TileFlow can perform the shared
   center-conditioned expansion task under the same protocol as baselines.
2. **TileFlow-specific value:** generation-time structural and optional
   player-behavior constraints can steer expansions from the same center
   context.

Preferred framing:

> Preliminary diffusion experiments motivated a shift toward discrete flow
> matching because this task requires controllable generation under structural
> and playability constraints.

Avoid framing the project as broad PCG, generic inpainting, or novelty from the
model family alone.

## Novelty Guardrail

Use the narrow claim:

> TileFlow applies constraint-guided discrete flow matching to
> center-conditioned 2D tile-map expansion, with structural/playability guidance
> and optional player-behavior difficulty control.

If directly matching prior work is found, narrow the claim to the implemented
constraint set and evaluation framing.
