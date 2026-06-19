# Main Benchmark

This is the only benchmark set used for the project submission. The goal is to evaluate practical tile-map expansion for a solo or indie developer working with limited Mario-style training data.

## Benchmark Scope

- Task: `center_expand`
- Training/evaluation policy: same project data for every learned method
- Pretraining policy: no released Hugging Face pretrained weights for MarioGPT or MarioDiffusion
- Training budget: 20 epochs
- TileFlow checkpoint: canonical v1.4 artifact
- Included methods: TileFlow, MarioGPT-style Causal AR, MarioDiffusion-style Same-Data, and Random Fill
- Excluded material: HF-pretrained supplementary runs, full-paper-scale comparisons, retry/smoke outputs, and baseline checkpoints
- Visual samples: rule-selected examples from the same benchmark run

## Result

| Method | Known preserved | Completable | Fidelity KL | Playable diversity | Structure errors | Boundary continuity | Style distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TileFlow | pass | 0.7578 | 0.1059 | 0.1758 | 0.1473 | 0.8695 | 1.4446 |
| MarioGPT-style Causal AR | pass | 0.1172 | 0.1301 | 0.1400 | 0.2742 | 0.7134 | 2.5565 |
| MarioDiffusion-style Same-Data | pass | 0.0078 | 0.1041 | 0.0000 | 0.3280 | 0.5381 | 3.3548 |
| Random Fill | pass | 0.0156 | 9.3438 | 0.9196 | 2.1900 | 0.3807 | 12.2919 |

## Takeaway

With the same small dataset and no pretrained baseline weights, TileFlow is the most practical option for center-conditioned Mario-style tile-map expansion. It preserves the known center, reaches the strongest completion rate, keeps the lowest structural error, and maintains the best boundary continuity and style distance.

MarioGPT-style Causal AR is a useful causal generation comparison, but the results show that left-to-right continuation alone is weak for center expansion. MarioDiffusion-style Same-Data captures local tile statistics, but under this small-data 20-epoch setup it does not produce useful playable expansions.

`MarioGPT-style Causal AR` and `MarioDiffusion-style Same-Data` are project-scale same-data baselines trained or fitted under the shared center-expansion protocol. They are not reported as full pretrained MarioGPT or MarioDiffusion results.
