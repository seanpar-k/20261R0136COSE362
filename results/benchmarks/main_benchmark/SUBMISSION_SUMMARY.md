# Main Benchmark

This is the only benchmark set used for the project submission. The goal is to evaluate practical tile-map expansion for a solo or indie developer working with limited Mario-style training data.

## Benchmark Scope

- Task: `center_expand`
- Training/evaluation policy: same project data for every learned method
- Pretraining policy: no released Hugging Face pretrained weights for MarioGPT or MarioDiffusion
- Training budget: 20 epochs
- Included methods: TileFlow, MarioGPT same-data small, MarioDiffusion same-data small, and random_fill
- Excluded material: HF-pretrained supplementary runs, full-paper-scale comparisons, retry/smoke outputs, and baseline checkpoints

## Result

| Method | Known preserved | Completable | Fidelity KL | Playable diversity | Structure errors | Boundary continuity | Style distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_fill | pass | 0.0156 | 9.3438 | 0.9196 | 2.1900 | 0.3807 | 12.2919 |
| MarioGPT same-data small | pass | 0.6641 | 3.7219 | 0.5197 | 0.7622 | 0.5826 | 5.9498 |
| MarioDiffusion same-data small | pass | 0.0078 | 0.1041 | 0.0000 | 0.3280 | 0.5381 | 3.3548 |
| TileFlow | pass | 0.7578 | 0.1059 | 0.1758 | 0.1473 | 0.8695 | 1.4446 |

## Takeaway

With the same small dataset and no pretrained baseline weights, TileFlow is the most practical option for center-conditioned Mario-style tile-map expansion. It preserves the known center, reaches the strongest completion rate, keeps the lowest structural error, and maintains the best boundary continuity and style distance.

MarioGPT same-data small is a useful autoregressive comparison, but it is weaker on structure and style matching. MarioDiffusion same-data small captures local tile statistics, but under this small-data 20-epoch setup it does not produce useful playable expansions.
