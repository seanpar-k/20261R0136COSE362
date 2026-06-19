# Benchmark Visual Sets

Generated visual subsets for human inspection. The full benchmark sample
archive remains under each `results/benchmarks/<method>/` directory and is
used for quantitative metrics.

Selection policy:

- `main`: median-quality sample for each method/source pair
- `appendix`: best, median, and worst samples for each method/source pair
- contexts use the same deterministic canonical window per source
- lower selection score is better

## Canonical Windows

| Source | Window |
| --- | --- |
| mario-1-2 | window_004 |
| mario-4-1 | window_007 |
| mario-6-3 | window_004 |

## Main Set

| Source | Method | Image | Score |
| --- | --- | --- | --- |
| mario-1-2 | mariodiffusion_full | ![mariodiffusion_full mario-1-2](main/mario-1-2/mariodiffusion_full__mario-1-2__window_004__sample_003__median.png) | 3.7681 |
| mario-1-2 | random_fill | ![random_fill mario-1-2](main/mario-1-2/random_fill__mario-1-2__window_004__sample_002__median.png) | 23.4198 |
| mario-1-2 | tileflow_v1.2 | ![tileflow_v1.2 mario-1-2](main/mario-1-2/tileflow_v1.2__mario-1-2__window_004__sample_000__median.png) | 2.3305 |
| mario-4-1 | mariodiffusion_full | ![mariodiffusion_full mario-4-1](main/mario-4-1/mariodiffusion_full__mario-4-1__window_007__sample_002__median.png) | 1.6116 |
| mario-4-1 | random_fill | ![random_fill mario-4-1](main/mario-4-1/random_fill__mario-4-1__window_007__sample_001__median.png) | 27.4746 |
| mario-4-1 | tileflow_v1.2 | ![tileflow_v1.2 mario-4-1](main/mario-4-1/tileflow_v1.2__mario-4-1__window_007__sample_003__median.png) | 1.4860 |
| mario-6-3 | mariodiffusion_full | ![mariodiffusion_full mario-6-3](main/mario-6-3/mariodiffusion_full__mario-6-3__window_004__sample_000__median.png) | 15.4919 |
| mario-6-3 | random_fill | ![random_fill mario-6-3](main/mario-6-3/random_fill__mario-6-3__window_004__sample_001__median.png) | 28.3496 |
| mario-6-3 | tileflow_v1.2 | ![tileflow_v1.2 mario-6-3](main/mario-6-3/tileflow_v1.2__mario-6-3__window_004__sample_001__median.png) | 16.6468 |

## Appendix Set

| Source | Method | Role | Image | Score |
| --- | --- | --- | --- | --- |
| mario-1-2 | mariodiffusion_full | best | ![best mariodiffusion_full mario-1-2](appendix/mario-1-2/mariodiffusion_full__mario-1-2__window_004__sample_000__best.png) | 3.5306 |
| mario-1-2 | mariodiffusion_full | median | ![median mariodiffusion_full mario-1-2](appendix/mario-1-2/mariodiffusion_full__mario-1-2__window_004__sample_003__median.png) | 3.7681 |
| mario-1-2 | mariodiffusion_full | worst | ![worst mariodiffusion_full mario-1-2](appendix/mario-1-2/mariodiffusion_full__mario-1-2__window_004__sample_001__worst.png) | 3.9351 |
| mario-1-2 | random_fill | best | ![best random_fill mario-1-2](appendix/mario-1-2/random_fill__mario-1-2__window_004__sample_001__best.png) | 22.1503 |
| mario-1-2 | random_fill | median | ![median random_fill mario-1-2](appendix/mario-1-2/random_fill__mario-1-2__window_004__sample_002__median.png) | 23.4198 |
| mario-1-2 | random_fill | worst | ![worst random_fill mario-1-2](appendix/mario-1-2/random_fill__mario-1-2__window_004__sample_000__worst.png) | 24.5995 |
| mario-1-2 | tileflow_v1.2 | best | ![best tileflow_v1.2 mario-1-2](appendix/mario-1-2/tileflow_v1.2__mario-1-2__window_004__sample_001__best.png) | 1.7190 |
| mario-1-2 | tileflow_v1.2 | median | ![median tileflow_v1.2 mario-1-2](appendix/mario-1-2/tileflow_v1.2__mario-1-2__window_004__sample_000__median.png) | 2.3305 |
| mario-1-2 | tileflow_v1.2 | worst | ![worst tileflow_v1.2 mario-1-2](appendix/mario-1-2/tileflow_v1.2__mario-1-2__window_004__sample_003__worst.png) | 3.6629 |
| mario-4-1 | mariodiffusion_full | best | ![best mariodiffusion_full mario-4-1](appendix/mario-4-1/mariodiffusion_full__mario-4-1__window_007__sample_003__best.png) | 1.6091 |
| mario-4-1 | mariodiffusion_full | median | ![median mariodiffusion_full mario-4-1](appendix/mario-4-1/mariodiffusion_full__mario-4-1__window_007__sample_002__median.png) | 1.6116 |
| mario-4-1 | mariodiffusion_full | worst | ![worst mariodiffusion_full mario-4-1](appendix/mario-4-1/mariodiffusion_full__mario-4-1__window_007__sample_000__worst.png) | 1.8156 |
| mario-4-1 | random_fill | best | ![best random_fill mario-4-1](appendix/mario-4-1/random_fill__mario-4-1__window_007__sample_003__best.png) | 26.9695 |
| mario-4-1 | random_fill | median | ![median random_fill mario-4-1](appendix/mario-4-1/random_fill__mario-4-1__window_007__sample_001__median.png) | 27.4746 |
| mario-4-1 | random_fill | worst | ![worst random_fill mario-4-1](appendix/mario-4-1/random_fill__mario-4-1__window_007__sample_002__worst.png) | 29.4080 |
| mario-4-1 | tileflow_v1.2 | best | ![best tileflow_v1.2 mario-4-1](appendix/mario-4-1/tileflow_v1.2__mario-4-1__window_007__sample_001__best.png) | 0.7618 |
| mario-4-1 | tileflow_v1.2 | median | ![median tileflow_v1.2 mario-4-1](appendix/mario-4-1/tileflow_v1.2__mario-4-1__window_007__sample_003__median.png) | 1.4860 |
| mario-4-1 | tileflow_v1.2 | worst | ![worst tileflow_v1.2 mario-4-1](appendix/mario-4-1/tileflow_v1.2__mario-4-1__window_007__sample_002__worst.png) | 1.9269 |
| mario-6-3 | mariodiffusion_full | best | ![best mariodiffusion_full mario-6-3](appendix/mario-6-3/mariodiffusion_full__mario-6-3__window_004__sample_002__best.png) | 15.3457 |
| mario-6-3 | mariodiffusion_full | median | ![median mariodiffusion_full mario-6-3](appendix/mario-6-3/mariodiffusion_full__mario-6-3__window_004__sample_000__median.png) | 15.4919 |
| mario-6-3 | mariodiffusion_full | worst | ![worst mariodiffusion_full mario-6-3](appendix/mario-6-3/mariodiffusion_full__mario-6-3__window_004__sample_001__worst.png) | 15.7203 |
| mario-6-3 | random_fill | best | ![best random_fill mario-6-3](appendix/mario-6-3/random_fill__mario-6-3__window_004__sample_003__best.png) | 27.1880 |
| mario-6-3 | random_fill | median | ![median random_fill mario-6-3](appendix/mario-6-3/random_fill__mario-6-3__window_004__sample_001__median.png) | 28.3496 |
| mario-6-3 | random_fill | worst | ![worst random_fill mario-6-3](appendix/mario-6-3/random_fill__mario-6-3__window_004__sample_002__worst.png) | 28.7622 |
| mario-6-3 | tileflow_v1.2 | best | ![best tileflow_v1.2 mario-6-3](appendix/mario-6-3/tileflow_v1.2__mario-6-3__window_004__sample_003__best.png) | 4.3347 |
| mario-6-3 | tileflow_v1.2 | median | ![median tileflow_v1.2 mario-6-3](appendix/mario-6-3/tileflow_v1.2__mario-6-3__window_004__sample_001__median.png) | 16.6468 |
| mario-6-3 | tileflow_v1.2 | worst | ![worst tileflow_v1.2 mario-6-3](appendix/mario-6-3/tileflow_v1.2__mario-6-3__window_004__sample_000__worst.png) | 16.7766 |
