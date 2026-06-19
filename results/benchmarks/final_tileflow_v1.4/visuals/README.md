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
| mario-1-2 | tileflow_v1.4 | ![tileflow_v1.4 mario-1-2](main/mario-1-2/tileflow_v1.4__mario-1-2__window_004__sample_000__median.png) | 2.2021 |
| mario-4-1 | tileflow_v1.4 | ![tileflow_v1.4 mario-4-1](main/mario-4-1/tileflow_v1.4__mario-4-1__window_007__sample_003__median.png) | 0.7658 |
| mario-6-3 | tileflow_v1.4 | ![tileflow_v1.4 mario-6-3](main/mario-6-3/tileflow_v1.4__mario-6-3__window_004__sample_003__median.png) | 5.7238 |

## Appendix Set

| Source | Method | Role | Image | Score |
| --- | --- | --- | --- | --- |
| mario-1-2 | tileflow_v1.4 | best | ![best tileflow_v1.4 mario-1-2](appendix/mario-1-2/tileflow_v1.4__mario-1-2__window_004__sample_001__best.png) | 1.6967 |
| mario-1-2 | tileflow_v1.4 | median | ![median tileflow_v1.4 mario-1-2](appendix/mario-1-2/tileflow_v1.4__mario-1-2__window_004__sample_000__median.png) | 2.2021 |
| mario-1-2 | tileflow_v1.4 | worst | ![worst tileflow_v1.4 mario-1-2](appendix/mario-1-2/tileflow_v1.4__mario-1-2__window_004__sample_002__worst.png) | 2.3089 |
| mario-4-1 | tileflow_v1.4 | best | ![best tileflow_v1.4 mario-4-1](appendix/mario-4-1/tileflow_v1.4__mario-4-1__window_007__sample_001__best.png) | 0.6091 |
| mario-4-1 | tileflow_v1.4 | median | ![median tileflow_v1.4 mario-4-1](appendix/mario-4-1/tileflow_v1.4__mario-4-1__window_007__sample_003__median.png) | 0.7658 |
| mario-4-1 | tileflow_v1.4 | worst | ![worst tileflow_v1.4 mario-4-1](appendix/mario-4-1/tileflow_v1.4__mario-4-1__window_007__sample_000__worst.png) | 2.2838 |
| mario-6-3 | tileflow_v1.4 | best | ![best tileflow_v1.4 mario-6-3](appendix/mario-6-3/tileflow_v1.4__mario-6-3__window_004__sample_001__best.png) | 4.1706 |
| mario-6-3 | tileflow_v1.4 | median | ![median tileflow_v1.4 mario-6-3](appendix/mario-6-3/tileflow_v1.4__mario-6-3__window_004__sample_003__median.png) | 5.7238 |
| mario-6-3 | tileflow_v1.4 | worst | ![worst tileflow_v1.4 mario-6-3](appendix/mario-6-3/tileflow_v1.4__mario-6-3__window_004__sample_002__worst.png) | 5.9820 |
