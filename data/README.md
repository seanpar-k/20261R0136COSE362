# TileFlow Data

This folder contains the long-term level text files used by TileFlow training and evaluation.

## Sources

- `mario-*.txt`: Super Mario Bros processed VGLC levels already used by this project.
- `lost-levels-*.txt`: VGLC `Super Mario Bros 2 (Japan)` / Lost Levels processed levels, copied from `Super Mario Bros 2 (Japan)/Processed/`.

VGLC repository: https://github.com/TheVGLC/TheVGLC

VGLC paper: https://arxiv.org/abs/1606.07487

License: MIT, from the VGLC repository `License.md`.

## Current Fixed Evaluation Split

The fixed held-out SMB1 eval files are:

- `mario-1-2.txt`
- `mario-4-1.txt`
- `mario-6-3.txt`

All other `*.txt` files in this folder, including every `lost-levels-*.txt` file, are training files when using `make_level_splits_with_eval_files(...)`.

## Validation Notes

- Added Lost Levels file count: 22
- Total top-level map files after expansion: 37
- Top-level Lost Levels files were preprocessed to exactly 14 rows for training.
- Raw downloaded Lost Levels files are preserved under `data/raw/lost_levels_original/`.
- Lost Levels unknown vocabulary characters: none
- Lost Levels observed tile characters: `-<>?BEQSX[]bo`
- Preprocessing uses bottom alignment: shorter raw maps get sky rows prepended, and taller raw maps keep the bottom 14 rows. This preserves ground and obstacle row placement better than padding below the level.
