"""Random fill baseline for the shared center expansion benchmark."""

from __future__ import annotations

import numpy as np

from tileflow.common.data import VOCAB
from tileflow.common.fill_api import FillModel, validate_fill_io


class RandomFillBenchmark(FillModel):
    """Fill unknown cells with independent uniform samples from the tile vocab."""

    name = "random_fill"

    def __init__(self, seed: int = 42, vocab: list[str] | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.vocab = list(vocab or VOCAB)

    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        rows = [list(row) for row in validate_fill_io(level, mask, width=mask.shape[1])]
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if not mask[r, c]:
                    rows[r][c] = str(self.rng.choice(self.vocab))
        return ["".join(row) for row in rows]
