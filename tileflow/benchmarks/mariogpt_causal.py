"""MarioGPT-style causal autoregressive benchmark baseline."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from tileflow.common.data import DEFAULT_W, MAP_HEIGHT, normalize_level
from tileflow.common.fill_api import FillModel, validate_fill_io


Column = tuple[str, ...]


class MarioGPTCausalAR(FillModel):
    """Column-transition proxy for MarioGPT-style left-to-right generation."""

    name = "mariogpt_causal_ar"
    sample_steps = 1

    def __init__(self, train_levels: list[list[str]], width: int = DEFAULT_W, seed: int = 42):
        self.width = width
        self.rng = np.random.default_rng(seed)
        self.start_counts: Counter[Column] = Counter()
        self.transition_counts: dict[Column, Counter[Column]] = defaultdict(Counter)
        self.column_counts: Counter[Column] = Counter()
        self._fit(train_levels)

    def _fit(self, train_levels: list[list[str]]) -> None:
        for level in train_levels:
            rows = normalize_level(level, width=self.width)
            columns = [tuple(rows[r][c] for r in range(MAP_HEIGHT)) for c in range(self.width)]
            if not columns:
                continue
            self.start_counts[columns[0]] += 1
            self.column_counts.update(columns)
            for prev, nxt in zip(columns, columns[1:]):
                self.transition_counts[prev][nxt] += 1

        if not self.column_counts:
            empty = tuple("-" for _ in range(MAP_HEIGHT))
            self.start_counts[empty] = 1
            self.column_counts[empty] = 1

    def _sample_counter(self, counter: Counter[Column]) -> Column:
        items = list(counter.items())
        values = [item[0] for item in items]
        weights = np.array([item[1] for item in items], dtype=np.float64)
        weights = weights / weights.sum()
        return values[int(self.rng.choice(len(values), p=weights))]

    def _sample_next_column(self, prev: Column | None) -> Column:
        if prev is not None and prev in self.transition_counts:
            return self._sample_counter(self.transition_counts[prev])
        if prev is None and self.start_counts:
            return self._sample_counter(self.start_counts)
        return self._sample_counter(self.column_counts)

    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        rows = [list(row) for row in validate_fill_io(level, mask, width=self.width)]
        prev: Column | None = None
        for c in range(self.width):
            if mask[:, c].all():
                prev = tuple(rows[r][c] for r in range(MAP_HEIGHT))
                continue
            candidate = self._sample_next_column(prev)
            for r in range(MAP_HEIGHT):
                if not mask[r, c]:
                    rows[r][c] = candidate[r]
            prev = tuple(rows[r][c] for r in range(MAP_HEIGHT))
        return ["".join(row) for row in rows]
