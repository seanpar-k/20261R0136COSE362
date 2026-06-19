"""Shared fill interface for every TileFlow model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .data import DEFAULT_W, MAP_HEIGHT, normalize_level


class FillModel(ABC):
    name: str = "unnamed"

    @abstractmethod
    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        """Fill unknown cells while preserving known cells exactly."""


def validate_fill_io(level: list[str], mask: np.ndarray, width: int = DEFAULT_W) -> list[str]:
    rows = normalize_level(level, width=width)
    if mask.shape != (MAP_HEIGHT, width):
        raise ValueError(f"Expected mask shape {(MAP_HEIGHT, width)}, got {mask.shape}.")
    if mask.dtype != np.bool_ and mask.dtype != bool:
        raise TypeError("Mask must be bool with True=known and False=generated.")
    return rows


def assert_known_preserved(original: list[str], filled: list[str], mask: np.ndarray) -> None:
    rows = validate_fill_io(original, mask, width=mask.shape[1])
    filled = normalize_level(filled, width=mask.shape[1])
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and filled[r][c] != rows[r][c]:
                raise AssertionError(f"Known cell changed at row={r}, col={c}.")

