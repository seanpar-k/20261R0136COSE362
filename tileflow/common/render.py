"""Matplotlib rendering helpers for TileFlow text levels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .data import MAP_HEIGHT, normalize_level

TILE_COLORS = {
    "-": [0.53, 0.81, 0.98],
    "X": [0.55, 0.35, 0.15],
    "S": [0.95, 0.75, 0.10],
    "Q": [1.00, 0.55, 0.00],
    "E": [0.20, 0.75, 0.25],
    "?": [1.00, 0.90, 0.10],
    "<": [0.20, 0.55, 0.20],
    ">": [0.20, 0.55, 0.20],
    "[": [0.15, 0.45, 0.15],
    "]": [0.15, 0.45, 0.15],
    "B": [0.80, 0.50, 0.20],
    "b": [0.70, 0.45, 0.18],
    "o": [1.00, 1.00, 0.50],
}
DEFAULT_COLOR = [0.75, 0.75, 0.75]


def level_to_image(level: list[str]) -> np.ndarray:
    rows = normalize_level(level, width=max(len(row) for row in level))
    width = len(rows[0])
    img = np.ones((MAP_HEIGHT, width, 3), dtype=np.float32)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            img[r, c] = TILE_COLORS.get(ch, DEFAULT_COLOR)
    return img


def render_level(
    level: list[str],
    output_path: str | Path,
    title: str = "",
    mask: np.ndarray | None = None,
    tile_px: float = 0.24,
    show_overlay: bool = False,
) -> None:
    rows = normalize_level(level, width=max(len(row) for row in level))
    width = len(rows[0])
    img = level_to_image(rows)

    fig, ax = plt.subplots(figsize=(max(width * tile_px, 6.0), MAP_HEIGHT * 0.42))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)

    for x in range(10, width, 10):
        ax.axvline(x=x - 0.5, color="white", lw=0.5, alpha=0.55)

    if mask is not None:
        unknown = ~mask
        if show_overlay:
            overlay = np.zeros((MAP_HEIGHT, width, 4), dtype=np.float32)
            overlay[unknown] = [1.0, 0.0, 0.0, 0.10]
            ax.imshow(overlay, aspect="auto", interpolation="nearest")
        for c in range(width):
            col = unknown[:, c]
            if col.any():
                if c == 0 or not unknown[:, c - 1].any():
                    ax.axvline(c - 0.5, color="red", lw=1.2, alpha=0.8)
                if c == width - 1 or not unknown[:, c + 1].any():
                    ax.axvline(c + 0.5, color="red", lw=1.2, alpha=0.8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_text_level(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()
