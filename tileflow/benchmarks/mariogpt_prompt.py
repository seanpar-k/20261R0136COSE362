"""Non-pretrained prompt features for same-data MarioGPT baselines."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


class CountFeaturePrompter:
    """MarioGPT-compatible prompt encoder without pretrained text weights."""

    def __init__(self, level_tokenizer: Any | None = None, hidden_dim: int = 768) -> None:
        self.level_tokenizer = level_tokenizer
        self.hidden_dim = hidden_dim

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        features = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            features[i % self.hidden_dim] += (byte / 255.0) - 0.5
        features[0] = prompt.count("pipe")
        features[1] = prompt.count("enem")
        features[2] = prompt.count("block")
        features[3] = prompt.count("coin")
        features[4] = prompt.count("floor")
        features[5] = prompt.count("gap")
        norm = features.norm()
        if norm > 0:
            features = features / norm
        return features.view(1, -1)

    def __call__(
        self,
        level: torch.Tensor | None = None,
        sample_prompt: bool = False,
    ):
        if sample_prompt or level is None or self.level_tokenizer is None:
            prompt = "some pipes, some enemies, some blocks, low elevation"
            return prompt, self.output_hidden(prompt), {}, None
        text = self.level_tokenizer.decode(level.detach().cpu()).replace(" ", "")
        pipe_count = text.count("<") + text.count(">") + text.count("[") + text.count("]")
        enemy_count = text.count("E") + text.count("B")
        block_count = sum(text.count(ch) for ch in ["X", "S", "?", "Q"])
        elevation = "high elevation" if any(ch in text[: max(1, len(text) // 2)] for ch in ["X", "<", ">"]) else "low elevation"
        prompt = (
            f"{_bucket(pipe_count)} pipes, "
            f"{_bucket(enemy_count)} enemies, "
            f"{_bucket(block_count)} blocks, "
            f"{elevation}"
        )
        return prompt, self.output_hidden(prompt, device=level.device), {}, None


def _bucket(count: int) -> str:
    if count <= 0:
        return "no"
    if count < 4:
        return "little"
    if count < 12:
        return "some"
    return "many"
