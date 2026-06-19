"""Adapters for original external Mario baseline repositories.

These adapters intentionally call the external repositories under
``external/``. They do not import archived TileFlow/MarioDiff proxy models from
``old/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from tileflow.common.data import DEFAULT_W, MAP_HEIGHT, VOCAB, normalize_level
from tileflow.common.fill_api import FillModel, validate_fill_io

ROOT = Path(__file__).resolve().parents[2]
MARIOGPT_ROOT = ROOT / "external" / "mario-gpt"
MARIODIFFUSION_ROOT = ROOT / "external" / "MarioDiffusion"


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _import_from(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def center_caption(level: list[str], mask: np.ndarray) -> str:
    """Create a deterministic Mario-style text prompt from the known center."""
    rows = validate_fill_io(level, mask, width=mask.shape[1])
    known_cols = np.where(mask.any(axis=0))[0]
    if len(known_cols) == 0:
        region = rows
    else:
        start, end = int(known_cols[0]), int(known_cols[-1]) + 1
        region = [row[start:end] for row in rows]

    width = max(len(region[0]), 1)
    floor = region[-1]
    floor_solids = sum(ch in {"X", "S", "?", "Q", "<", ">", "[", "]", "B", "b"} for ch in floor)
    floor_gaps = 0
    in_gap = False
    for ch in floor:
        is_gap = ch in {"-", "E", "o"}
        if is_gap and not in_gap:
            floor_gaps += 1
            in_gap = True
        elif not is_gap:
            in_gap = False

    phrases: list[str] = []
    if floor_solids == width:
        phrases.append("full floor")
    elif floor_solids > width // 2:
        phrases.append(f"floor with {_quantity(floor_gaps)} gap{'s' if floor_gaps != 1 else ''}")
    elif floor_solids == 0:
        pass
    else:
        phrases.append("giant gap with a few chunks of floor")

    text = "\n".join(region)
    phrases.extend(
        [
            _count_phrase(text.count("E"), "enemy", "enemies"),
            _count_phrase(text.count("?") + text.count("Q"), "question block", "question blocks"),
            _count_phrase(text.count("S"), "loose block", "loose blocks"),
            _count_phrase(text.count("<") + text.count(">") + text.count("[") + text.count("]"), "pipe", "pipes"),
            _count_phrase(text.count("B") + text.count("b"), "cannon", "cannons"),
            _count_phrase(text.count("o"), "coin", "coins"),
        ]
    )
    return ". ".join(phrase for phrase in phrases if phrase).strip() + "."


def _quantity(count: int) -> str:
    if count <= 0:
        return "no"
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    if count < 5:
        return "a few"
    if count < 10:
        return "several"
    return "many"


def _count_phrase(count: int, singular: str, plural: str) -> str:
    if count <= 0:
        return ""
    return f"{_quantity(count)} {singular if count == 1 else plural}"


def _overlay_known(original: list[str], generated: list[str], mask: np.ndarray) -> list[str]:
    original_rows = normalize_level(original, width=mask.shape[1])
    generated_rows = normalize_level(generated, width=mask.shape[1])
    out: list[str] = []
    for r in range(MAP_HEIGHT):
        chars = list(generated_rows[r])
        for c in range(mask.shape[1]):
            if mask[r, c]:
                chars[c] = original_rows[r][c]
        out.append("".join(chars))
    return out


def _sanitize_external_level(level: list[str], width: int) -> list[str]:
    rows = normalize_level(level, width=max(width, max((len(row) for row in level), default=width)))
    return normalize_level(rows, width=width)


def _decode_mariogpt_tensor(level_tensor: torch.Tensor, tokenizer, width: int) -> list[str]:
    ids = level_tensor.detach().cpu().flatten().tolist()
    chars: list[str] = []
    allowed = set(VOCAB)
    for token_id in ids:
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False).replace(" ", "")
        char = next((ch for ch in text if ch in allowed), "-")
        chars.append(char)

    if len(chars) % MAP_HEIGHT:
        chars = chars[: len(chars) - (len(chars) % MAP_HEIGHT)]
    columns = [chars[i : i + MAP_HEIGHT] for i in range(0, len(chars), MAP_HEIGHT)]
    if not columns:
        return ["-" * width for _ in range(MAP_HEIGHT)]

    rows = []
    for row in range(MAP_HEIGHT):
        rows.append("".join(column[::-1][row] for column in columns if len(column) == MAP_HEIGHT))
    return normalize_level(rows, width=width)


class MarioGPTFullAdapter(FillModel):
    """Original MarioGPT HF model projected into the center-expand contract."""

    name = "mariogpt_full"

    def __init__(
        self,
        model_path: str = "shyamsn97/Mario-GPT2-700-context-length",
        tokenizer_path: str | None = None,
        width: int = DEFAULT_W,
        temperature: float = 2.0,
        device: str = "cpu",
        count_prompter: bool = False,
    ) -> None:
        _require_path(MARIOGPT_ROOT, "external/mario-gpt checkout")
        _import_from(MARIOGPT_ROOT)
        try:
            from mario_gpt import MarioLM
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MarioGPT full adapter requires the original repo dependencies, "
                "especially transformers. Install external/mario-gpt requirements "
                "or provide an environment with mario-gpt available."
            ) from exc

        self.width = width
        self.temperature = temperature
        self.device = torch.device(device)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        prompter = None
        if count_prompter:
            from tileflow.benchmarks.mariogpt_prompt import CountFeaturePrompter

            prompter = CountFeaturePrompter()
        self.model = MarioLM(
            lm_path=model_path,
            tokenizer_path=self.tokenizer_path,
            prompter=prompter,
        ).to(self.device)
        if count_prompter:
            self.model.prompter.level_tokenizer = self.model.tokenizer
            self.model.prompter.hidden_dim = getattr(self.model.lm.config, "n_embd", 768)

    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        validate_fill_io(level, mask, width=self.width)
        prompt = center_caption(level, mask).replace(". ", ", ").rstrip(".")
        generated = self.model.sample(
            prompts=[prompt],
            num_steps=self.width * MAP_HEIGHT,
            temperature=self.temperature,
            use_tqdm=False,
        )
        if isinstance(generated, list):
            generated = generated[0]
        if generated.level is None:
            tensor = getattr(generated, "sample_predictions_tensor", None)
            if tensor is None:
                tensor = getattr(generated, "level_tensor", None)
            if tensor is None:
                raise RuntimeError("MarioGPT returned neither text nor token tensor.")
            candidate = _decode_mariogpt_tensor(tensor, self.model.tokenizer, self.width)
        else:
            candidate = _sanitize_external_level(generated.level, self.width)
        return _overlay_known(level, candidate, mask)


class MarioDiffusionFullAdapter(FillModel):
    """Original MarioDiffusion HF pipeline projected into center-expand."""

    name = "mariodiffusion_full"

    def __init__(
        self,
        model_path: str = "schrum2/MarioDiffusion-MLM-regular0",
        width: int = DEFAULT_W,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        _require_path(MARIODIFFUSION_ROOT, "external/MarioDiffusion checkout")
        _import_from(MARIODIFFUSION_ROOT)
        try:
            from captions.util import extract_tileset
            from level_dataset import convert_to_level_format
            from models.pipeline_loader import get_pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MarioDiffusion full adapter requires the original repo dependencies, "
                "especially diffusers, transformers, and huggingface_hub."
            ) from exc

        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = torch.device(device)
        self.model_path = model_path
        self.seed = seed
        self.sample_index = 0
        self.pipe = get_pipeline(model_path).to(self.device)
        self.convert_to_level_format = convert_to_level_format
        tileset = MARIODIFFUSION_ROOT / "datasets" / "smb.json"
        _, self.id_to_char, _, _ = extract_tileset(str(tileset))

    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        validate_fill_io(level, mask, width=self.width)
        caption = center_caption(level, mask)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + self.sample_index)
        self.sample_index += 1
        output = self.pipe(
            caption=caption,
            generator=generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=16,
            width=self.width,
            output_type="tensor",
            batch_size=1,
            show_progress_bar=False,
        )
        indices = self.convert_to_level_format(output.images, getattr(self.pipe, "block_embeddings", None))[0]
        rows = ["".join(self.id_to_char.get(int(tile), "-") for tile in row) for row in indices]
        candidate = _sanitize_external_level(rows[-MAP_HEIGHT:], self.width)
        candidate = ["".join(ch if ch in VOCAB else "-" for ch in row) for row in candidate]
        return _overlay_known(level, candidate, mask)
