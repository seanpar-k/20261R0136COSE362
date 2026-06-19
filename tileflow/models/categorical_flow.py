"""Small categorical TileFlow prototype for center-conditioned expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tileflow.common.data import CHAR2IDX, DEFAULT_W, IDX2CHAR, MAP_HEIGHT, VOCAB, normalize_level
from tileflow.common.eval import playability, structural_violations
from tileflow.common.fill_api import FillModel, validate_fill_io
from tileflow.common.style import style_descriptor, style_metrics

OBSTACLE_TILES = ("S", "Q", "?", "E", "<", ">", "[", "]", "B", "b", "o")
BLOCK_TILES = ("S", "Q", "?")
PIPE_TILES = ("<", ">", "[", "]")
CANNON_TILES = ("B", "b")
ENEMY_TILES = ("E",)
COIN_TILES = ("o",)

TERRAIN_AIR = 0
TERRAIN_STABLE_GROUND = 1
TERRAIN_GAP = 2
TERRAIN_LANDING = 3
TERRAIN_BAD_TOOTH = 4

CONTINUATION_OTHER = 0
CONTINUATION_CONTEXT_RUN = 1
CONTINUATION_LANDING_SPAN = 2
CONTINUATION_NOISE = 3

SKELETON_GAP = 0
SKELETON_GROUND_RUN = 1
SKELETON_RAISED_GROUND = 2
SKELETON_LANDING_ISLAND = 3
SKELETON_MID_PLATFORM = 4
SKELETON_STRUCTURE_ZONE = 5

SKELETON_HEIGHT_NONE = 0
SKELETON_HEIGHT_1 = 1
SKELETON_HEIGHT_2 = 2
SKELETON_HEIGHT_3 = 3
SKELETON_HEIGHT_4_PLUS = 4
SKELETON_HEIGHT_MID_PLATFORM = 5

MOTIF_LOW = 0
MOTIF_MEDIUM = 1
MOTIF_HIGH = 2


@dataclass(frozen=True)
class TileFlowConfig:
    version: str
    width: int = DEFAULT_W
    hidden: int = 48
    dilations: tuple[int, ...] = (1, 1, 1, 1)
    class_weighted_loss: bool = False
    position_channels: bool = False
    structure_heads: bool = False
    utility_heads: bool = False
    context_channels: bool = False
    strict_utility_labels: bool = False
    terrain_heads: bool = False
    continuation_heads: bool = False
    skeleton_heads: bool = False
    skeleton_conditioning: bool = False
    stochastic_decode: bool = False
    support_conditioning: bool = False
    support_logit_bias: float = 0.0
    activation: str = "gelu"
    time_conditioning: bool = False
    dfm_target: str = "x"
    dfm_schedule: str = "linear"
    dfm_source: str = "air"
    dfm_source_probs: tuple[float, ...] | None = None
    dfm_source_style_probs: tuple[tuple[float, ...], ...] | None = None
    sample_steps: int = 1
    gap_heavy_sample_steps: int = 0
    candidate_samples: int = 1
    adaptive_support_bias: bool = False


def activation_layer(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, activation: str = "gelu") -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            activation_layer(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.act = activation_layer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CategoricalFlowNet(nn.Module):
    """Predict categorical tile states from known center context and mask planes."""

    def __init__(self, config: TileFlowConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = (
            len(VOCAB)
            + 2
            + (2 if config.position_channels else 0)
            + (19 if config.context_channels else 0)
            + (2 if config.time_conditioning else 0)
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, config.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, config.hidden),
            activation_layer(config.activation),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(config.hidden, d, config.activation) for d in config.dilations))
        self.head = nn.Conv2d(config.hidden, len(VOCAB), kernel_size=1)
        self.skeleton_state_head = nn.Conv2d(config.hidden, 6, kernel_size=1) if config.skeleton_heads else None
        self.skeleton_height_head = nn.Conv2d(config.hidden, 6, kernel_size=1) if config.skeleton_heads else None
        self.motif_budget_head = nn.Conv2d(config.hidden, 3, kernel_size=1) if config.skeleton_heads else None
        self.skeleton_proj = (
            nn.Sequential(
                nn.Conv2d(15, config.hidden, kernel_size=1),
                activation_layer(config.activation),
                nn.Conv2d(config.hidden, config.hidden, kernel_size=1),
            )
            if config.skeleton_conditioning
            else None
        )
        self.support_head = nn.Conv2d(config.hidden, 4, kernel_size=1) if config.structure_heads else None
        self.landing_head = nn.Conv2d(config.hidden, 3, kernel_size=1) if config.structure_heads else None
        self.support_proj = (
            nn.Sequential(
                nn.Conv2d(7, config.hidden, kernel_size=1),
                activation_layer(config.activation),
                nn.Conv2d(config.hidden, config.hidden, kernel_size=1),
            )
            if config.support_conditioning
            else None
        )
        self.utility_head = nn.Conv2d(config.hidden, 6, kernel_size=1) if config.utility_heads else None
        self.terrain_head = nn.Conv2d(config.hidden, 5, kernel_size=1) if config.terrain_heads else None
        self.continuation_head = nn.Conv2d(config.hidden, 4, kernel_size=1) if config.continuation_heads else None

    def _skeleton_logits(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.skeleton_state_head is None or self.skeleton_height_head is None or self.motif_budget_head is None:
            return {}
        pooled = features.mean(dim=2, keepdim=True)
        return {
            "skeleton_state": self.skeleton_state_head(pooled),
            "skeleton_height": self.skeleton_height_head(pooled),
            "motif_budget": self.motif_budget_head(pooled),
        }

    def _conditioned_features(
        self,
        features: torch.Tensor,
        skeleton: dict[str, torch.Tensor],
        skeleton_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.skeleton_proj is None or not skeleton:
            return features
        height = features.shape[2]
        if skeleton_targets is None:
            probs = torch.cat(
                [
                    torch.softmax(skeleton["skeleton_state"], dim=1).expand(-1, -1, height, -1),
                    torch.softmax(skeleton["skeleton_height"], dim=1).expand(-1, -1, height, -1),
                    torch.softmax(skeleton["motif_budget"], dim=1).expand(-1, -1, height, -1),
                ],
                dim=1,
            )
        else:
            state, surface_height, motif = skeleton_targets
            probs = torch.cat(
                [
                    torch.nn.functional.one_hot(state, num_classes=6).permute(0, 2, 1).unsqueeze(2).float(),
                    torch.nn.functional.one_hot(surface_height, num_classes=6).permute(0, 2, 1).unsqueeze(2).float(),
                    torch.nn.functional.one_hot(motif, num_classes=3).permute(0, 2, 1).unsqueeze(2).float(),
                ],
                dim=1,
            ).expand(-1, -1, height, -1)
        return features + self.skeleton_proj(probs)

    def _support_conditioned_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.support_proj is None or self.support_head is None or self.landing_head is None:
            return features
        height = features.shape[2]
        support = torch.softmax(self.support_head(features), dim=1)
        landing = torch.softmax(self.landing_head(features).mean(dim=2, keepdim=True), dim=1).expand(-1, -1, height, -1)
        return features + self.support_proj(torch.cat([support, landing], dim=1))

    def _support_biased_logits(
        self,
        logits: torch.Tensor,
        support_logits: torch.Tensor | None,
        landing_logits: torch.Tensor | None,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        strength = float(self.config.support_logit_bias)
        if strength <= 0.0 or support_logits is None or landing_logits is None:
            return logits

        strength_t: float | torch.Tensor = strength
        if self.config.adaptive_support_bias and x is not None and self.config.context_channels:
            base = len(VOCAB) + 2 + (2 if self.config.position_channels else 0)
            gap_heavy = x[:, base + 12 : base + 13, :1, :1]
            obstacle_heavy = x[:, base + 13 : base + 14, :1, :1]
            factor = (1.05 - 0.20 * gap_heavy - 0.08 * obstacle_heavy).clamp(0.75, 1.10)
            strength_t = strength * factor

        eps = 1e-4
        support = torch.softmax(support_logits, dim=1).clamp_min(eps)
        landing = torch.softmax(landing_logits.mean(dim=2, keepdim=True), dim=1).expand_as(landing_logits).clamp_min(eps)
        air_prior = (support[:, 0:1] + 0.15 * landing[:, 0:1]).clamp(eps, 1.0)
        solid_prior = (support[:, 1:3].sum(dim=1, keepdim=True) + 0.35 * landing[:, 2:3]).clamp(eps, 1.0)
        surface_prior = (support[:, 2:3] + 0.20 * solid_prior).clamp(eps, 1.0)
        entity_prior = (support[:, 3:4] + 0.20 * solid_prior).clamp(eps, 1.0)

        biased = logits.clone()
        air_id = VOCAB.index("-")
        biased[:, air_id : air_id + 1] = biased[:, air_id : air_id + 1] + strength_t * torch.log(air_prior)
        for ch in "X<>[]Bb":
            if ch in VOCAB:
                idx = VOCAB.index(ch)
                biased[:, idx : idx + 1] = biased[:, idx : idx + 1] + strength_t * torch.log(solid_prior)
        for ch in "SQ?":
            if ch in VOCAB:
                idx = VOCAB.index(ch)
                biased[:, idx : idx + 1] = biased[:, idx : idx + 1] + strength_t * torch.log(surface_prior)
        if "E" in VOCAB:
            idx = VOCAB.index("E")
            biased[:, idx : idx + 1] = biased[:, idx : idx + 1] + strength_t * torch.log(entity_prior)
        return biased

    def _sample_skeleton_targets(
        self,
        skeleton: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
        temperature: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = []
        for key in ("skeleton_state", "skeleton_height", "motif_budget"):
            logits = skeleton[key].squeeze(2).permute(0, 2, 1).contiguous()
            flat = logits.view(-1, logits.shape[-1]) / max(temperature, 1e-6)
            probs = torch.softmax(flat, dim=1)
            if generator is not None and torch.device(generator.device) != probs.device:
                generator = None
            sampled = torch.multinomial(probs, num_samples=1, generator=generator).view(logits.shape[0], logits.shape[1])
            targets.append(sampled)
        return targets[0], targets[1], targets[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        features = self._support_conditioned_features(features)
        skeleton = self._skeleton_logits(features)
        conditioned = self._conditioned_features(features, skeleton)
        support_logits = self.support_head(features) if self.support_head is not None else None
        landing_logits = self.landing_head(features) if self.landing_head is not None else None
        return self._support_biased_logits(self.head(conditioned), support_logits, landing_logits, x)

    def forward_all(
        self,
        x: torch.Tensor,
        skeleton_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(x))
        features = self._support_conditioned_features(features)
        skeleton = self._skeleton_logits(features)
        support_logits = self.support_head(features) if self.support_head is not None else None
        landing_logits = self.landing_head(features) if self.landing_head is not None else None
        tile_logits = self.head(self._conditioned_features(features, skeleton, skeleton_targets))
        out = {"tile": self._support_biased_logits(tile_logits, support_logits, landing_logits, x)}
        out.update(skeleton)
        if support_logits is not None:
            out["support"] = support_logits
        if landing_logits is not None:
            out["landing"] = landing_logits
        if self.utility_head is not None:
            out["utility"] = self.utility_head(features)
        if self.terrain_head is not None:
            out["terrain"] = self.terrain_head(features)
        if self.continuation_head is not None:
            out["continuation"] = self.continuation_head(features)
        return out

    def forward_sampled_skeleton(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        temperature: float = 0.9,
    ) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(x))
        features = self._support_conditioned_features(features)
        skeleton = self._skeleton_logits(features)
        if not skeleton:
            return self.forward_all(x)
        skeleton_targets = self._sample_skeleton_targets(skeleton, generator=generator, temperature=temperature)
        support_logits = self.support_head(features) if self.support_head is not None else None
        landing_logits = self.landing_head(features) if self.landing_head is not None else None
        tile_logits = self.head(self._conditioned_features(features, skeleton, skeleton_targets))
        out = {"tile": self._support_biased_logits(tile_logits, support_logits, landing_logits)}
        out.update(skeleton)
        if support_logits is not None:
            out["support"] = support_logits
        if landing_logits is not None:
            out["landing"] = landing_logits
        if self.utility_head is not None:
            out["utility"] = self.utility_head(features)
        if self.terrain_head is not None:
            out["terrain"] = self.terrain_head(features)
        if self.continuation_head is not None:
            out["continuation"] = self.continuation_head(features)
        out["sampled_skeleton_state"] = skeleton_targets[0]
        out["sampled_skeleton_height"] = skeleton_targets[1]
        out["sampled_motif_budget"] = skeleton_targets[2]
        return out


def level_to_idx(level: list[str], width: int = DEFAULT_W) -> np.ndarray:
    rows = normalize_level(level, width=width)
    return np.array([[CHAR2IDX[ch] for ch in row] for row in rows], dtype=np.int64)


def idx_to_level(idx: np.ndarray) -> list[str]:
    idx = np.asarray(idx)
    return ["".join(IDX2CHAR[int(idx[r, c])] for c in range(idx.shape[1])) for r in range(MAP_HEIGHT)]


def make_model_input(
    idx: np.ndarray,
    mask: np.ndarray,
    device: torch.device | str = "cpu",
    position_channels: bool = False,
    context_channels: bool = False,
    reveal_unknown: bool = False,
    time_value: float | None = None,
) -> torch.Tensor:
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    onehot = torch.nn.functional.one_hot(idx_t, num_classes=len(VOCAB)).permute(2, 0, 1).float()
    if not reveal_unknown:
        onehot = onehot * mask_t.unsqueeze(0).float()
    known = mask_t.unsqueeze(0).float()
    unknown = (~mask_t).unsqueeze(0).float()
    planes = [onehot, known, unknown]
    if position_channels:
        rows = torch.linspace(-1.0, 1.0, steps=MAP_HEIGHT, device=device).view(1, MAP_HEIGHT, 1)
        cols = torch.linspace(-1.0, 1.0, steps=mask.shape[1], device=device).view(1, 1, mask.shape[1])
        planes.append(rows.expand(1, MAP_HEIGHT, mask.shape[1]))
        planes.append(cols.expand(1, MAP_HEIGHT, mask.shape[1]))
    if context_channels:
        planes.extend(center_context_planes(idx_t, mask_t))
    if time_value is not None:
        t = float(time_value)
        planes.append(torch.full((1, MAP_HEIGHT, mask.shape[1]), t, device=device))
        planes.append(torch.full((1, MAP_HEIGHT, mask.shape[1]), 1.0 - t, device=device))
    return torch.cat(planes, dim=0)


def center_context_planes(idx_t: torch.Tensor, mask_t: torch.Tensor) -> list[torch.Tensor]:
    """Coarse center summaries that avoid direct tile-copy conditioning."""
    width = mask_t.shape[1]
    known = mask_t.float()
    known_cols = mask_t.any(dim=0)
    col_count = known.sum(dim=1).clamp_min(1.0)

    solid_ids = torch.as_tensor([CHAR2IDX[ch] for ch in "XSQ?<>[]Bb"], device=idx_t.device)
    platform_ids = torch.as_tensor([CHAR2IDX[ch] for ch in "XSQ?"], device=idx_t.device)
    obstacle_ids = torch.as_tensor([CHAR2IDX[ch] for ch in "SQ?E<>[]Bbo"], device=idx_t.device)
    pipe_ids = torch.as_tensor([CHAR2IDX[ch] for ch in "<>[]"], device=idx_t.device)
    enemy_ids = torch.as_tensor([CHAR2IDX["E"]], device=idx_t.device)
    block_ids = torch.as_tensor([CHAR2IDX[ch] for ch in "SQ?"], device=idx_t.device)
    solid = torch.isin(idx_t, solid_ids) & mask_t
    platform = torch.isin(idx_t, platform_ids) & mask_t
    obstacle = torch.isin(idx_t, obstacle_ids) & mask_t
    pipes = torch.isin(idx_t, pipe_ids) & mask_t
    enemies = torch.isin(idx_t, enemy_ids) & mask_t
    blocks = torch.isin(idx_t, block_ids) & mask_t

    row_solid = (solid.float().sum(dim=1) / col_count).view(1, MAP_HEIGHT, 1).expand(1, MAP_HEIGHT, width)
    row_platform = (platform.float().sum(dim=1) / col_count).view(1, MAP_HEIGHT, 1).expand(1, MAP_HEIGHT, width)
    row_obstacle = (obstacle.float().sum(dim=1) / col_count).view(1, MAP_HEIGHT, 1).expand(1, MAP_HEIGHT, width)

    if bool(known_cols.any()):
        grounded = (solid[MAP_HEIGHT - 1] | solid[MAP_HEIGHT - 2]) & known_cols
        ground_void = 1.0 - grounded.float().sum() / known_cols.float().sum().clamp_min(1.0)
        grounded_list = [bool(grounded[c].item()) for c in torch.nonzero(known_cols, as_tuple=False).flatten()]
        gap_runs: list[int] = []
        ground_runs: list[int] = []
        run_value = None
        run_len = 0
        for value in grounded_list + [not grounded_list[-1] if grounded_list else True]:
            if run_value is None:
                run_value = value
                run_len = 1
                continue
            if value == run_value:
                run_len += 1
                continue
            if run_value:
                ground_runs.append(run_len)
            else:
                gap_runs.append(run_len)
            run_value = value
            run_len = 1
        left_col = int(torch.nonzero(known_cols, as_tuple=False)[0].item())
        right_col = int(torch.nonzero(known_cols, as_tuple=False)[-1].item())
        left_boundary = solid[:, left_col].float().view(1, MAP_HEIGHT, 1).expand(1, MAP_HEIGHT, width)
        right_boundary = solid[:, right_col].float().view(1, MAP_HEIGHT, 1).expand(1, MAP_HEIGHT, width)
    else:
        ground_void = torch.tensor(0.0, device=idx_t.device)
        gap_runs = []
        ground_runs = []
        left_boundary = torch.zeros(1, MAP_HEIGHT, width, device=idx_t.device)
        right_boundary = torch.zeros(1, MAP_HEIGHT, width, device=idx_t.device)

    mid_known = known[4 : MAP_HEIGHT - 3].sum().clamp_min(1.0)
    mid_platform = platform[4 : MAP_HEIGHT - 3].float().sum() / mid_known
    obstacle_density = obstacle.float().sum() / known.sum().clamp_min(1.0)
    pipe_density = pipes.float().sum() / known.sum().clamp_min(1.0)
    enemy_density = enemies.float().sum() / known.sum().clamp_min(1.0)
    block_density = blocks.float().sum() / known.sum().clamp_min(1.0)
    sky_solid_density = solid[:6].float().sum() / known[:6].sum().clamp_min(1.0)
    no_pipe = (pipe_density <= 1e-6).float()
    no_enemy = (enemy_density <= 1e-6).float()
    mostly_empty_sky = (sky_solid_density < 0.02).float()
    mean_gap = torch.tensor(float(np.mean(gap_runs)) if gap_runs else 0.0, device=idx_t.device)
    longest_ground = torch.tensor(float(max(ground_runs or [0])), device=idx_t.device)
    gap_heavy = ((ground_void >= 0.35) | (mean_gap >= 5.0)).float()
    obstacle_heavy = ((gap_heavy <= 0.0) & ((obstacle_density >= 0.035) | (mid_platform >= 0.08))).float()
    plain = ((gap_heavy <= 0.0) & (obstacle_heavy <= 0.0)).float()

    scalar = lambda value: value.view(1, 1, 1).expand(1, MAP_HEIGHT, width)
    return [
        row_solid,
        row_platform,
        row_obstacle,
        scalar(ground_void),
        scalar(mid_platform),
        scalar(obstacle_density),
        scalar(pipe_density),
        scalar(enemy_density),
        scalar(block_density),
        scalar(no_pipe),
        scalar(no_enemy),
        scalar(mostly_empty_sky),
        scalar(gap_heavy),
        scalar(obstacle_heavy),
        scalar(plain),
        scalar(mean_gap / 8.0),
        scalar(longest_ground / max(width, 1)),
        left_boundary,
        right_boundary,
    ]


class TileFlowModel(FillModel):
    """Fill unknown cells with a trained categorical flow network."""

    def __init__(
        self,
        model: CategoricalFlowNet,
        config: TileFlowConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = torch.device(device)
        self.name = f"tileflow_{config.version}"
        self.sample_steps = 1
        rng_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        self.rng = torch.Generator(device=rng_device)
        self.rng.manual_seed(42)

    def _decode_logits(self, logits: torch.Tensor, mask: np.ndarray) -> np.ndarray:
        if not self.config.stochastic_decode:
            return logits.argmax(dim=1).squeeze(0).cpu().numpy()
        scaled = logits.squeeze(0).permute(1, 2, 0).contiguous().view(-1, len(VOCAB)) / 0.80
        values, indices = torch.topk(scaled, k=min(5, len(VOCAB)), dim=1)
        filtered = torch.full_like(scaled, -torch.inf)
        filtered.scatter_(1, indices, values)
        probs = torch.softmax(filtered.cpu(), dim=1)
        sampled = torch.multinomial(probs, num_samples=1, generator=self.rng).view(MAP_HEIGHT, mask.shape[1])
        return sampled.numpy()

    def _source_unknown_idx(self, idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        source = np.array(idx, copy=True)
        unknown = ~mask
        if self.config.dfm_source == "uniform":
            flat = torch.randint(
                low=0,
                high=len(VOCAB),
                size=(int(unknown.sum()),),
                generator=self.rng,
                device="cpu",
            ).numpy()
            source[unknown] = flat
        elif self.config.dfm_source == "train_prior" and self.config.dfm_source_probs is not None:
            probs = torch.as_tensor(self.config.dfm_source_probs, dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-8)
            flat = torch.multinomial(probs, num_samples=int(unknown.sum()), replacement=True, generator=self.rng).numpy()
            source[unknown] = flat
        elif self.config.dfm_source == "style_prior" and self.config.dfm_source_style_probs is not None:
            style_index = {"gap-heavy": 0, "obstacle-heavy": 1, "plain/low-obstacle": 2}
            style = style_descriptor(idx_to_level(idx), mask).style_class
            probs = torch.as_tensor(self.config.dfm_source_style_probs[style_index.get(style, 2)], dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-8)
            flat = torch.multinomial(probs, num_samples=int(unknown.sum()), replacement=True, generator=self.rng).numpy()
            source[unknown] = flat
        elif (
            self.config.dfm_source == "gap_style_prior"
            and self.config.dfm_source_probs is not None
            and self.config.dfm_source_style_probs is not None
        ):
            style = style_descriptor(idx_to_level(idx), mask).style_class
            if style == "gap-heavy":
                probs = torch.as_tensor(self.config.dfm_source_style_probs[0], dtype=torch.float32)
            else:
                probs = torch.as_tensor(self.config.dfm_source_probs, dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-8)
            flat = torch.multinomial(probs, num_samples=int(unknown.sum()), replacement=True, generator=self.rng).numpy()
            source[unknown] = flat
        else:
            source[unknown] = CHAR2IDX["-"]
        return source

    @torch.no_grad()
    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        rows = validate_fill_io(level, mask, width=mask.shape[1])
        if int(self.config.candidate_samples) > 1:
            candidates = [self._fill_once(rows, mask) for _ in range(int(self.config.candidate_samples))]
            return max(candidates, key=lambda candidate: self._candidate_score(rows, candidate, mask))
        return self._fill_once(rows, mask)

    def _candidate_score(self, original: list[str], candidate: list[str], mask: np.ndarray) -> float:
        known_desc = style_descriptor(original, mask)
        play = playability(candidate)
        struct = structural_violations(candidate)
        style = style_metrics(candidate, mask)
        score = (
            1.25 * float(play["completable"])
            + float(play["progress"])
            + 0.25 * float(play.get("gap_progress", play["progress"]))
            - 0.90 * float(struct["per_col"])
            - 0.18 * float(style["descriptor_distance"])
            - 0.18 * float(style["pipe_pair_error_rate"])
            - 0.12 * float(style["overlong_block_run_rate"])
            - 0.10 * float(style["bulky_mass_rate"])
        )
        if self.config.version == "v1.4":
            score += 0.10 * float(style["style_class_match_rate"])
            score -= 0.10 * float(style["overlong_block_run_rate"])
            score -= 0.08 * float(style["bulky_mass_rate"])
            score -= 0.06 * float(style["sky_mass_rate"])
            score -= 0.24 * float(style["pipe_pair_error_rate"])
            score -= 0.06 * float(style["complete_structure_viol_per_col"])
        if known_desc.style_class == "gap-heavy":
            score -= 0.35 * float(style["ground_void_ratio_gap"])
            score -= 0.12 * float(style["sparse_context_mismatch"])
        return float(score)

    def _pipe_pair_coupled_logits(self, logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        if self.config.version != "v1.4":
            return logits
        out = logits.clone()
        unknown = torch.as_tensor(~mask, dtype=torch.bool, device=logits.device)
        paired = unknown[:, :-1] & unknown[:, 1:]
        if not bool(paired.any()):
            return logits
        probs = torch.softmax(logits, dim=1)
        strength = 0.75
        left = paired.unsqueeze(0)
        right = paired.unsqueeze(0)
        out[:, CHAR2IDX[">"], :, 1:] += strength * (probs[:, CHAR2IDX["<"], :, :-1] - 0.20) * left
        out[:, CHAR2IDX["<"], :, :-1] += strength * (probs[:, CHAR2IDX[">"], :, 1:] - 0.20) * right
        out[:, CHAR2IDX["]"], :, 1:] += strength * (probs[:, CHAR2IDX["["], :, :-1] - 0.20) * left
        out[:, CHAR2IDX["["], :, :-1] += strength * (probs[:, CHAR2IDX["]"], :, 1:] - 0.20) * right
        edge_penalty = 0.55
        for tile in ("<", "["):
            out[:, CHAR2IDX[tile], :, -1] -= edge_penalty
        for tile in (">", "]"):
            out[:, CHAR2IDX[tile], :, 0] -= edge_penalty
        return out

    def _fill_once(self, rows: list[str], mask: np.ndarray) -> list[str]:
        idx = level_to_idx(rows, width=mask.shape[1])
        if self.config.time_conditioning:
            current = self._source_unknown_idx(idx, mask)
            base_steps = max(1, int(self.config.sample_steps))
            steps = base_steps
            if int(self.config.gap_heavy_sample_steps) > steps:
                known_desc = style_descriptor(rows, mask)
                if known_desc.style_class == "gap-heavy":
                    steps = int(self.config.gap_heavy_sample_steps)
            restore_rng_state = None
            for step in range(1, steps + 1):
                t = step / steps
                x = make_model_input(
                    current,
                    mask,
                    self.device,
                    self.config.position_channels,
                    self.config.context_channels,
                    reveal_unknown=True,
                    time_value=t,
                ).unsqueeze(0)
                if self.config.skeleton_conditioning and self.config.stochastic_decode:
                    logits = self.model.forward_sampled_skeleton(x, generator=self.rng)["tile"]
                else:
                    logits = self.model(x)
                logits = self._pipe_pair_coupled_logits(logits, mask)
                current[~mask] = self._decode_logits(logits, mask)[~mask]
                if step == base_steps and steps > base_steps:
                    restore_rng_state = self.rng.get_state()
            if restore_rng_state is not None:
                self.rng.set_state(restore_rng_state)
            pred = current
        else:
            x = make_model_input(idx, mask, self.device, self.config.position_channels, self.config.context_channels).unsqueeze(0)
            if self.config.skeleton_conditioning and self.config.stochastic_decode:
                logits = self.model.forward_sampled_skeleton(x, generator=self.rng)["tile"]
            else:
                logits = self.model(x)
            logits = self._pipe_pair_coupled_logits(logits, mask)
            pred = self._decode_logits(logits, mask)
        pred[mask] = idx[mask]
        return idx_to_level(pred)


class TileFlowEnsemble(FillModel):
    """Average logits from multiple TileFlow models before categorical decode."""

    def __init__(self, members: list[tuple[TileFlowModel, float]], name: str = "tileflow_ensemble") -> None:
        total = sum(weight for _, weight in members)
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self.members = [(model, weight / total) for model, weight in members]
        self.name = name
        self.sample_steps = 1
        self.model = self.members[0][0].model

    @torch.no_grad()
    def logits_for(self, level: list[str], mask: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        rows = validate_fill_io(level, mask, width=mask.shape[1])
        idx = level_to_idx(rows, width=mask.shape[1])
        logits_sum = None
        for model, weight in self.members:
            x = make_model_input(idx, mask, model.device, model.config.position_channels, model.config.context_channels).unsqueeze(0)
            logits = model.model(x).cpu() * weight
            logits_sum = logits if logits_sum is None else logits_sum + logits
        if logits_sum is None:
            raise ValueError("Ensemble has no members.")
        return idx, logits_sum

    @torch.no_grad()
    def aux_logits_for(self, level: list[str], mask: np.ndarray) -> dict[str, torch.Tensor]:
        rows = validate_fill_io(level, mask, width=mask.shape[1])
        idx = level_to_idx(rows, width=mask.shape[1])
        aux_sum: dict[str, torch.Tensor] = {}
        aux_weights: dict[str, float] = {}
        for model, weight in self.members:
            if not (
                model.config.structure_heads
                or model.config.utility_heads
                or model.config.terrain_heads
                or model.config.continuation_heads
            ):
                continue
            x = make_model_input(idx, mask, model.device, model.config.position_channels, model.config.context_channels).unsqueeze(0)
            outputs = model.model.forward_all(x)
            available = [key for key in ("support", "landing", "utility", "terrain", "continuation") if key in outputs]
            if not available:
                continue
            for key in available:
                logits = outputs[key].cpu() * weight
                aux_sum[key] = logits if key not in aux_sum else aux_sum[key] + logits
                aux_weights[key] = aux_weights.get(key, 0.0) + weight
        if not aux_weights:
            return {}
        return {key: value / aux_weights[key] for key, value in aux_sum.items()}

    @torch.no_grad()
    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        idx, logits_sum = self.logits_for(level, mask)
        pred = logits_sum.argmax(dim=1).squeeze(0).numpy()
        pred[mask] = idx[mask]
        return idx_to_level(pred)


class TileFlowStyleGuidedEnsemble(TileFlowEnsemble):
    """Inject center-style and difficulty constraints into logits before decode."""

    def __init__(
        self,
        members: list[tuple[TileFlowModel, float]],
        name: str = "tileflow_style_guided",
        strength: float = 1.0,
    ) -> None:
        super().__init__(members, name=name)
        self.strength = strength

    def _add_tile_bias(self, bias: torch.Tensor, tiles: tuple[str, ...], value: float, rows: range | None = None) -> None:
        if abs(value) < 1e-8:
            return
        row_slice = slice(None) if rows is None else slice(rows.start, rows.stop, rows.step)
        for tile in tiles:
            bias[0, CHAR2IDX[tile], row_slice, :] += value

    def _add_landing_scaffold(self, bias: torch.Tensor, mask: np.ndarray, value: float, spacing: int = 5) -> None:
        width = mask.shape[1]
        unknown_cols = [c for c in range(width) if not bool(mask[:, c].any())]
        if not unknown_cols:
            return
        regions: list[list[int]] = []
        current: list[int] = []
        previous = None
        for c in unknown_cols:
            if previous is None or c == previous + 1:
                current.append(c)
            else:
                regions.append(current)
                current = [c]
            previous = c
        if current:
            regions.append(current)

        for cols in regions:
            for i, c in enumerate(cols):
                near_edge = i in {0, 1, len(cols) - 2, len(cols) - 1}
                periodic = i % spacing in {0, 1}
                if near_edge or periodic:
                    bias[0, CHAR2IDX["X"], MAP_HEIGHT - 1, c] += value
                    bias[0, CHAR2IDX["X"], MAP_HEIGHT - 2, c] += value * 0.35
                    bias[0, CHAR2IDX["-"], MAP_HEIGHT - 1, c] -= value * 0.45

    def _add_platform_row_scaffold(
        self,
        bias: torch.Tensor,
        level: list[str],
        mask: np.ndarray,
        value: float,
    ) -> None:
        rows = normalize_level(level, width=mask.shape[1])
        known_cols = [c for c in range(mask.shape[1]) if bool(mask[:, c].any())]
        unknown_regions = []
        current = []
        for c in range(mask.shape[1]):
            if bool(mask[:, c].any()):
                if current:
                    unknown_regions.append(current)
                    current = []
                continue
            current.append(c)
        if current:
            unknown_regions.append(current)
        if not known_cols or not unknown_regions:
            return

        platform_tiles = set("XSQ?")
        candidates: list[tuple[int, float]] = []
        for r in range(4, MAP_HEIGHT - 2):
            density = sum(rows[r][c] in platform_tiles for c in known_cols) / len(known_cols)
            if density >= 0.08:
                candidates.append((r, density))
        if not candidates:
            return

        for row, density in sorted(candidates, key=lambda item: item[1], reverse=True)[:3]:
            row_value = value * min(1.0, 0.45 + 2.2 * density)
            spacing = 4 if density >= 0.16 else 5
            for cols in unknown_regions:
                for i, c in enumerate(cols):
                    near_edge = i in {0, 1, len(cols) - 2, len(cols) - 1}
                    stagger = (i + row) % spacing in {0, 1}
                    if near_edge or stagger:
                        bias[0, CHAR2IDX["X"], row, c] += row_value
                        bias[0, CHAR2IDX["-"], row, c] -= row_value * 0.20

    def guided_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        known_desc = style_descriptor(level, mask)
        bias = torch.zeros_like(logits)
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        s = self.strength

        obstacle_pressure = max(0.0, known_desc.obstacle_density - 0.02)
        structure_pressure = max(0.0, known_desc.structural_density - 0.20)
        gap_pressure = known_desc.ground_void_ratio - 0.28

        self._add_tile_bias(bias, OBSTACLE_TILES, s * min(1.8, 4.0 * obstacle_pressure + 0.18 * structure_pressure))
        self._add_tile_bias(bias, BLOCK_TILES, s * min(1.2, 5.5 * known_desc.block_density), range(5, 11))
        self._add_tile_bias(bias, PIPE_TILES, s * min(1.3, 7.0 * known_desc.pipe_density), range(7, MAP_HEIGHT))
        self._add_tile_bias(bias, CANNON_TILES, s * min(1.0, 7.0 * known_desc.cannon_density), range(7, MAP_HEIGHT))
        self._add_tile_bias(bias, ENEMY_TILES, s * min(1.0, 8.0 * known_desc.enemy_density + 0.25 * obstacle_pressure), range(8, 12))
        self._add_tile_bias(bias, COIN_TILES, s * min(0.9, 1.5 * obstacle_pressure), range(4, 9))

        if gap_pressure > 0:
            capped_gap = min(0.45, gap_pressure)
            self._add_tile_bias(bias, ("X",), -s * min(0.65, 0.9 * capped_gap), range(MAP_HEIGHT - 2, MAP_HEIGHT))
            self._add_tile_bias(bias, ("-",), s * min(0.45, 0.65 * capped_gap), range(MAP_HEIGHT - 2, MAP_HEIGHT))
            self._add_tile_bias(bias, ("X",), s * 0.10, range(MAP_HEIGHT - 1, MAP_HEIGHT))
        else:
            self._add_tile_bias(bias, ("X",), s * min(1.0, 2.0 * abs(gap_pressure)), range(MAP_HEIGHT - 2, MAP_HEIGHT))

        if known_desc.style_class == "plain/low-obstacle":
            self._add_tile_bias(bias, OBSTACLE_TILES, -0.35 * s)
        elif known_desc.style_class == "obstacle-heavy":
            self._add_tile_bias(bias, OBSTACLE_TILES, 0.35 * s)
        elif known_desc.style_class == "gap-heavy":
            self._add_tile_bias(bias, ("X",), 0.08 * s, range(MAP_HEIGHT - 1, MAP_HEIGHT))
            if not getattr(self, "structure_coupling", False):
                self._add_landing_scaffold(bias, mask, value=0.55 * s, spacing=5)
                self._add_platform_row_scaffold(bias, level, mask, value=0.70 * s)

        return logits + bias.masked_fill(~unknown, 0.0)

    @torch.no_grad()
    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        idx, logits = self.logits_for(level, mask)
        guided = self.guided_logits(level, mask, logits)
        pred = guided.argmax(dim=1).squeeze(0).numpy()
        pred[mask] = idx[mask]
        return idx_to_level(pred)


class TileFlowStochasticGuidedEnsemble(TileFlowStyleGuidedEnsemble):
    """Sample guided categorical maps with optional difficulty constraints."""

    def __init__(
        self,
        members: list[tuple[TileFlowModel, float]],
        name: str = "tileflow_stochastic_guided",
        strength: float = 0.7,
        temperature: float = 0.9,
        top_k: int = 5,
        difficulty: str = "neutral",
        seed: int = 42,
        structure_coupling: bool = False,
        coupling_strength: float = 2.8,
        learned_structure_bias: bool = False,
        learned_bias_strength: float = 1.0,
        utility_guidance: bool = False,
        utility_guidance_strength: float = 1.0,
        coherence_guidance: bool = False,
        coherence_guidance_strength: float = 1.0,
        context_guidance: bool = False,
        context_guidance_strength: float = 1.0,
        terrain_guidance: bool = False,
        terrain_guidance_strength: float = 1.0,
        continuation_guidance: bool = False,
        continuation_guidance_strength: float = 1.0,
    ) -> None:
        super().__init__(members, name=name, strength=strength)
        if difficulty not in {"easy", "neutral", "hard"}:
            raise ValueError("difficulty must be one of: easy, neutral, hard")
        self.temperature = temperature
        self.top_k = top_k
        self.difficulty = difficulty
        self.structure_coupling = structure_coupling
        self.coupling_strength = coupling_strength
        self.learned_structure_bias = learned_structure_bias
        self.learned_bias_strength = learned_bias_strength
        self.utility_guidance = utility_guidance
        self.utility_guidance_strength = utility_guidance_strength
        self.coherence_guidance = coherence_guidance
        self.coherence_guidance_strength = coherence_guidance_strength
        self.context_guidance = context_guidance
        self.context_guidance_strength = context_guidance_strength
        self.terrain_guidance = terrain_guidance
        self.terrain_guidance_strength = terrain_guidance_strength
        self.continuation_guidance = continuation_guidance
        self.continuation_guidance_strength = continuation_guidance_strength
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

    def difficulty_logits(self, logits: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros_like(logits)
        s = self.strength
        if self.difficulty == "easy":
            self._add_tile_bias(bias, ENEMY_TILES, -0.9 * s, range(7, 12))
            self._add_tile_bias(bias, CANNON_TILES, -0.7 * s, range(7, MAP_HEIGHT))
            self._add_tile_bias(bias, PIPE_TILES, -0.4 * s, range(7, MAP_HEIGHT))
            self._add_tile_bias(bias, ("X",), 0.55 * s, range(MAP_HEIGHT - 2, MAP_HEIGHT))
            self._add_tile_bias(bias, ("-",), -0.35 * s, range(MAP_HEIGHT - 2, MAP_HEIGHT))
        elif self.difficulty == "hard":
            self._add_tile_bias(bias, ENEMY_TILES, 0.9 * s, range(7, 12))
            self._add_tile_bias(bias, CANNON_TILES, 0.55 * s, range(7, MAP_HEIGHT))
            self._add_tile_bias(bias, PIPE_TILES, 0.35 * s, range(7, MAP_HEIGHT))
            self._add_tile_bias(bias, BLOCK_TILES, 0.25 * s, range(5, 11))
            self._add_tile_bias(bias, ("X",), -0.45 * s, range(MAP_HEIGHT - 2, MAP_HEIGHT))
            self._add_tile_bias(bias, ("-",), 0.30 * s, range(MAP_HEIGHT - 2, MAP_HEIGHT))
        return logits + bias

    def _sample_indices(self, logits: torch.Tensor, mask: np.ndarray) -> np.ndarray:
        logits = logits.squeeze(0).permute(1, 2, 0).contiguous()
        flat_logits = logits.view(-1, len(VOCAB)) / max(self.temperature, 1e-6)
        if self.top_k > 0 and self.top_k < len(VOCAB):
            values, indices = torch.topk(flat_logits, k=self.top_k, dim=1)
            filtered = torch.full_like(flat_logits, -torch.inf)
            filtered.scatter_(1, indices, values)
            flat_logits = filtered
        probs = torch.softmax(flat_logits.cpu(), dim=1)
        sampled = torch.multinomial(probs, num_samples=1, generator=self.rng).view(MAP_HEIGHT, mask.shape[1])
        return sampled.numpy()

    def _center_platform_rows(self, level: list[str], mask: np.ndarray) -> list[int]:
        rows = normalize_level(level, width=mask.shape[1])
        known_cols = [c for c in range(mask.shape[1]) if bool(mask[:, c].any())]
        if not known_cols:
            return []
        platform_tiles = set("XSQ?")
        scored: list[tuple[int, float]] = []
        for r in range(3, MAP_HEIGHT):
            density = sum(rows[r][c] in platform_tiles for c in known_cols) / len(known_cols)
            if density >= 0.08:
                scored.append((r, density))
        return [row for row, _ in sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)[:3]]

    def _add_run_logits(
        self,
        logits: torch.Tensor,
        row: int,
        cols: list[int],
        run_start: int,
        run_length: int,
        value: float,
    ) -> None:
        for c in cols[run_start : min(len(cols), run_start + run_length)]:
            logits[0, CHAR2IDX["X"], row, c] += value
            if 4 <= row <= 10:
                logits[0, CHAR2IDX["S"], row, c] += value * 0.35
                logits[0, CHAR2IDX["Q"], row, c] += value * 0.15
                logits[0, CHAR2IDX["?"], row, c] += value * 0.15
            logits[0, CHAR2IDX["-"], row, c] -= value * 0.60

    def _structure_coupled_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.structure_coupling:
            return logits
        known_desc = style_descriptor(level, mask)
        if known_desc.style_class != "gap-heavy":
            return logits

        out = logits.clone()
        platform_rows = self._center_platform_rows(level, mask)
        if not platform_rows:
            platform_rows = [MAP_HEIGHT - 1]
        lower_rows = sorted(platform_rows, reverse=True)
        primary = lower_rows[0]
        secondary = next((row for row in lower_rows[1:] if abs(row - primary) <= 4), primary)
        value = self.coupling_strength * max(0.5, self.strength)
        for cols in self._unknown_regions(mask):
            if len(cols) < 3:
                continue
            step = 5
            starts = [0]
            starts.extend(range(4, max(5, len(cols) - 2), step))
            if len(cols) >= 4:
                starts.append(max(0, len(cols) - 3))
            used: set[tuple[int, int]] = set()
            for i, start in enumerate(starts):
                run_length = 3 if i % 3 == 0 else 2
                key = (primary, start)
                if key not in used:
                    used.add(key)
                    self._add_run_logits(out, primary, cols, start, run_length, value)
                    if primary - 1 >= 0:
                        for c in cols[start : min(len(cols), start + run_length)]:
                            out[0, CHAR2IDX["-"], primary - 1, c] += value * 0.12
                if secondary != primary and i % 3 == 1:
                    secondary_start = min(max(0, start + 1), max(0, len(cols) - 2))
                    secondary_key = (secondary, secondary_start)
                    if secondary_key in used:
                        continue
                    used.add(secondary_key)
                    self._add_run_logits(out, secondary, cols, secondary_start, 2, value * 0.75)
                    if secondary - 1 >= 0:
                        for c in cols[secondary_start : min(len(cols), secondary_start + 2)]:
                            out[0, CHAR2IDX["-"], secondary - 1, c] += value * 0.08
        return out

    def _learned_structure_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.learned_structure_bias:
            return logits
        aux = self.aux_logits_for(level, mask)
        if not aux:
            return logits

        out = logits.clone()
        s = self.learned_bias_strength * max(0.5, self.strength)
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

        support_probs = torch.softmax(aux["support"], dim=1)
        support_score = support_probs[:, 1:2] + 1.25 * support_probs[:, 2:3] + support_probs[:, 3:4]
        run_score = support_probs[:, 2:3]
        object_support_score = support_probs[:, 3:4]
        air_score = support_probs[:, 0:1]
        for tile in ("X", "S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 1.10 * support_score
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] -= s * 0.65 * support_score
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] += s * 0.55 * run_score
        for tile in ("E", "<", ">", "B"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.35 * object_support_score
        for tile in ("X", "S", "Q", "?", "<", ">", "[", "]", "B", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.25 * air_score

        landing_probs = torch.softmax(aux["landing"].mean(dim=2), dim=1).unsqueeze(2)
        gap_score = landing_probs[:, 0:1]
        landing_score = landing_probs[:, 1:2]
        grounded_score = landing_probs[:, 2:3]
        lower = torch.zeros_like(out)
        lower[:, :, MAP_HEIGHT - 2 : MAP_HEIGHT, :] = 1.0
        bottom = torch.zeros_like(out)
        bottom[:, :, MAP_HEIGHT - 1 : MAP_HEIGHT, :] = 1.0
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] += lower[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] * (
            s * (1.10 * landing_score + 0.45 * grounded_score - 0.55 * gap_score)
        )
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += bottom[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] * (
            s * (0.35 * gap_score - 0.45 * landing_score)
        )
        return torch.where(unknown, out, logits)

    def _utility_guided_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.utility_guidance:
            return logits
        aux = self.aux_logits_for(level, mask)
        if "utility" not in aux:
            return logits

        out = logits.clone()
        s = self.utility_guidance_strength * max(0.5, self.strength)
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

        probs = torch.softmax(aux["utility"], dim=1)
        empty_score = probs[:, 0:1]
        route_surface_score = probs[:, 1:2]
        platform_body_score = probs[:, 2:3]
        platform_edge_score = probs[:, 3:4]
        supported_entity_score = probs[:, 4:5]
        gap_void_score = probs[:, 5:6]

        surface_score = route_surface_score + 0.85 * platform_body_score + 1.15 * platform_edge_score
        mid_platform_score = platform_body_score + 1.25 * platform_edge_score

        lower_band = torch.zeros_like(out)
        lower_band[:, :, MAP_HEIGHT - 3 : MAP_HEIGHT, :] = 1.0
        mid_band = torch.zeros_like(out)
        mid_band[:, :, 4 : MAP_HEIGHT - 3, :] = 1.0

        for tile in ("X", "S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 1.00 * surface_score
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] += s * 0.45 * lower_band[:, :1] * route_surface_score
        for tile in ("S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.35 * mid_band[:, :1] * mid_platform_score
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] -= s * 0.70 * surface_score

        for tile in ("E", "<", ">", "B"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.42 * supported_entity_score
        for tile in ("[", "]", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.22 * supported_entity_score

        bottom_band = torch.zeros_like(out)
        bottom_band[:, :, MAP_HEIGHT - 2 : MAP_HEIGHT, :] = 1.0
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += s * 0.45 * bottom_band[:, :1] * gap_void_score
        for tile in ("X", "S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.42 * bottom_band[:, :1] * gap_void_score

        for tile in ("<", ">", "[", "]", "B", "b", "E"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.20 * empty_score
        return torch.where(unknown, out, logits)

    def _coherence_guided_logits(self, mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.coherence_guidance:
            return logits

        out = logits.clone()
        s = self.coherence_guidance_strength * max(0.5, self.strength)
        probs = torch.softmax(logits, dim=1)
        solid_ids = [CHAR2IDX[ch] for ch in ("X", "S", "Q", "?", "<", ">", "[", "]", "B", "b")]
        solid_prob = probs[:, solid_ids].sum(dim=1, keepdim=True)
        left = torch.nn.functional.pad(solid_prob[:, :, :, :-1], (1, 0, 0, 0))
        right = torch.nn.functional.pad(solid_prob[:, :, :, 1:], (0, 1, 0, 0))
        below = torch.nn.functional.pad(solid_prob[:, :, 1:, :], (0, 0, 0, 1))
        support_score = torch.maximum(torch.maximum(left, right), below)
        isolate_risk = (1.0 - support_score).clamp(0.0, 1.0)

        mid_sky = torch.zeros_like(isolate_risk)
        mid_sky[:, :, 2 : MAP_HEIGHT - 2, :] = 1.0
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        risk = isolate_risk * mid_sky

        for tile in ("S", "Q", "?", "o", "E", "<", ">", "[", "]", "B", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.55 * risk
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] -= s * 0.25 * risk
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += s * 0.18 * risk
        return torch.where(unknown, out, logits)

    def _terrain_guided_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.terrain_guidance:
            return logits
        aux = self.aux_logits_for(level, mask)
        if "terrain" not in aux:
            return logits

        out = logits.clone()
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        s = self.terrain_guidance_strength * max(0.5, self.strength)
        probs = torch.softmax(aux["terrain"], dim=1)

        stable_score = probs[:, TERRAIN_STABLE_GROUND : TERRAIN_STABLE_GROUND + 1]
        gap_score = probs[:, TERRAIN_GAP : TERRAIN_GAP + 1]
        landing_score = probs[:, TERRAIN_LANDING : TERRAIN_LANDING + 1]
        tooth_score = probs[:, TERRAIN_BAD_TOOTH : TERRAIN_BAD_TOOTH + 1]
        surface_score = stable_score + 0.90 * landing_score

        lower_band = torch.zeros((1, 1, MAP_HEIGHT, mask.shape[1]), dtype=logits.dtype, device=logits.device)
        lower_band[:, :, MAP_HEIGHT - 4 : MAP_HEIGHT, :] = 1.0
        bottom_band = torch.zeros_like(lower_band)
        bottom_band[:, :, MAP_HEIGHT - 2 : MAP_HEIGHT, :] = 1.0

        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] += s * 0.85 * lower_band * surface_score
        for tile in ("S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.18 * lower_band * landing_score
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] -= s * 0.45 * lower_band * surface_score

        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += s * 0.55 * bottom_band * gap_score
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] -= s * 0.45 * bottom_band * gap_score

        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] -= s * 0.75 * bottom_band * tooth_score
        for tile in ("S", "Q", "?", "<", ">", "[", "]", "B", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.25 * tooth_score
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += s * 0.22 * bottom_band * tooth_score

        tile_probs = torch.softmax(logits, dim=1)
        for row in (MAP_HEIGHT - 1, MAP_HEIGHT - 2):
            ground_p = tile_probs[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1, row : row + 1, :]
            left = torch.nn.functional.pad(ground_p[:, :, :, :-1], (1, 0, 0, 0))
            right = torch.nn.functional.pad(ground_p[:, :, :, 1:], (0, 1, 0, 0))
            one_col_hole = left * right * (1.0 - ground_p)
            one_col_island = ground_p * (1.0 - left) * (1.0 - right)
            out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1, row : row + 1, :] += s * (
                0.65 * one_col_hole - 0.70 * one_col_island
            )
            out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1, row : row + 1, :] += s * (
                0.32 * one_col_island - 0.42 * one_col_hole
            )
        return torch.where(unknown, out, logits)

    def _continuation_guided_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.continuation_guidance:
            return logits
        aux = self.aux_logits_for(level, mask)
        if "continuation" not in aux:
            return logits

        out = logits.clone()
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        s = self.continuation_guidance_strength * max(0.5, self.strength)
        probs = torch.softmax(aux["continuation"], dim=1)
        context_run = probs[:, CONTINUATION_CONTEXT_RUN : CONTINUATION_CONTEXT_RUN + 1]
        landing_span = probs[:, CONTINUATION_LANDING_SPAN : CONTINUATION_LANDING_SPAN + 1]
        noise = probs[:, CONTINUATION_NOISE : CONTINUATION_NOISE + 1]

        mid_band = torch.zeros((1, 1, MAP_HEIGHT, mask.shape[1]), dtype=logits.dtype, device=logits.device)
        mid_band[:, :, 3 : MAP_HEIGHT - 2, :] = 1.0
        lower_band = torch.zeros_like(mid_band)
        lower_band[:, :, MAP_HEIGHT - 4 : MAP_HEIGHT, :] = 1.0

        run_score = context_run + 0.85 * landing_span
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] += s * (0.70 * mid_band * context_run + 0.90 * lower_band * landing_span)
        for tile in ("S", "Q", "?"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] += s * 0.35 * mid_band * context_run
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] -= s * 0.45 * run_score

        for tile in ("S", "Q", "?", "o", "E", "<", ">", "[", "]", "B", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= s * 0.45 * noise
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] -= s * 0.22 * noise
        out[:, CHAR2IDX["-"] : CHAR2IDX["-"] + 1] += s * 0.18 * noise
        return torch.where(unknown, out, logits)

    def _context_guided_logits(self, level: list[str], mask: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
        if not self.context_guidance:
            return logits
        known_desc = style_descriptor(level, mask)
        rows = normalize_level(level, width=mask.shape[1])
        known_cols = [c for c in range(mask.shape[1]) if bool(mask[:, c].any())]
        if not known_cols:
            return logits

        platform_tiles = set("XSQ?")
        scored: list[tuple[int, float]] = []
        for r in range(3, MAP_HEIGHT - 2):
            density = sum(rows[r][c] in platform_tiles for c in known_cols) / len(known_cols)
            if density >= 0.08:
                scored.append((r, density))
        if not scored:
            return logits
        if known_desc.style_class != "gap-heavy":
            return logits

        out = logits.clone()
        unknown = torch.as_tensor(~mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        value = self.context_guidance_strength * max(0.5, self.strength)
        add_scale = 1.25
        cleanup_scale = 0.10
        allowed = torch.zeros((1, 1, MAP_HEIGHT, mask.shape[1]), dtype=logits.dtype, device=logits.device)
        for row_rank, (base_row, density) in enumerate(sorted(scored, key=lambda item: item[1], reverse=True)[:3]):
            row_value = add_scale * value * min(1.0, 0.55 + 2.0 * density)
            for cols in self._unknown_regions(mask):
                if len(cols) < 5:
                    continue
                island_count = 1 if len(cols) < 14 else 2
                if density >= 0.18 and len(cols) >= 24:
                    island_count += 1
                for island in range(island_count):
                    row_offset = 0 if row_rank == 0 else (-1 if island % 2 == 0 else 1)
                    row = max(3, min(MAP_HEIGHT - 3, base_row + row_offset))
                    run = min(len(cols), 3 + int(density >= 0.14) + int(len(cols) >= 24 and island == 1))
                    anchor = (island + 1) / (island_count + 1)
                    center_i = int(round(anchor * (len(cols) - 1)))
                    start = max(0, min(len(cols) - run, center_i - run // 2))
                    for c in cols[start : min(len(cols), start + run)]:
                        allowed[0, 0, row, c] = 1.0
                        out[0, CHAR2IDX["X"], row, c] += row_value * 1.05
                        out[0, CHAR2IDX["-"], row, c] -= row_value * 0.55
                        if row - 1 >= 0:
                            out[0, CHAR2IDX["-"], row - 1, c] += row_value * 0.15

        edge_value = value * (0.90 if known_desc.style_class == "gap-heavy" else 0.40)
        left_edge = known_cols[0]
        right_edge = known_cols[-1]
        left_near = known_cols[: min(4, len(known_cols))]
        right_near = known_cols[max(0, len(known_cols) - 4) :]

        def add_boundary_run(edge_col: int, near_cols: list[int], direction: int) -> None:
            if not near_cols:
                return
            for base_row, density in sorted(scored, key=lambda item: item[1], reverse=True)[:3]:
                near_density = sum(rows[base_row][c] in platform_tiles for c in near_cols) / len(near_cols)
                if near_density < 0.25:
                    continue
                cols: list[int] = []
                for step in range(1, 5):
                    c = edge_col + direction * step
                    if not (0 <= c < mask.shape[1]) or bool(mask[:, c].any()):
                        break
                    cols.append(c)
                if not cols:
                    continue
                run_len = min(len(cols), 2 + int(near_density >= 0.50))
                row = max(3, min(MAP_HEIGHT - 3, base_row))
                row_value = edge_value * min(1.0, 0.45 + density + near_density)
                for c in cols[:run_len]:
                    allowed[0, 0, row, c] = 1.0
                    out[0, CHAR2IDX["X"], row, c] += row_value * 1.00
                    out[0, CHAR2IDX["-"], row, c] -= row_value * 0.50
                    if row - 1 >= 0:
                        out[0, CHAR2IDX["-"], row - 1, c] += row_value * 0.10

        add_boundary_run(left_edge, left_near, -1)
        add_boundary_run(right_edge, right_near, 1)

        mid_unknown = torch.zeros_like(allowed)
        mid_unknown[:, :, 3 : MAP_HEIGHT - 2, :] = 1.0
        outside_allowed = (1.0 - allowed) * mid_unknown
        for tile in ("S", "Q", "?", "o", "E", "<", ">", "[", "]", "B", "b"):
            out[:, CHAR2IDX[tile] : CHAR2IDX[tile] + 1] -= cleanup_scale * value * outside_allowed
        out[:, CHAR2IDX["X"] : CHAR2IDX["X"] + 1] -= cleanup_scale * value * 0.55 * outside_allowed
        return torch.where(unknown, out, logits)

    def _unknown_regions(self, mask: np.ndarray) -> list[list[int]]:
        width = mask.shape[1]
        regions: list[list[int]] = []
        current: list[int] = []
        for c in range(width):
            if bool(mask[:, c].any()):
                if current:
                    regions.append(current)
                    current = []
                continue
            current.append(c)
        if current:
            regions.append(current)
        return regions

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.rng).item())

    def _choice(self, values: tuple[str, ...]) -> str:
        index = int(torch.randint(len(values), (1,), generator=self.rng).item())
        return values[index]

    def _surface_row(self, idx: np.ndarray, c: int) -> int | None:
        solid = {CHAR2IDX[ch] for ch in ("X", "S", "Q", "?", "<", ">", "[", "]", "B", "b")}
        for r in range(MAP_HEIGHT - 1, 0, -1):
            if int(idx[r, c]) in solid and int(idx[r - 1, c]) == CHAR2IDX["-"]:
                return r - 1
        return None

    def _break_long_gaps(self, idx: np.ndarray, mask: np.ndarray, known_desc) -> None:
        """Keep stochastic gaps, but prevent neutral samples from becoming endless voids."""
        max_gap = 6 if known_desc.style_class == "gap-heavy" else 4
        for cols in self._unknown_regions(mask):
            run: list[int] = []
            for c in cols:
                grounded = idx[MAP_HEIGHT - 1, c] == CHAR2IDX["X"] or idx[MAP_HEIGHT - 2, c] == CHAR2IDX["X"]
                if grounded:
                    run = []
                    continue
                run.append(c)
                if len(run) > max_gap:
                    landing = run[-2]
                    idx[MAP_HEIGHT - 1, landing] = CHAR2IDX["X"]
                    if self._rand() < 0.35:
                        idx[MAP_HEIGHT - 2, landing] = CHAR2IDX["X"]
                    run = []

    def _clear_unnatural_noise(self, idx: np.ndarray, mask: np.ndarray) -> None:
        """Remove common independent-sampling artifacts inside generated cells."""
        support = {CHAR2IDX[ch] for ch in ("X", "S", "Q", "?", "<", ">", "[", "]", "B", "b")}
        pipe_or_cannon = {CHAR2IDX[ch] for ch in ("<", ">", "[", "]", "B", "b")}
        block = {CHAR2IDX[ch] for ch in ("S", "Q", "?")}
        for r in range(MAP_HEIGHT):
            for c in range(mask.shape[1]):
                if mask[r, c]:
                    continue
                tile = int(idx[r, c])
                below = int(idx[r + 1, c]) if r + 1 < MAP_HEIGHT else CHAR2IDX["X"]
                if tile == CHAR2IDX["E"] and below not in support:
                    idx[r, c] = CHAR2IDX["-"]
                if tile in pipe_or_cannon and r < MAP_HEIGHT - 2 and below == CHAR2IDX["-"]:
                    idx[r, c] = CHAR2IDX["-"]
                if tile in block and r not in range(4, 11) and below == CHAR2IDX["-"]:
                    idx[r, c] = CHAR2IDX["-"]

    def _place_pipe(self, idx: np.ndarray, mask: np.ndarray, c: int, height: int) -> None:
        if c + 1 >= mask.shape[1]:
            return
        if bool(mask[:, c].any()) or bool(mask[:, c + 1].any()):
            return
        height = max(2, min(height, 5))
        top_r = MAP_HEIGHT - height - 1
        if top_r < 5:
            return
        for r in range(top_r):
            idx[r, c] = CHAR2IDX["-"]
            idx[r, c + 1] = CHAR2IDX["-"]
        idx[top_r, c] = CHAR2IDX["<"]
        idx[top_r, c + 1] = CHAR2IDX[">"]
        for r in range(top_r + 1, MAP_HEIGHT):
            idx[r, c] = CHAR2IDX["["]
            idx[r, c + 1] = CHAR2IDX["]"]

    def _place_block_run(self, idx: np.ndarray, mask: np.ndarray, cols: list[int], row: int) -> None:
        if len(cols) < 3:
            return
        start_i = int(torch.randint(max(1, len(cols) - 3), (1,), generator=self.rng).item())
        run = int(torch.randint(2, 5, (1,), generator=self.rng).item())
        for c in cols[start_i : min(len(cols), start_i + run)]:
            if mask[row, c]:
                continue
            idx[row, c] = CHAR2IDX[self._choice(("S", "S", "Q", "?"))]
            if row - 1 >= 0 and not mask[row - 1, c] and self._rand() < 0.28:
                idx[row - 1, c] = CHAR2IDX["o"]

    def _place_enemy(self, idx: np.ndarray, mask: np.ndarray, cols: list[int]) -> None:
        candidates = []
        for c in cols:
            surface = self._surface_row(idx, c)
            if surface is not None and 1 <= surface < MAP_HEIGHT - 1 and not mask[surface, c]:
                candidates.append((surface, c))
        if not candidates:
            return
        surface, c = candidates[int(torch.randint(len(candidates), (1,), generator=self.rng).item())]
        idx[surface, c] = CHAR2IDX["E"]

    def _add_mario_motifs(self, idx: np.ndarray, mask: np.ndarray, known_desc) -> None:
        """Inject sparse Mario-like motifs as part of stochastic decoding."""
        for cols in self._unknown_regions(mask):
            if len(cols) < 6:
                continue
            pipe_chance = 0.025 + min(0.08, known_desc.pipe_density * 4.0)
            enemy_chance = 0.055 + min(0.08, known_desc.enemy_density * 4.0)
            block_chance = 0.055 + min(0.12, known_desc.block_density * 4.0)
            if known_desc.style_class == "plain/low-obstacle":
                pipe_chance *= 0.7
                enemy_chance *= 0.7
                block_chance *= 0.75
            if known_desc.style_class == "obstacle-heavy":
                pipe_chance *= 1.2
                enemy_chance *= 1.2
                block_chance *= 1.25

            if self.difficulty == "easy":
                enemy_chance *= 0.35
                pipe_chance *= 0.75
            elif self.difficulty == "hard":
                enemy_chance *= 1.6
                pipe_chance *= 1.25
                block_chance *= 1.2

            if self._rand() < pipe_chance and len(cols) >= 8:
                c = cols[int(torch.randint(max(1, len(cols) - 2), (1,), generator=self.rng).item())]
                self._place_pipe(idx, mask, c, height=int(torch.randint(2, 5, (1,), generator=self.rng).item()))
            if self._rand() < block_chance:
                self._place_block_run(idx, mask, cols, row=int(torch.randint(5, 10, (1,), generator=self.rng).item()))
            if self._rand() < enemy_chance:
                self._place_enemy(idx, mask, cols)

    def _naturalize_sample(self, idx: np.ndarray, mask: np.ndarray, level: list[str]) -> np.ndarray:
        out = idx.copy()
        out[mask] = idx[mask]
        return out

    @torch.no_grad()
    def fill(self, level: list[str], mask: np.ndarray) -> list[str]:
        idx, logits = self.logits_for(level, mask)
        guided = self.difficulty_logits(self.guided_logits(level, mask, logits))
        guided = self._learned_structure_logits(level, mask, guided)
        guided = self._utility_guided_logits(level, mask, guided)
        guided = self._terrain_guided_logits(level, mask, guided)
        guided = self._continuation_guided_logits(level, mask, guided)
        guided = self._context_guided_logits(level, mask, guided)
        guided = self._coherence_guided_logits(mask, guided)
        guided = self._structure_coupled_logits(level, mask, guided)
        pred = self._sample_indices(guided, mask)
        pred = self._naturalize_sample(pred, mask, level)
        pred[mask] = idx[mask]
        return idx_to_level(pred)


def load_tileflow_model(path: str | Path, device: torch.device | str = "cpu") -> TileFlowModel:
    payload: dict[str, Any] = torch.load(Path(path), map_location=device)
    config = TileFlowConfig(**payload["config"])
    model = CategoricalFlowNet(config)
    model.load_state_dict(payload["model"])
    return TileFlowModel(model=model, config=config, device=device)


def load_tileflow_ensemble(
    paths: list[str | Path],
    weights: list[float],
    device: torch.device | str = "cpu",
    name: str = "tileflow_ensemble",
) -> TileFlowEnsemble:
    if len(paths) != len(weights):
        raise ValueError("paths and weights must have the same length.")
    return TileFlowEnsemble(
        [(load_tileflow_model(path, device=device), weight) for path, weight in zip(paths, weights)],
        name=name,
    )


def load_tileflow_style_guided_ensemble(
    paths: list[str | Path],
    weights: list[float],
    device: torch.device | str = "cpu",
    name: str = "tileflow_style_guided",
    strength: float = 1.0,
) -> TileFlowStyleGuidedEnsemble:
    if len(paths) != len(weights):
        raise ValueError("paths and weights must have the same length.")
    return TileFlowStyleGuidedEnsemble(
        [(load_tileflow_model(path, device=device), weight) for path, weight in zip(paths, weights)],
        name=name,
        strength=strength,
    )


def load_tileflow_stochastic_guided_ensemble(
    paths: list[str | Path],
    weights: list[float],
    device: torch.device | str = "cpu",
    name: str = "tileflow_stochastic_guided",
    strength: float = 0.7,
    temperature: float = 0.9,
    top_k: int = 5,
    difficulty: str = "neutral",
    seed: int = 42,
    structure_coupling: bool = False,
    coupling_strength: float = 2.8,
    learned_structure_bias: bool = False,
    learned_bias_strength: float = 1.0,
    utility_guidance: bool = False,
    utility_guidance_strength: float = 1.0,
    coherence_guidance: bool = False,
    coherence_guidance_strength: float = 1.0,
    context_guidance: bool = False,
    context_guidance_strength: float = 1.0,
    terrain_guidance: bool = False,
    terrain_guidance_strength: float = 1.0,
    continuation_guidance: bool = False,
    continuation_guidance_strength: float = 1.0,
) -> TileFlowStochasticGuidedEnsemble:
    if len(paths) != len(weights):
        raise ValueError("paths and weights must have the same length.")
    return TileFlowStochasticGuidedEnsemble(
        [(load_tileflow_model(path, device=device), weight) for path, weight in zip(paths, weights)],
        name=name,
        strength=strength,
        temperature=temperature,
        top_k=top_k,
        difficulty=difficulty,
        seed=seed,
        structure_coupling=structure_coupling,
        coupling_strength=coupling_strength,
        learned_structure_bias=learned_structure_bias,
        learned_bias_strength=learned_bias_strength,
        utility_guidance=utility_guidance,
        utility_guidance_strength=utility_guidance_strength,
        coherence_guidance=coherence_guidance,
        coherence_guidance_strength=coherence_guidance_strength,
        context_guidance=context_guidance,
        context_guidance_strength=context_guidance_strength,
        terrain_guidance=terrain_guidance,
        terrain_guidance_strength=terrain_guidance_strength,
        continuation_guidance=continuation_guidance,
        continuation_guidance_strength=continuation_guidance_strength,
    )


def checkpoint_payload(model: CategoricalFlowNet, config: TileFlowConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "model": model.state_dict(),
        "metrics": metrics,
        "vocab": list(VOCAB),
    }
