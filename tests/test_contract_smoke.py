from pathlib import Path

import numpy as np

from tileflow.benchmarks.random_fill import RandomFillBenchmark
from tileflow.common.data import CHAR2IDX, DEFAULT_W, MAP_HEIGHT, make_level_splits_with_eval_files
from tileflow.common.eval import evaluate
from tileflow.common.fill_api import assert_known_preserved
from tileflow.common.masks import center_expand, make_mask_set
from tileflow.models.categorical_flow import CategoricalFlowNet, TileFlowConfig, TileFlowModel, level_to_idx, make_model_input


def synthetic_level(width: int = DEFAULT_W) -> list[str]:
    rows = ["-" * width for _ in range(MAP_HEIGHT)]
    rows[-2] = "X" * width
    rows[-1] = "X" * width
    return rows


def test_masks_are_bool_known_true():
    names = []
    for name, mask in make_mask_set(42, W=DEFAULT_W):
        names.append(name)
        assert mask.shape == (MAP_HEIGHT, DEFAULT_W)
        assert mask.dtype == np.bool_
        assert mask.any()
        assert (~mask).any()
    assert "center_expand" in names


def test_random_fill_preserves_known_cells():
    level = synthetic_level(DEFAULT_W)
    model = RandomFillBenchmark(seed=42)
    for _, mask in make_mask_set(42, W=DEFAULT_W):
        filled = model.fill(level, mask)
        assert_known_preserved(level, filled, mask)
        assert len(filled) == MAP_HEIGHT
        assert all(len(row) == DEFAULT_W for row in filled)


def test_eval_smoke_reports_plan_metrics():
    level = synthetic_level(DEFAULT_W)
    report = evaluate(RandomFillBenchmark(seed=42), make_mask_set(42, W=DEFAULT_W), [level], N=1)
    center = report["inpaint_center"]
    for key in [
        "completable_rate",
        "progress_mean",
        "tpk_kl_2x2",
        "diversity",
        "seam_kl_left",
        "seam_kl_right",
        "struct_viol_per_col",
        "context_tile_acc",
        "context_valid_rate",
    ]:
        assert key in center


def test_top_level_training_data_matches_height_contract():
    data_dir = Path("data")
    files = sorted(data_dir.glob("*.txt"))
    assert files

    for path in files:
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows) == MAP_HEIGHT, path.name
        assert all(set(row) <= set(CHAR2IDX) for row in rows), path.name

    splits = make_level_splits_with_eval_files(
        data_dir,
        ["mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt"],
    )
    assert not any(name.startswith("lost-levels-") for name in splits["eval"])


def test_center_expand_preserves_center_context():
    mask = center_expand(DEFAULT_W)
    start = (DEFAULT_W - 24) // 2
    end = start + 24
    assert mask.shape == (MAP_HEIGHT, DEFAULT_W)
    assert mask[:, :start].sum() == 0
    assert mask[:, start:end].all()
    assert mask[:, end:].sum() == 0


def test_v1_model_smoke_preserves_contract():
    level = synthetic_level(DEFAULT_W)
    mask = center_expand(DEFAULT_W)
    idx = level_to_idx(level)
    config = TileFlowConfig(
        version="v1.0",
        position_channels=True,
        context_channels=True,
        skeleton_heads=True,
        skeleton_conditioning=True,
        stochastic_decode=False,
    )
    model = CategoricalFlowNet(config)
    x = make_model_input(idx, mask, position_channels=True, context_channels=True).unsqueeze(0)
    outputs = model.forward_all(x)
    assert outputs["tile"].shape == (1, len(CHAR2IDX), MAP_HEIGHT, DEFAULT_W)
    assert outputs["skeleton_state"].shape[-1] == DEFAULT_W

    filled = TileFlowModel(model, config).fill(level, mask)
    assert_known_preserved(level, filled, mask)


def test_v1_1_support_conditioning_smoke_preserves_contract():
    level = synthetic_level(DEFAULT_W)
    mask = center_expand(DEFAULT_W)
    idx = level_to_idx(level)
    config = TileFlowConfig(
        version="v1.1",
        position_channels=True,
        context_channels=True,
        structure_heads=True,
        skeleton_heads=True,
        skeleton_conditioning=True,
        support_conditioning=True,
        stochastic_decode=False,
    )
    model = CategoricalFlowNet(config)
    x = make_model_input(idx, mask, position_channels=True, context_channels=True).unsqueeze(0)
    outputs = model.forward_all(x)
    assert outputs["tile"].shape == (1, len(CHAR2IDX), MAP_HEIGHT, DEFAULT_W)
    assert outputs["support"].shape == (1, 4, MAP_HEIGHT, DEFAULT_W)
    assert outputs["skeleton_state"].shape[-1] == DEFAULT_W

    filled = TileFlowModel(model, config).fill(level, mask)
    assert_known_preserved(level, filled, mask)


def test_v1_2_time_conditioned_smoke_preserves_contract():
    level = synthetic_level(DEFAULT_W)
    mask = center_expand(DEFAULT_W)
    idx = level_to_idx(level)
    config = TileFlowConfig(
        version="v1.2",
        position_channels=True,
        context_channels=True,
        structure_heads=True,
        skeleton_heads=True,
        skeleton_conditioning=True,
        support_conditioning=True,
        stochastic_decode=False,
        time_conditioning=True,
        sample_steps=2,
    )
    model = CategoricalFlowNet(config)
    x = make_model_input(
        idx,
        mask,
        position_channels=True,
        context_channels=True,
        reveal_unknown=True,
        time_value=0.5,
    ).unsqueeze(0)
    outputs = model.forward_all(x)
    assert outputs["tile"].shape == (1, len(CHAR2IDX), MAP_HEIGHT, DEFAULT_W)
    assert outputs["support"].shape == (1, 4, MAP_HEIGHT, DEFAULT_W)
    assert outputs["skeleton_state"].shape[-1] == DEFAULT_W

    filled = TileFlowModel(model, config).fill(level, mask)
    assert_known_preserved(level, filled, mask)


def test_v1_3_style_prior_adaptive_smoke_preserves_contract():
    level = synthetic_level(DEFAULT_W)
    mask = center_expand(DEFAULT_W)
    idx = level_to_idx(level)
    uniform_probs = tuple([1.0 / len(CHAR2IDX)] * len(CHAR2IDX))
    config = TileFlowConfig(
        version="v1.3",
        position_channels=True,
        context_channels=True,
        structure_heads=True,
        skeleton_heads=True,
        skeleton_conditioning=True,
        support_conditioning=True,
        support_logit_bias=0.65,
        adaptive_support_bias=True,
        stochastic_decode=False,
        time_conditioning=True,
        dfm_source="style_prior",
        dfm_source_style_probs=(uniform_probs, uniform_probs, uniform_probs),
        sample_steps=2,
    )
    model = CategoricalFlowNet(config)
    x = make_model_input(
        idx,
        mask,
        position_channels=True,
        context_channels=True,
        reveal_unknown=True,
        time_value=0.5,
    ).unsqueeze(0)
    outputs = model.forward_all(x)
    assert outputs["tile"].shape == (1, len(CHAR2IDX), MAP_HEIGHT, DEFAULT_W)
    assert outputs["support"].shape == (1, 4, MAP_HEIGHT, DEFAULT_W)
    assert outputs["skeleton_state"].shape[-1] == DEFAULT_W

    filled = TileFlowModel(model, config).fill(level, mask)
    assert_known_preserved(level, filled, mask)


def test_v1_4_candidate_selection_smoke_preserves_contract():
    level = synthetic_level(DEFAULT_W)
    mask = center_expand(DEFAULT_W)
    config = TileFlowConfig(
        version="v1.4",
        position_channels=True,
        context_channels=True,
        structure_heads=True,
        skeleton_heads=True,
        skeleton_conditioning=True,
        support_conditioning=True,
        support_logit_bias=0.65,
        stochastic_decode=True,
        time_conditioning=True,
        dfm_source="train_prior",
        dfm_source_probs=tuple([1.0 / len(CHAR2IDX)] * len(CHAR2IDX)),
        sample_steps=2,
        candidate_samples=2,
    )
    model = CategoricalFlowNet(config)
    filled = TileFlowModel(model, config).fill(level, mask)
    assert_known_preserved(level, filled, mask)
