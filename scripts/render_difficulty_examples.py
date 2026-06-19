"""Render controlled easy/neutral/hard TileFlow examples from one center seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.common.data import DEFAULT_W, load_level_windows
from tileflow.common.eval import playability, structural_violations
from tileflow.common.masks import center_expand
from tileflow.common.render import render_level
from tileflow.common.style import style_metrics
from tileflow.models import load_tileflow_stochastic_guided_ensemble


def write_level(path: Path, level: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(level) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--version", default="v0.13")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", default="mario-1-2.txt")
    parser.add_argument("--window-index", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--structure-coupling", action="store_true")
    parser.add_argument("--coupling-strength", type=float, default=2.8)
    parser.add_argument("--learned-structure-bias", action="store_true")
    parser.add_argument("--learned-bias-strength", type=float, default=1.0)
    parser.add_argument("--utility-guidance", action="store_true")
    parser.add_argument("--utility-guidance-strength", type=float, default=1.0)
    parser.add_argument("--coherence-guidance", action="store_true")
    parser.add_argument("--coherence-guidance-strength", type=float, default=1.0)
    parser.add_argument("--context-guidance", action="store_true")
    parser.add_argument("--context-guidance-strength", type=float, default=1.0)
    parser.add_argument("--terrain-guidance", action="store_true")
    parser.add_argument("--terrain-guidance-strength", type=float, default=1.0)
    parser.add_argument("--continuation-guidance", action="store_true")
    parser.add_argument("--continuation-guidance-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("results/tileflow") / f"tileflow_{args.version}" / "difficulty_examples"

    windows = load_level_windows(args.data_dir, width=args.width, stride=args.stride, file_names=[args.source])
    selected = next((record for record in windows if record.window_index == args.window_index), windows[len(windows) // 2])
    mask = center_expand(args.width)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_path = args.output_dir / "context.txt"
    write_level(context_path, selected.level)
    render_level(selected.level, args.output_dir / "context.png", title=f"{selected.source} / context", mask=mask, show_overlay=True)

    manifest = {
        "source": selected.source,
        "window_index": selected.window_index,
        "context": context_path.as_posix(),
        "difficulties": {},
    }
    for difficulty in ["easy", "neutral", "hard"]:
        model = load_tileflow_stochastic_guided_ensemble(
            args.paths,
            args.weights,
            device=args.device,
            name=f"tileflow_{args.version}_{difficulty}",
            strength=args.strength,
            temperature=args.temperature,
            top_k=args.top_k,
            difficulty=difficulty,
            seed=args.seed,
            structure_coupling=args.structure_coupling,
            coupling_strength=args.coupling_strength,
            learned_structure_bias=args.learned_structure_bias,
            learned_bias_strength=args.learned_bias_strength,
            utility_guidance=args.utility_guidance,
            utility_guidance_strength=args.utility_guidance_strength,
            coherence_guidance=args.coherence_guidance,
            coherence_guidance_strength=args.coherence_guidance_strength,
            context_guidance=args.context_guidance,
            context_guidance_strength=args.context_guidance_strength,
            terrain_guidance=args.terrain_guidance,
            terrain_guidance_strength=args.terrain_guidance_strength,
            continuation_guidance=args.continuation_guidance,
            continuation_guidance_strength=args.continuation_guidance_strength,
        )
        entries = []
        for sample_index in range(args.samples):
            level = model.fill(selected.level, mask)
            sample_path = args.output_dir / difficulty / f"sample_{sample_index:03d}.txt"
            image_path = sample_path.with_suffix(".png")
            write_level(sample_path, level)
            render_level(
                level,
                image_path,
                title=f"tileflow_{args.version} / {difficulty} / sample_{sample_index:03d}",
                mask=mask,
                show_overlay=True,
            )
            play = playability(level)
            struct = structural_violations(level)
            style = style_metrics(level, mask)
            entries.append(
                {
                    "sample": sample_path.as_posix(),
                    "image": image_path.as_posix(),
                    "completable": bool(play["completable"]),
                    "progress": float(play["progress"]),
                    "struct_viol_per_col": float(struct["per_col"]),
                    "descriptor_distance": float(style["descriptor_distance"]),
                    "ground_void_ratio_gap": float(style["ground_void_ratio_gap"]),
                    "obstacle_density_gap": float(style["obstacle_density_gap"]),
                }
            )
        manifest["difficulties"][difficulty] = entries

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
