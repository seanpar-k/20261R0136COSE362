"""Prepare controlled same-data training inputs for external baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARIODIFFUSION_ROOT = ROOT / "external" / "MarioDiffusion"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MARIODIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(MARIODIFFUSION_ROOT))

from tileflow.common.data import DEFAULT_W, MAP_HEIGHT, load_level_windows

MARIODIFFUSION_HEIGHT = 16


def _load_splits(path: Path) -> dict[str, list[str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "eval": ["mario-1-2.txt", "mario-4-1.txt", "mario-6-3.txt"],
        "train": [],
    }


def _scene_to_strings(scene: list[list[int]], id_to_char: dict[int, str]) -> list[str]:
    return ["".join(id_to_char[int(tile)] for tile in row) for row in scene]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, default=Path("tileflow/common/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/same_data/data"))
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--dev-count", type=int, default=8)
    args = parser.parse_args()

    from captions.util import extract_tileset
    from create_ascii_captions import assign_caption
    from tokenizer import Tokenizer

    _, id_to_char_raw, char_to_id, tile_descriptors = extract_tileset(
        str(MARIODIFFUSION_ROOT / "datasets" / "smb.json")
    )
    id_to_char = {int(k): v for k, v in id_to_char_raw.items()}

    splits = _load_splits(args.splits)
    train_files = splits.get("train")
    if not train_files:
        eval_files = set(splits["eval"])
        train_files = sorted(path.name for path in args.data_dir.glob("*.txt") if path.name not in eval_files)

    train_windows = load_level_windows(
        args.data_dir,
        width=args.width,
        stride=args.stride,
        file_names=train_files,
    )
    if len(train_windows) <= args.dev_count:
        raise ValueError(f"Need more than {args.dev_count} windows, found {len(train_windows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_windows = train_windows[: args.dev_count]
    baseline_train_windows = train_windows[args.dev_count :]

    def convert(records) -> list[dict]:
        out = []
        for record in records:
            padded_level = ["-" * args.width for _ in range(MARIODIFFUSION_HEIGHT - MAP_HEIGHT)]
            padded_level.extend(record.level)
            scene = [[char_to_id.get(ch, char_to_id["-"]) for ch in row] for row in padded_level]
            caption = assign_caption(
                scene,
                id_to_char,
                char_to_id,
                tile_descriptors,
                False,
                False,
            )
            out.append(
                {
                    "prompt": None,
                    "scene": scene,
                    "caption": caption,
                    "source": record.source,
                    "window_index": record.window_index,
                }
            )
        return out

    train_json = convert(baseline_train_windows)
    val_json = convert(dev_windows)

    (args.output_dir / "mariodiffusion_train.json").write_text(
        json.dumps(train_json, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "mariodiffusion_val.json").write_text(
        json.dumps(val_json, indent=2),
        encoding="utf-8",
    )

    tokenizer = Tokenizer()
    tokenizer.build_vocab(str(args.output_dir / "mariodiffusion_train.json"))
    tokenizer.save(str(args.output_dir / "mariodiffusion_tokenizer.pkl"))

    mario_gpt_text = "\n".join("\n".join(record.level) for record in baseline_train_windows)
    (args.output_dir / "mariogpt_train.txt").write_text(mario_gpt_text, encoding="utf-8")

    manifest = {
        "scope": "same-data external baseline training inputs",
        "train_files": train_files,
        "eval_files": splits["eval"],
        "width": args.width,
        "stride": args.stride,
        "train_windows": len(baseline_train_windows),
        "val_windows": len(dev_windows),
        "paths": {
            "mariogpt_train": str(args.output_dir / "mariogpt_train.txt"),
            "mariodiffusion_train": str(args.output_dir / "mariodiffusion_train.json"),
            "mariodiffusion_val": str(args.output_dir / "mariodiffusion_val.json"),
            "mariodiffusion_tokenizer": str(args.output_dir / "mariodiffusion_tokenizer.pkl"),
        },
        "sample_train_caption": train_json[0]["caption"] if train_json else None,
        "sample_train_level": _scene_to_strings(train_json[0]["scene"], id_to_char) if train_json else None,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
