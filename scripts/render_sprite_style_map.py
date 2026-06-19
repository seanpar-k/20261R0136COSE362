"""Render text Mario levels with custom sprite-like tiles.

The renderer intentionally uses simple generated shapes rather than original
Mario game assets. It is meant for presentation/debug readability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.common.data import DEFAULT_W, MAP_HEIGHT, normalize_level
from tileflow.common.masks import make_mask_set


SKY = (104, 177, 232)
SKY2 = (127, 199, 245)
GROUND = (146, 83, 36)
GROUND_DARK = (93, 52, 24)
BRICK = (205, 96, 28)
QUESTION = (245, 180, 36)
USED_BLOCK = (224, 164, 48)
PIPE = (36, 155, 60)
PIPE_DARK = (26, 103, 42)
PIPE_LIGHT = (93, 210, 91)
COIN = (255, 216, 35)
COIN_DARK = (209, 143, 19)
ENEMY = (142, 71, 34)
ENEMY_DARK = (64, 35, 24)
CANNON = (74, 76, 82)
CANNON_DARK = (28, 29, 34)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_level(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line]


def draw_sky(draw: ImageDraw.ImageDraw, x: int, y: int, s: int) -> None:
    draw.rectangle([x, y, x + s, y + s], fill=SKY)
    draw.rectangle([x, y, x + s, y + s // 3], fill=SKY2)


def draw_ground(draw: ImageDraw.ImageDraw, x: int, y: int, s: int) -> None:
    draw.rectangle([x, y, x + s, y + s], fill=GROUND)
    draw.rectangle([x, y, x + s, y + 2], fill=(180, 110, 52))
    for ox in range(0, s, max(6, s // 3)):
        draw.line([x + ox, y, x + ox + s // 2, y + s], fill=GROUND_DARK, width=1)
    draw.rectangle([x, y, x + s, y + s], outline=GROUND_DARK)


def draw_block(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, fill: tuple[int, int, int], label: str = "") -> None:
    draw.rectangle([x + 1, y + 1, x + s - 1, y + s - 1], fill=fill, outline=(115, 66, 26))
    draw.rectangle([x + 3, y + 3, x + s - 4, y + s // 2], outline=(255, 229, 114))
    if label:
        font = _font(max(10, s // 2))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (s - (bbox[2] - bbox[0])) // 2, y + (s - (bbox[3] - bbox[1])) // 2 - 1), label, fill=(92, 47, 20), font=font)


def draw_coin(draw: ImageDraw.ImageDraw, x: int, y: int, s: int) -> None:
    pad = max(3, s // 5)
    draw.ellipse([x + pad, y + 2, x + s - pad, y + s - 2], fill=COIN, outline=COIN_DARK, width=max(1, s // 12))
    draw.arc([x + pad + 4, y + 5, x + s - pad - 4, y + s - 5], start=90, end=270, fill=(255, 245, 135), width=2)


def draw_pipe(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, ch: str) -> None:
    draw.rectangle([x + 1, y + 1, x + s - 1, y + s - 1], fill=PIPE, outline=PIPE_DARK)
    if ch in "<>":
        if ch == "<":
            draw.rectangle([x + 1, y + 1, x + s, y + s - 1], fill=PIPE, outline=PIPE_DARK)
            draw.rectangle([x + s // 4, y + 3, x + s - 2, y + s - 4], fill=PIPE_LIGHT)
        else:
            draw.rectangle([x, y + 1, x + s - 1, y + s - 1], fill=PIPE, outline=PIPE_DARK)
            draw.rectangle([x + 2, y + 3, x + s // 2, y + s - 4], fill=PIPE_LIGHT)
    else:
        draw.rectangle([x + s // 5, y + 1, x + s // 5 + 3, y + s - 1], fill=PIPE_LIGHT)
        draw.line([x, y, x + s, y], fill=PIPE_DARK, width=2)


def draw_enemy(draw: ImageDraw.ImageDraw, x: int, y: int, s: int) -> None:
    draw.ellipse([x + 3, y + 2, x + s - 3, y + s - 5], fill=ENEMY, outline=ENEMY_DARK)
    draw.rectangle([x + 5, y + s // 2, x + s - 5, y + s - 3], fill=(232, 205, 145), outline=ENEMY_DARK)
    eye = max(2, s // 8)
    draw.ellipse([x + s // 3 - eye, y + s // 3, x + s // 3 + eye, y + s // 3 + eye * 2], fill=ENEMY_DARK)
    draw.ellipse([x + 2 * s // 3 - eye, y + s // 3, x + 2 * s // 3 + eye, y + s // 3 + eye * 2], fill=ENEMY_DARK)


def draw_cannon(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, top: bool) -> None:
    draw.rectangle([x + 3, y + 2, x + s - 3, y + s - 1], fill=CANNON, outline=CANNON_DARK)
    if top:
        draw.rectangle([x + 1, y + 1, x + s - 1, y + s // 3], fill=CANNON_DARK)
        draw.rectangle([x + 4, y + 3, x + s - 4, y + s // 3 + 1], fill=(130, 132, 138))
    else:
        draw.rectangle([x + 5, y, x + s - 5, y + s], fill=(105, 107, 112))


def render(level: list[str], output: Path, mask: np.ndarray | None, tile_size: int = 24, legend: bool = True) -> None:
    rows = normalize_level(level, width=max(len(row) for row in level))
    width = len(rows[0])
    legend_w = 260 if legend else 0
    title_h = 40
    img = Image.new("RGB", (width * tile_size + legend_w, MAP_HEIGHT * tile_size + title_h), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(18)
    small_font = _font(13)
    draw.text((10, 10), "Sprite-style TileFlow render (custom tiles, not original game assets)", fill=(30, 30, 30), font=title_font)

    y0 = title_h
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            x = c * tile_size
            y = y0 + r * tile_size
            draw_sky(draw, x, y, tile_size)
            if ch == "X":
                draw_ground(draw, x, y, tile_size)
            elif ch == "S":
                draw_block(draw, x, y, tile_size, USED_BLOCK)
            elif ch == "Q":
                draw_block(draw, x, y, tile_size, BRICK)
            elif ch == "?":
                draw_block(draw, x, y, tile_size, QUESTION, "?")
            elif ch == "o":
                draw_coin(draw, x, y, tile_size)
            elif ch in "<>[]":
                draw_pipe(draw, x, y, tile_size, ch)
            elif ch == "E":
                draw_enemy(draw, x, y, tile_size)
            elif ch in "Bb":
                draw_cannon(draw, x, y, tile_size, top=(ch == "B"))

    if mask is not None:
        unknown = ~mask
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for c in range(width):
            if unknown[:, c].any():
                x1 = c * tile_size
                x2 = x1 + tile_size
                od.rectangle([x1, y0, x2, y0 + MAP_HEIGHT * tile_size], fill=(120, 92, 255, 30))
                if c == 0 or not unknown[:, c - 1].any():
                    od.line([x1, y0, x1, y0 + MAP_HEIGHT * tile_size], fill=(225, 30, 45, 230), width=2)
                if c == width - 1 or not unknown[:, c + 1].any():
                    od.line([x2, y0, x2, y0 + MAP_HEIGHT * tile_size], fill=(225, 30, 45, 230), width=2)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    if legend:
        lx = width * tile_size + 18
        ly = title_h + 8
        entries = [
            ("-", "sky / empty"),
            ("X", "ground / solid"),
            ("S", "solid block"),
            ("Q", "brick block"),
            ("?", "question block"),
            ("o", "coin"),
            ("<>", "pipe top"),
            ("[]", "pipe body"),
            ("E", "enemy"),
            ("B/b", "cannon"),
        ]
        draw.text((lx, ly), "Legend", fill=(30, 30, 30), font=title_font)
        ly += 32
        for token, label in entries:
            tx, ty = lx, ly
            if token == "-":
                draw_sky(draw, tx, ty, tile_size)
            elif token == "X":
                draw_ground(draw, tx, ty, tile_size)
            elif token == "S":
                draw_block(draw, tx, ty, tile_size, USED_BLOCK)
            elif token == "Q":
                draw_block(draw, tx, ty, tile_size, BRICK)
            elif token == "?":
                draw_block(draw, tx, ty, tile_size, QUESTION, "?")
            elif token == "o":
                draw_sky(draw, tx, ty, tile_size)
                draw_coin(draw, tx, ty, tile_size)
            elif token == "<>":
                draw_pipe(draw, tx, ty, tile_size, "<")
                draw_pipe(draw, tx + tile_size, ty, tile_size, ">")
            elif token == "[]":
                draw_pipe(draw, tx, ty, tile_size, "[")
                draw_pipe(draw, tx + tile_size, ty, tile_size, "]")
            elif token == "E":
                draw_sky(draw, tx, ty, tile_size)
                draw_enemy(draw, tx, ty, tile_size)
            elif token == "B/b":
                draw_cannon(draw, tx, ty, tile_size, top=True)
            draw.text((lx + 62, ly + 5), f"{token}: {label}", fill=(30, 30, 30), font=small_font)
            ly += tile_size + 10
        draw.text((lx, ly + 6), "Purple tint: generated area", fill=(70, 50, 130), font=small_font)
        draw.text((lx, ly + 26), "Red lines: mask boundary", fill=(160, 30, 45), font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="")
    parser.add_argument("--mask", default="")
    parser.add_argument("--tile-size", type=int, default=24)
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_sprite_style.png")
    masks = dict(make_mask_set(args.mask_seed, W=args.width))
    mask = masks.get(args.mask) if args.mask else None
    render(read_level(input_path), output, mask=mask, tile_size=args.tile_size, legend=not args.no_legend)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
