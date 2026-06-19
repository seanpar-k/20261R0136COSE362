"""Render generated TileFlow maps and benchmark visual image sets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tileflow-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tileflow-cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileflow.common.continuity import continuity_metrics
from tileflow.common.eval import playability, structural_violations
from tileflow.common.fill_api import assert_known_preserved
from tileflow.common.masks import make_mask_set
from tileflow.common.render import load_text_level, render_level
from tileflow.common.style import style_metrics


@dataclass(frozen=True)
class SampleRecord:
    method: str
    source: str
    window_index: int
    mask_name: str
    sample_index: int
    sample_path: Path
    context_path: Path
    score: float
    metrics: dict[str, float | bool]


def infer_mask(mask_name: str, width: int):
    for name, mask in make_mask_set(W=width):
        if name == mask_name:
            return mask
    return None


def render_all_text_maps(root: Path, width: int, overlay: bool) -> int:
    rendered = 0
    for txt_path in sorted(root.rglob("*.txt")):
        level = load_text_level(txt_path)
        if txt_path.name == "context.txt":
            title = "context"
            mask = None
        else:
            mask_name = txt_path.parent.name
            level_name = txt_path.parents[2].name if len(txt_path.parents) > 2 else ""
            title = f"{level_name} / {mask_name} / {txt_path.stem}"
            mask = infer_mask(mask_name, width)
        png_path = txt_path.with_suffix(".png")
        render_level(level, png_path, title=title, mask=mask, show_overlay=overlay)
        rendered += 1
        print(f"rendered {png_path}")
    return rendered


def _parse_sample_path(root: Path, txt_path: Path) -> tuple[str, str, int, str, int, Path] | None:
    if not txt_path.name.startswith("sample_") or txt_path.suffix != ".txt":
        return None

    sample_dir = txt_path.parent
    mask_name = sample_dir.name
    window_dir = sample_dir.parent
    source_dir = window_dir.parent
    if not window_dir.name.startswith("window_"):
        return None

    try:
        window_index = int(window_dir.name.removeprefix("window_"))
        sample_index = int(txt_path.stem.removeprefix("sample_"))
    except ValueError:
        return None

    context_path = window_dir / "context.txt"
    if not context_path.exists():
        return None

    method = root.name if (root / "metrics.json").exists() else source_dir.parent.name
    return method, source_dir.name, window_index, mask_name, sample_index, context_path


def _sample_score(sample: list[str], context: list[str], mask) -> tuple[float, dict[str, float | bool]]:
    known_preserved = True
    try:
        assert_known_preserved(context, sample, mask)
    except AssertionError:
        known_preserved = False

    play = playability(sample)
    struct = structural_violations(sample)
    continuity = continuity_metrics(sample, mask)
    style = style_metrics(sample, mask)

    progress = float(play["progress"])
    completable = bool(play["completable"])
    struct_error = float(struct["per_col"])
    continuity_score = float(continuity["continuity_score"])
    descriptor_distance = float(style["descriptor_distance"])

    score = 0.0
    score += 1000.0 if not known_preserved else 0.0
    score += 10.0 if not completable else 0.0
    score += 2.0 * (1.0 - progress)
    score += struct_error
    score += 2.0 * (1.0 - continuity_score)
    score += descriptor_distance

    return score, {
        "known_preserved": known_preserved,
        "completable": completable,
        "progress": progress,
        "struct_viol_per_col": struct_error,
        "continuity_score": continuity_score,
        "descriptor_distance": descriptor_distance,
        "selection_score": float(score),
    }


def _collect_samples(root: Path, width: int, mask_name: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    mask = infer_mask(mask_name, width)
    if mask is None:
        raise ValueError(f"Unknown mask for benchmark visual set: {mask_name}")

    for txt_path in sorted(root.rglob("sample_*.txt")):
        parsed = _parse_sample_path(root, txt_path)
        if parsed is None:
            continue
        method, source, window_index, sample_mask, sample_index, context_path = parsed
        if sample_mask != mask_name:
            continue

        sample = load_text_level(txt_path)
        context = load_text_level(context_path)
        score, metrics = _sample_score(sample, context, mask)
        records.append(
            SampleRecord(
                method=method,
                source=source,
                window_index=window_index,
                mask_name=sample_mask,
                sample_index=sample_index,
                sample_path=txt_path,
                context_path=context_path,
                score=score,
                metrics=metrics,
            )
        )
    return records


def _choose_canonical_windows(records: Iterable[SampleRecord]) -> dict[str, int]:
    by_source_method: dict[tuple[str, str], set[int]] = defaultdict(set)
    methods_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_source_method[(record.source, record.method)].add(record.window_index)
        methods_by_source[record.source].add(record.method)

    canonical: dict[str, int] = {}
    for source, methods in sorted(methods_by_source.items()):
        common: set[int] | None = None
        union: set[int] = set()
        for method in methods:
            windows = by_source_method[(source, method)]
            union |= windows
            common = set(windows) if common is None else common & windows
        candidates = sorted(common or union)
        if candidates:
            canonical[source] = candidates[len(candidates) // 2]
    return canonical


def _nearest_window(records: list[SampleRecord], target: int) -> int | None:
    windows = sorted({record.window_index for record in records})
    if not windows:
        return None
    return min(windows, key=lambda value: (abs(value - target), value))


def _ranked(records: list[SampleRecord]) -> list[SampleRecord]:
    return sorted(records, key=lambda record: (record.score, record.sample_index))


def _role_selection(records: list[SampleRecord]) -> dict[str, SampleRecord]:
    ranked = _ranked(records)
    if not ranked:
        return {}
    return {
        "best": ranked[0],
        "median": ranked[len(ranked) // 2],
        "worst": ranked[-1],
    }


def _default_visual_root(root: Path) -> Path:
    if (root / "metrics.json").exists():
        return root.parent / "visual_sets"
    return root / "visual_sets"


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _render_selected(
    record: SampleRecord,
    output_path: Path,
    role: str,
    width: int,
    overlay: bool,
) -> dict:
    mask = infer_mask(record.mask_name, width)
    title = (
        f"{record.method} / {record.source} / window_{record.window_index:03d} "
        f"/ sample_{record.sample_index:03d} / {role}"
    )
    render_level(
        load_text_level(record.sample_path),
        output_path,
        title=title,
        mask=mask,
        show_overlay=overlay,
    )
    return {
        "method": record.method,
        "source": record.source,
        "window_index": record.window_index,
        "sample_index": record.sample_index,
        "role": role,
        "image": output_path.as_posix(),
        "sample": record.sample_path.as_posix(),
        "context": record.context_path.as_posix(),
        "metrics": record.metrics,
    }


def _render_context(record: SampleRecord, output_path: Path, width: int, overlay: bool) -> dict:
    mask = infer_mask(record.mask_name, width)
    render_level(
        load_text_level(record.context_path),
        output_path,
        title=f"{record.source} / window_{record.window_index:03d} / context",
        mask=mask,
        show_overlay=overlay,
    )
    return {
        "source": record.source,
        "window_index": record.window_index,
        "image": output_path.as_posix(),
        "context": record.context_path.as_posix(),
    }


def _write_visual_readme(visual_root: Path, manifest: dict) -> None:
    lines = [
        "# Benchmark Visual Sets",
        "",
        "Generated visual subsets for human inspection. The full benchmark sample",
        "archive remains under each `results/benchmarks/<method>/` directory and is",
        "used for quantitative metrics.",
        "",
        "Selection policy:",
        "",
        "- `main`: median-quality sample for each method/source pair",
        "- `appendix`: best, median, and worst samples for each method/source pair",
        "- contexts use the same deterministic canonical window per source",
        "- lower selection score is better",
        "",
        "## Canonical Windows",
        "",
        "| Source | Window |",
        "| --- | --- |",
    ]
    for source, window_index in sorted(manifest["canonical_windows"].items()):
        lines.append(f"| {source} | window_{int(window_index):03d} |")

    lines.extend(["", "## Main Set", "", "| Source | Method | Image | Score |", "| --- | --- | --- | --- |"])
    for entry in manifest["main"]:
        rel_image = _rel(Path(entry["image"]), visual_root)
        score = entry["metrics"]["selection_score"]
        lines.append(
            f"| {entry['source']} | {entry['method']} | "
            f"![{entry['method']} {entry['source']}]({rel_image}) | {score:.4f} |"
        )

    lines.extend(["", "## Appendix Set", "", "| Source | Method | Role | Image | Score |", "| --- | --- | --- | --- | --- |"])
    for entry in manifest["appendix"]:
        rel_image = _rel(Path(entry["image"]), visual_root)
        score = entry["metrics"]["selection_score"]
        lines.append(
            f"| {entry['source']} | {entry['method']} | {entry['role']} | "
            f"![{entry['role']} {entry['method']} {entry['source']}]({rel_image}) | {score:.4f} |"
        )

    (visual_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_benchmark_image_sets(
    root: Path,
    width: int,
    mask_name: str,
    overlay: bool,
    visual_root: Path | None = None,
) -> dict:
    records = _collect_samples(root, width=width, mask_name=mask_name)
    if not records:
        raise ValueError(f"No benchmark samples found under {root} for mask {mask_name}.")

    visual_root = visual_root or _default_visual_root(root)
    visual_root.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_key[(record.source, record.method)].append(record)

    canonical_windows = _choose_canonical_windows(records)
    main_entries: list[dict] = []
    appendix_entries: list[dict] = []
    context_entries: dict[tuple[str, int], dict] = {}

    for source in sorted({record.source for record in records}):
        target_window = canonical_windows[source]
        for method in sorted({record.method for record in records if record.source == source}):
            method_source_records = by_key[(source, method)]
            chosen_window = _nearest_window(method_source_records, target_window)
            if chosen_window is None:
                continue
            window_records = [
                record for record in method_source_records
                if record.window_index == chosen_window
            ]
            selected = _role_selection(window_records)
            if not selected:
                continue

            context_key = (source, chosen_window)
            if context_key not in context_entries:
                context_path = visual_root / "contexts" / f"{source}__window_{chosen_window:03d}__context.png"
                context_entries[context_key] = _render_context(
                    selected["median"],
                    context_path,
                    width=width,
                    overlay=overlay,
                )

            main_record = selected["median"]
            main_path = (
                visual_root
                / "main"
                / source
                / f"{method}__{source}__window_{main_record.window_index:03d}"
                f"__sample_{main_record.sample_index:03d}__median.png"
            )
            main_entries.append(_render_selected(main_record, main_path, "median", width, overlay))

            for role in ["best", "median", "worst"]:
                record = selected[role]
                appendix_path = (
                    visual_root
                    / "appendix"
                    / source
                    / f"{method}__{source}__window_{record.window_index:03d}"
                    f"__sample_{record.sample_index:03d}__{role}.png"
                )
                appendix_entries.append(_render_selected(record, appendix_path, role, width, overlay))

    manifest = {
        "root": root.as_posix(),
        "visual_root": visual_root.as_posix(),
        "mask_name": mask_name,
        "selection_policy": {
            "main": "median-quality sample per eval source and method",
            "appendix": "best, median, and worst samples per eval source and method",
            "canonical_window": "middle available window shared by methods when possible",
            "score": "lower is better; combines contract, playability, structure, continuity, and style",
        },
        "canonical_windows": canonical_windows,
        "contexts": list(context_entries.values()),
        "main": main_entries,
        "appendix": appendix_entries,
    }
    (visual_root / "visual_selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_visual_readme(visual_root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="results/tileflow")
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--image-sets", action="store_true")
    parser.add_argument("--skip-all", action="store_true")
    parser.add_argument("--mask-name", default="center_expand")
    parser.add_argument("--visual-root", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rendered = 0
    if not args.skip_all:
        rendered = render_all_text_maps(root, width=args.width, overlay=args.overlay)
        print(f"rendered_count={rendered}")

    if args.image_sets:
        manifest = render_benchmark_image_sets(
            root,
            width=args.width,
            mask_name=args.mask_name,
            overlay=True,
            visual_root=Path(args.visual_root) if args.visual_root else None,
        )
        print(f"visual_root={manifest['visual_root']}")
        print(f"main_count={len(manifest['main'])}")
        print(f"appendix_count={len(manifest['appendix'])}")


if __name__ == "__main__":
    main()
