from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageStat


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_indexed_images(image_dir: str) -> Dict[int, Path]:
    images: Dict[int, Path] = {}
    for path in Path(image_dir).iterdir():
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            idx = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        images[idx] = path
    return images


def parse_ranges(value: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if not value:
        return ranges
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending range: {part}")
        else:
            start = end = int(part)
        ranges.append((start, end))
    return ranges


def contiguous_ranges(indices: Iterable[int]) -> List[Tuple[int, int]]:
    sorted_indices = sorted(set(indices))
    if not sorted_indices:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def frame_stats(path: Path) -> dict:
    with Image.open(path).convert("RGB") as img:
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        return {
            "size": [img.width, img.height],
            "mean": stat.mean[0],
            "max": gray.getextrema()[1],
        }


def classify_frames(
    frames: Dict[int, Path],
    black_mean_threshold: float,
    black_max_threshold: int,
    dark_mean_threshold: float,
) -> dict:
    sizes: Dict[str, int] = {}
    black: List[int] = []
    dark: List[int] = []
    for idx, path in sorted(frames.items()):
        stats = frame_stats(path)
        sizes[f"{stats['size'][0]}x{stats['size'][1]}"] = sizes.get(f"{stats['size'][0]}x{stats['size'][1]}", 0) + 1
        if stats["mean"] <= black_mean_threshold and stats["max"] <= black_max_threshold:
            black.append(idx)
        if stats["mean"] <= dark_mean_threshold:
            dark.append(idx)
    return {
        "count": len(frames),
        "sizes": sizes,
        "black_ranges": contiguous_ranges(black),
        "dark_ranges": contiguous_ranges(dark),
    }


def find_path(frames: Dict[int, Path], idx: int, source_offset: int = 0) -> Optional[Path]:
    return frames.get(idx + source_offset)


def draw_contact_sheet(
    sources: Sequence[Tuple[str, Dict[int, Path], int]],
    start: int,
    end: int,
    output_path: Path,
    thumb_width: int = 180,
    label_height: int = 24,
    columns: int = 6,
) -> None:
    rows_per_frame = len(sources)
    tile_height = round(thumb_width * 9 / 16) + label_height
    frame_count = end - start + 1
    groups = math.ceil(frame_count / columns)
    sheet_width = columns * thumb_width
    sheet_height = groups * rows_per_frame * tile_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for n, idx in enumerate(range(start, end + 1)):
        col = n % columns
        group_row = n // columns
        x = col * thumb_width
        for source_row, (label, frames, offset) in enumerate(sources):
            y = (group_row * rows_per_frame + source_row) * tile_height
            path = find_path(frames, idx, offset)
            draw.rectangle((x, y, x + thumb_width - 1, y + tile_height - 1), outline=(210, 210, 210))
            draw.text((x + 4, y + 4), f"{idx:04d} {label}", fill=(0, 0, 0))
            if path is None:
                draw.text((x + 4, y + label_height + 28), "missing", fill=(180, 0, 0))
                continue
            with Image.open(path).convert("RGB") as img:
                img.thumbnail((thumb_width, tile_height - label_height), Image.Resampling.LANCZOS)
                px = x + (thumb_width - img.width) // 2
                py = y + label_height + ((tile_height - label_height) - img.height) // 2
                sheet.paste(img, (px, py))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit trailer still-frame sets and build contact sheets.")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--stylized_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_frames", type=int, default=None)
    parser.add_argument("--source_index_offset", type=int, default=0)
    parser.add_argument("--contact_ranges", default="", help="Comma-separated inclusive ranges, e.g. 0-69,1118-1209.")
    parser.add_argument("--black_mean_threshold", type=float, default=8.0)
    parser.add_argument("--black_max_threshold", type=int, default=45)
    parser.add_argument("--dark_mean_threshold", type=float, default=28.0)
    args = parser.parse_args()

    source = collect_indexed_images(args.source_dir)
    stylized = collect_indexed_images(args.stylized_dir)
    expected = args.expected_frames
    if expected is None:
        max_idx = max([*source.keys(), *stylized.keys()])
        expected = max_idx + 1

    missing_stylized = [idx for idx in range(expected) if idx not in stylized]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source_dir": args.source_dir,
        "stylized_dir": args.stylized_dir,
        "expected_frames": expected,
        "source": classify_frames(source, args.black_mean_threshold, args.black_max_threshold, args.dark_mean_threshold),
        "stylized": classify_frames(stylized, args.black_mean_threshold, args.black_max_threshold, args.dark_mean_threshold),
        "missing_stylized_ranges": contiguous_ranges(missing_stylized),
        "contact_sheets": [],
    }

    for start, end in parse_ranges(args.contact_ranges):
        output_path = out_dir / f"contact_{start}_{end}.jpg"
        draw_contact_sheet(
            [
                ("source", source, args.source_index_offset),
                ("stylized", stylized, 0),
            ],
            start,
            end,
            output_path,
        )
        report["contact_sheets"].append(str(output_path))

    report_path = out_dir / "frame_audit.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(report_path), "contact_sheets": report["contact_sheets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
