from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
from PIL import ImageStat

FIT_MODES = ("contain", "cover", "active_aspect")
PRESETS = ("manual", "clean_still_trailer")


def _run_json(cmd: List[str]) -> dict:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def media_duration(path: str) -> float:
    data = _run_json([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ])
    return float(data["format"]["duration"])


def _collect_indexed_images(image_dir: str) -> Dict[int, Path]:
    images: Dict[int, Path] = {}
    for path in Path(image_dir).iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = path.stem
        try:
            idx = int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        images[idx] = path
    return images


def _nearest_index(target: int, available: List[int]) -> Optional[int]:
    if not available:
        return None
    return min(available, key=lambda idx: (abs(idx - target), idx))


def _parse_ranges(value: str) -> Set[int]:
    indices: Set[int] = set()
    if not value:
        return indices
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
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def _merge_ranges(*values: str) -> str:
    parts: List[str] = []
    for value in values:
        if value:
            parts.extend(part.strip() for part in value.split(",") if part.strip())
    return ",".join(parts)


def _source_path_for(source_image_dir: str, index: int, offset: int) -> Optional[Path]:
    base = Path(source_image_dir)
    source_index = index + offset
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = base / f"frame_{source_index:06d}{ext}"
        if path.exists():
            return path
    return None


def _is_black_source_frame(path: Path, mean_threshold: float, max_threshold: int) -> bool:
    with Image.open(path).convert("L") as img:
        stat = ImageStat.Stat(img)
        max_value = img.getextrema()[1]
        return stat.mean[0] <= mean_threshold and max_value <= max_threshold


def _apply_letterbox_matte(canvas: Image.Image, letterbox_aspect: Optional[float]) -> None:
    if not letterbox_aspect:
        return
    width, height = canvas.size
    active_height = round(width / letterbox_aspect)
    if active_height >= height:
        return
    if active_height <= 0:
        raise ValueError("--letterbox_aspect must produce a positive active image height")
    bar = (height - active_height) // 2
    if bar <= 0:
        return
    canvas.paste("black", (0, 0, width, bar))
    canvas.paste("black", (0, height - bar, width, height))


def _apply_fixed_letterbox_matte(canvas: Image.Image, pixels: Optional[int]) -> None:
    if not pixels:
        return
    if pixels < 0:
        raise ValueError("--overlay_letterbox_pixels must be >= 0")
    width, height = canvas.size
    bar = min(pixels, height // 2)
    if bar <= 0:
        return
    canvas.paste("black", (0, 0, width, bar))
    canvas.paste("black", (0, height - bar, width, height))


def _is_dark_strip(img: Image.Image, box: Tuple[int, int, int, int], mean_threshold: float, max_threshold: int) -> bool:
    strip = img.crop(box).convert("L")
    if strip.width <= 0 or strip.height <= 0:
        return False
    stat = ImageStat.Stat(strip)
    extrema = strip.getextrema()
    if extrema is None:
        return False
    return stat.mean[0] <= mean_threshold and extrema[1] <= max_threshold


def _crop_black_borders(
    img: Image.Image,
    mean_threshold: float = 6.0,
    max_threshold: int = 32,
    min_crop_px: int = 8,
) -> Image.Image:
    width, height = img.size
    top = 0
    bottom = height
    left = 0
    right = width

    while top < bottom and _is_dark_strip(img, (0, top, width, top + 1), mean_threshold, max_threshold):
        top += 1
    while bottom > top and _is_dark_strip(img, (0, bottom - 1, width, bottom), mean_threshold, max_threshold):
        bottom -= 1
    while left < right and _is_dark_strip(img, (left, top, left + 1, bottom), mean_threshold, max_threshold):
        left += 1
    while right > left and _is_dark_strip(img, (right - 1, top, right, bottom), mean_threshold, max_threshold):
        right -= 1

    crop_w = right - left
    crop_h = bottom - top
    cropped_any = (
        top >= min_crop_px
        or height - bottom >= min_crop_px
        or left >= min_crop_px
        or width - right >= min_crop_px
    )
    if not cropped_any or crop_w <= 0 or crop_h <= 0:
        return img
    if crop_w < width * 0.35 or crop_h < height * 0.35:
        return img
    return img.crop((left, top, right, bottom))


def _center_crop_to_aspect(img: Image.Image, aspect: Optional[float]) -> Image.Image:
    if not aspect:
        return img
    width, height = img.size
    current = width / height
    if abs(current - aspect) < 0.01:
        return img
    if current > aspect:
        new_width = round(height * aspect)
        left = max(0, (width - new_width) // 2)
        return img.crop((left, 0, left + new_width, height))
    new_height = round(width / aspect)
    top = max(0, (height - new_height) // 2)
    return img.crop((0, top, width, top + new_height))


def _paste_contain(src: Image.Image, canvas: Image.Image, box: Tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    target_w = right - left
    target_h = bottom - top
    img = src.copy()
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    x = left + (target_w - img.width) // 2
    y = top + (target_h - img.height) // 2
    canvas.paste(img, (x, y))


def _write_normalized_frame(
    src_path: Path,
    out_path: Path,
    width: int,
    height: int,
    letterbox_aspect: Optional[float] = None,
    crop_bars: bool = False,
    crop_to_aspect: Optional[float] = None,
    fit_mode: str = "contain",
    overlay_letterbox_pixels: Optional[int] = None,
) -> None:
    with Image.open(src_path).convert("RGB") as img:
        if crop_bars:
            img = _crop_black_borders(img)
        img = _center_crop_to_aspect(img, crop_to_aspect)
        canvas = Image.new("RGB", (width, height), "black")
        if fit_mode == "active_aspect":
            if not letterbox_aspect:
                raise ValueError("--generated_fit_mode active_aspect requires --letterbox_aspect")
            active_h = min(height, round(width / letterbox_aspect))
            active_top = (height - active_h) // 2
            _paste_contain(img, canvas, (0, active_top, width, active_top + active_h))
        elif fit_mode == "cover":
            scale = max(width / img.width, height / img.height)
            resized = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
            x = (resized.width - width) // 2
            y = (resized.height - height) // 2
            canvas.paste(resized.crop((x, y, x + width, y + height)), (0, 0))
        elif fit_mode == "contain":
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = (width - img.width) // 2
            y = (height - img.height) // 2
            canvas.paste(img, (x, y))
        else:
            raise ValueError(f"Unknown fit mode: {fit_mode}")
        _apply_letterbox_matte(canvas, letterbox_aspect)
        _apply_fixed_letterbox_matte(canvas, overlay_letterbox_pixels)
        canvas.save(out_path, format="PNG")


def _apply_sequence_preset(
    preset: str,
    *,
    source_ranges: str,
    title_source_ranges: str,
    tail_source_range: str,
    source_black_mean_threshold: Optional[float],
    source_black_max_threshold: int,
    crop_source_bars: bool,
    source_fit_mode: str,
    generated_fit_mode: str,
    overlay_letterbox_pixels: Optional[int],
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown sequence preset: {preset}")
    if preset == "manual":
        return {
            "source_ranges": source_ranges,
            "source_black_mean_threshold": source_black_mean_threshold,
            "source_black_max_threshold": source_black_max_threshold,
            "crop_source_bars": crop_source_bars,
            "source_fit_mode": source_fit_mode,
            "generated_fit_mode": generated_fit_mode,
            "overlay_letterbox_pixels": overlay_letterbox_pixels,
        }

    return {
        "source_ranges": _merge_ranges(source_ranges, title_source_ranges, tail_source_range),
        "source_black_mean_threshold": (
            5.0 if source_black_mean_threshold is None else source_black_mean_threshold
        ),
        "source_black_max_threshold": source_black_max_threshold,
        "crop_source_bars": True if not crop_source_bars else crop_source_bars,
        "source_fit_mode": "cover" if source_fit_mode == "contain" else source_fit_mode,
        "generated_fit_mode": "cover" if generated_fit_mode == "contain" else generated_fit_mode,
        "overlay_letterbox_pixels": overlay_letterbox_pixels,
    }


def build_sequence_trailer(
    image_dir: str,
    audio_source: str,
    output_path: str,
    input_fps: float,
    output_fps: float,
    expected_frames: Optional[int] = None,
    width: int = 1280,
    height: int = 720,
    fill_missing: bool = False,
    audio_start: float = 0.0,
    duration: Optional[float] = None,
    source_image_dir: Optional[str] = None,
    source_index_offset: int = 1,
    source_for_missing: bool = False,
    source_flash_for_missing: bool = False,
    source_ranges: str = "",
    source_black_mean_threshold: Optional[float] = None,
    source_black_max_threshold: int = 45,
    letterbox_aspect: Optional[float] = None,
    crop_generated_bars: bool = False,
    generated_fit_mode: str = "contain",
    generated_crop_aspect: Optional[float] = None,
    crop_source_bars: bool = False,
    source_fit_mode: str = "contain",
    overlay_letterbox_pixels: Optional[int] = None,
    preset: str = "manual",
) -> dict:
    indexed = _collect_indexed_images(image_dir)
    if not indexed:
        raise ValueError(f"No indexed images found in {image_dir}")

    if expected_frames is None:
        expected_frames = max(indexed) + 1

    available = sorted(indexed)
    missing = [idx for idx in range(expected_frames) if idx not in indexed]
    if source_flash_for_missing:
        source_for_missing = True

    if missing and not fill_missing and not source_for_missing:
        raise ValueError(
            f"Missing {len(missing)} frames. Re-run with --fill_missing to hold nearest generated frames."
        )
    if (source_for_missing or source_ranges or source_black_mean_threshold is not None) and not source_image_dir:
        raise ValueError("Source-frame fallback requires --source_image_dir")

    explicit_source_indices = _parse_ranges(source_ranges)

    output = Path(output_path)
    work_dir = output.parent / f"{output.stem}_sequence"
    seq_dir = work_dir / "frames"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    filled = []
    source_overrides = []
    for idx in range(expected_frames):
        use_source = idx in explicit_source_indices
        source_path = _source_path_for(source_image_dir, idx, source_index_offset) if source_image_dir else None
        if (
            not use_source
            and source_black_mean_threshold is not None
            and source_path is not None
            and _is_black_source_frame(source_path, source_black_mean_threshold, source_black_max_threshold)
        ):
            use_source = True

        src_idx = idx
        if src_idx not in indexed and source_for_missing and source_path is not None:
            use_source = True

        if use_source:
            if source_path is None:
                raise ValueError(f"Source frame for index {idx} was requested but not found")
            source_overrides.append({"index": idx, "source": str(source_path)})
            _write_normalized_frame(
                source_path,
                seq_dir / f"frame_{idx:06d}.png",
                width,
                height,
                letterbox_aspect=letterbox_aspect,
                crop_bars=crop_source_bars,
                fit_mode=source_fit_mode,
                overlay_letterbox_pixels=overlay_letterbox_pixels,
            )
            continue

        if src_idx not in indexed:
            nearest = _nearest_index(src_idx, available)
            if nearest is None:
                raise ValueError("No generated frames available for fill.")
            src_idx = nearest
            filled.append({"index": idx, "source_index": src_idx})
        _write_normalized_frame(
            indexed[src_idx],
            seq_dir / f"frame_{idx:06d}.png",
            width,
            height,
            letterbox_aspect=letterbox_aspect,
            crop_bars=crop_generated_bars,
            crop_to_aspect=generated_crop_aspect,
            fit_mode=generated_fit_mode,
            overlay_letterbox_pixels=overlay_letterbox_pixels,
        )

    video_only = work_dir / "video_only.mp4"
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={output_fps},format=yuv420p"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(input_fps),
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        str(video_only),
    ], check=True)

    audio_total_duration = media_duration(audio_source)
    if audio_start < 0:
        raise ValueError("--audio_start must be >= 0")
    if audio_start >= audio_total_duration:
        raise ValueError("--audio_start must be before the end of the audio source")
    video_duration = media_duration(str(video_only))
    total_duration = duration if duration is not None else audio_total_duration - audio_start
    total_duration = min(total_duration, audio_total_duration - audio_start)
    if total_duration <= 0:
        raise ValueError("Output duration must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    mux_cmd = [
        "ffmpeg", "-y",
    ]
    if total_duration > video_duration:
        mux_cmd.extend(["-stream_loop", "-1"])
    mux_cmd.extend([
        "-i", str(video_only),
        "-ss", f"{audio_start:.6f}", "-t", f"{total_duration:.6f}", "-i", audio_source,
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{total_duration:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        str(output),
    ])
    subprocess.run(mux_cmd, check=True)

    manifest = {
        "output": str(output),
        "image_dir": image_dir,
        "audio_source": audio_source,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "expected_frames": expected_frames,
        "available_frames": len(indexed),
        "missing_frames": len(missing),
        "filled_frames": filled,
        "source_override_frames": source_overrides,
        "source_override_count": len(source_overrides),
        "source_ranges": source_ranges,
        "source_video_duration": video_duration,
        "audio_source_duration": audio_total_duration,
        "audio_start": audio_start,
        "output_duration": total_duration,
        "resolution": f"{width}x{height}",
        "letterbox_aspect": letterbox_aspect,
        "crop_generated_bars": crop_generated_bars,
        "generated_fit_mode": generated_fit_mode,
        "generated_crop_aspect": generated_crop_aspect,
        "crop_source_bars": crop_source_bars,
        "source_fit_mode": source_fit_mode,
        "source_black_mean_threshold": source_black_mean_threshold,
        "source_black_max_threshold": source_black_max_threshold,
        "overlay_letterbox_pixels": overlay_letterbox_pixels,
        "preset": preset,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a trailer from a numbered stylized frame sequence.")
    parser.add_argument("--preset", choices=PRESETS, default="manual", help="Reusable assembly defaults. clean_still_trailer preserves black source frames, crops/fits source fallbacks, and uses cover fit.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--audio_source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_fps", type=float, required=True)
    parser.add_argument("--output_fps", type=float, default=12.0)
    parser.add_argument("--expected_frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fill_missing", action="store_true")
    parser.add_argument("--audio_start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--source_image_dir", default=None)
    parser.add_argument("--source_index_offset", type=int, default=1)
    parser.add_argument("--source_for_missing", action="store_true")
    parser.add_argument("--source_flash_for_missing", action="store_true", help="Intentional hybrid mode: fill missing stylized frames with source frames, creating real-image flashes.")
    parser.add_argument("--source_ranges", default="", help="Comma-separated 0-based frame indices/ranges to take from source, e.g. 0-69,1207-1289.")
    parser.add_argument("--title_source_ranges", default="", help="Logo/title frame ranges to keep from source; merged into --source_ranges.")
    parser.add_argument("--tail_source_range", default="", help="End-card/tail frame range to keep from source; merged into --source_ranges.")
    parser.add_argument("--source_black_mean_threshold", type=float, default=None)
    parser.add_argument("--source_black_max_threshold", type=int, default=45)
    parser.add_argument("--letterbox_aspect", type=float, default=None, help="Burn stable black bars for this active aspect ratio, e.g. 2.39 for scope.")
    parser.add_argument("--crop_generated_bars", action="store_true", help="Crop black bars returned by the image model before fitting generated frames.")
    parser.add_argument("--generated_fit_mode", choices=["contain", "cover", "active_aspect"], default="contain", help="How generated frames are fit after optional bar cropping.")
    parser.add_argument("--generated_crop_aspect", type=float, default=None, help="Center-crop generated frames to this aspect ratio before fitting, e.g. 2.39.")
    parser.add_argument("--crop_source_bars", action="store_true", help="Crop embedded black bars from source fallback frames before fitting.")
    parser.add_argument("--source_fit_mode", choices=FIT_MODES, default="contain", help="How source fallback frames are fit.")
    parser.add_argument("--overlay_letterbox_pixels", type=int, default=None, help="Overlay fixed black bars in pixels after fitting every frame, e.g. 129.")
    args = parser.parse_args()
    preset_values = _apply_sequence_preset(
        args.preset,
        source_ranges=args.source_ranges,
        title_source_ranges=args.title_source_ranges,
        tail_source_range=args.tail_source_range,
        source_black_mean_threshold=args.source_black_mean_threshold,
        source_black_max_threshold=args.source_black_max_threshold,
        crop_source_bars=args.crop_source_bars,
        source_fit_mode=args.source_fit_mode,
        generated_fit_mode=args.generated_fit_mode,
        overlay_letterbox_pixels=args.overlay_letterbox_pixels,
    )
    manifest = build_sequence_trailer(
        args.image_dir,
        args.audio_source,
        args.output,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        expected_frames=args.expected_frames,
        width=args.width,
        height=args.height,
        fill_missing=args.fill_missing,
        audio_start=args.audio_start,
        duration=args.duration,
        source_image_dir=args.source_image_dir,
        source_index_offset=args.source_index_offset,
        source_for_missing=args.source_for_missing,
        source_flash_for_missing=args.source_flash_for_missing,
        source_ranges=preset_values["source_ranges"],
        source_black_mean_threshold=preset_values["source_black_mean_threshold"],
        source_black_max_threshold=preset_values["source_black_max_threshold"],
        letterbox_aspect=args.letterbox_aspect,
        crop_generated_bars=args.crop_generated_bars,
        generated_fit_mode=preset_values["generated_fit_mode"],
        generated_crop_aspect=args.generated_crop_aspect,
        crop_source_bars=preset_values["crop_source_bars"],
        source_fit_mode=preset_values["source_fit_mode"],
        overlay_letterbox_pixels=preset_values["overlay_letterbox_pixels"],
        preset=args.preset,
    )
    print(json.dumps({
        "output": manifest["output"],
        "available_frames": manifest["available_frames"],
        "missing_frames": manifest["missing_frames"],
        "input_fps": manifest["input_fps"],
        "output_fps": manifest["output_fps"],
    }, indent=2))


if __name__ == "__main__":
    main()
