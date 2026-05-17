from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List


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


def collect_images(input_dir: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    return sorted(images)


def build_stills_trailer(
    image_dir: str,
    audio_source: str,
    output_path: str,
    fps: float = 12.0,
    width: int = 1280,
    height: int = 720,
    crossfade: float = 0.0,
) -> dict:
    images = collect_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    total_duration = media_duration(audio_source)
    output = Path(output_path)
    work_dir = output.parent / f"{output.stem}_segments"
    work_dir.mkdir(parents=True, exist_ok=True)

    if crossfade < 0:
        raise ValueError("--crossfade must be >= 0")
    if len(images) == 1:
        crossfade = 0.0

    segment_duration = total_duration / len(images)
    if crossfade >= segment_duration:
        raise ValueError(
            f"--crossfade ({crossfade}) must be shorter than each segment ({segment_duration:.3f}s)"
        )

    segments = []
    for i, image_path in enumerate(images):
        audio_start = i * segment_duration
        audio_end = total_duration if i == len(images) - 1 else (i + 1) * segment_duration
        if crossfade:
            start = max(0.0, audio_start - (crossfade / 2 if i > 0 else 0.0))
            end = min(total_duration, audio_end + (crossfade / 2 if i < len(images) - 1 else 0.0))
        else:
            start = audio_start
            end = audio_end
        duration = end - start
        segment_path = work_dir / f"seg_{i:03d}.mp4"
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},format=yuv420p"
        )
        subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps), "-t", f"{duration:.6f}",
            "-i", str(image_path),
            "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", audio_source,
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest",
            str(segment_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segments.append({
            "index": i,
            "start": start,
            "end": end,
            "image": str(image_path),
            "segment": str(segment_path),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    if crossfade:
        video_labels = []
        audio_labels = []
        cmd = ["ffmpeg", "-y"]
        for segment in segments:
            cmd.extend(["-i", segment["segment"]])
        filter_parts = []
        for i in range(len(segments)):
            filter_parts.append(
                f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]"
            )
            video_labels.append(f"v{i}")
            audio_labels.append(f"a{i}")

        current_v = video_labels[0]
        current_a = audio_labels[0]
        accumulated = float(segments[0]["end"] - segments[0]["start"])
        for i in range(1, len(segments)):
            offset = max(0.0, accumulated - crossfade)
            next_v = f"vx{i}"
            next_a = f"ax{i}"
            filter_parts.append(
                f"[{current_v}][v{i}]xfade=transition=fade:duration={crossfade:.6f}:offset={offset:.6f}[{next_v}]"
            )
            filter_parts.append(
                f"[{current_a}][a{i}]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[{next_a}]"
            )
            current_v = next_v
            current_a = next_a
            accumulated += float(segments[i]["end"] - segments[i]["start"]) - crossfade

        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", f"[{current_v}]", "-map", f"[{current_a}]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac",
            str(output),
        ])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        concat_path = work_dir / "concat_list.txt"
        with concat_path.open("w") as f:
            for segment in segments:
                f.write(f"file '{Path(segment['segment']).resolve()}'\n")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-c", "copy",
            str(output),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    manifest = {
        "output": str(output),
        "image_dir": image_dir,
        "audio_source": audio_source,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "crossfade": crossfade,
        "duration": total_duration,
        "segments": segments,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stills-only trailer from stylized images and source audio.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--audio_source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crossfade", type=float, default=0.0, help="Seconds of xfade/acrossfade between adjacent still anchors.")
    args = parser.parse_args()
    manifest = build_stills_trailer(
        args.image_dir,
        args.audio_source,
        args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        crossfade=args.crossfade,
    )
    print(json.dumps({
        "output": manifest["output"],
        "duration": manifest["duration"],
        "frames": len(manifest["segments"]),
        "fps": manifest["fps"],
    }, indent=2))


if __name__ == "__main__":
    main()
