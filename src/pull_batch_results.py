import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from batch_stylizer import poll_batch_job, process_batch_results
from stylizer import GHIBLI_PROMPT


def _image_paths(input_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        p for p in sorted(Path(input_dir).iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poll and pull a known Gemini Batch image-stylization job."
    )
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--max_wait_hours", type=float, default=24.0)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    client = genai.Client()

    frames = _image_paths(args.input_dir)
    if not frames:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        return 2

    session_dir = Path(args.session_dir)
    output_dir = session_dir / "stylized_frames"
    cache_dir = Path("data/cache/stylized")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    items = [
        {
            "key": f"frame_{i:06d}",
            "frame_path": str(path),
            "original_frame_index": i,
            "prompt": GHIBLI_PROMPT,
            "scene_description": "",
        }
        for i, path in enumerate(frames)
    ]

    if args.poll:
        job = poll_batch_job(
            client,
            args.job_name,
            session_dir=args.session_dir,
            poll_interval=args.poll_interval,
            max_wait_hours=args.max_wait_hours,
        )
    else:
        job = client.batches.get(name=args.job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state != "JOB_STATE_SUCCEEDED":
            print(f"{args.job_name} is {state}; not pulling yet.")
            return 1

    results = process_batch_results(
        client,
        job,
        items,
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        model_id=args.model_id,
    )
    print(f"Pulled {len(results)}/{len(items)} results into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
