from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import batch_stylizer
from model_catalog import normalize_stylizer_model
from stylizer import GHIBLI_PROMPT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(input_dir: str):
    return [
        path for path in sorted(Path(input_dir).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit still frames to Gemini Batch image stylization.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--model", default="nano-banana-2")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    model_id = normalize_stylizer_model(args.model)
    frames = image_paths(args.input_dir)
    if not frames:
        raise ValueError(f"No image frames found in {args.input_dir}")

    os.makedirs(args.session_dir, exist_ok=True)
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

    client = genai.Client()
    job_name, resumed = batch_stylizer.get_or_submit_job(
        client,
        items,
        model_id,
        args.session_dir,
        args.temperature,
        args.top_p,
        args.top_k,
    )
    result = {
        "job_name": job_name,
        "resumed": resumed,
        "model_id": model_id,
        "n_requests": len(items),
        "input_dir": args.input_dir,
        "session_dir": args.session_dir,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
