"""Manual ChatGPT Images fallback for stylized frames.

This provider does not call an API. It writes a per-session queue of source
frames and prompts that can be handled in ChatGPT Images, then imports completed
PNG files on the next run.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Dict, List

from model_catalog import model_cache_slug
from models import FrameInfo
from stylizer import get_file_hash

MANUAL_MODEL_ID = "manual-chatgpt-images"
MANIFEST_FILENAME = "chatgpt_manual_queue.json"


def _manual_dir(session_dir: str) -> str:
    return os.path.join(session_dir, "manual_chatgpt")


def manual_upload_path(session_dir: str, original_frame_index: int) -> str:
    return os.path.join(
        _manual_dir(session_dir),
        "uploads",
        f"stylized_{original_frame_index:06d}.png",
    )


def _load_manifest(session_dir: str) -> Dict[str, Dict]:
    path = os.path.join(_manual_dir(session_dir), MANIFEST_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {str(item["original_frame_index"]): item for item in data.get("items", [])}


def _save_manifest(session_dir: str, items_by_index: Dict[str, Dict]) -> None:
    manual_dir = _manual_dir(session_dir)
    os.makedirs(os.path.join(manual_dir, "uploads"), exist_ok=True)
    path = os.path.join(manual_dir, MANIFEST_FILENAME)
    payload = {"model_id": MANUAL_MODEL_ID, "items": list(items_by_index.values())}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_readme(session_dir: str, items: List[Dict]) -> None:
    manual_dir = _manual_dir(session_dir)
    os.makedirs(os.path.join(manual_dir, "uploads"), exist_ok=True)
    readme_path = os.path.join(manual_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Manual ChatGPT Image Fallback\n\n")
        f.write(
            "Use this queue when API image buckets are constrained. For each item, "
            "open the source frame in ChatGPT Images, apply the prompt, and save "
            "the result to the exact upload path. Rerun the same NanoGhibli "
            "session and completed uploads will be imported automatically.\n\n"
        )
        for item in sorted(items, key=lambda x: x["original_frame_index"]):
            f.write(f"## Frame {item['original_frame_index']:06d}\n\n")
            f.write(f"Source: `{item['source_path']}`\n\n")
            f.write(f"Upload result: `{item['upload_path']}`\n\n")
            f.write("Prompt:\n\n")
            f.write(item["prompt"])
            f.write("\n\n")


def queue_manual_frames(
    frames: List[FrameInfo],
    prompt: str,
    session_dir: str,
    output_dir: str,
    cache_dir: str,
) -> List[FrameInfo]:
    """Queue frames for manual ChatGPT Images processing and import any
    completed uploads. Returns only frames whose upload exists and has been
    copied into the normal stylized output/cache locations.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    existing = _load_manifest(session_dir)
    queued_items: Dict[str, Dict] = dict(existing)
    ready: List[FrameInfo] = []
    model_slug = model_cache_slug(MANUAL_MODEL_ID)

    for frame in frames:
        orig_index = frame["original_frame_index"]
        key = str(orig_index)
        out_path = os.path.join(output_dir, f"stylized_{orig_index:06d}.png")
        upload_path = manual_upload_path(session_dir, orig_index)

        queued_items[key] = {
            "original_frame_index": orig_index,
            "source_path": frame["path"],
            "upload_path": upload_path,
            "output_path": out_path,
            "prompt": prompt,
        }

        if os.path.exists(out_path):
            ready.append({"path": out_path, "original_frame_index": orig_index})
            continue

        if not os.path.exists(upload_path):
            continue

        shutil.copy(upload_path, out_path)
        frame_hash = get_file_hash(frame["path"])
        cache_path = os.path.join(cache_dir, f"{frame_hash}_{model_slug}.png")
        shutil.copy(upload_path, cache_path)
        ready.append({"path": out_path, "original_frame_index": orig_index})

    _save_manifest(session_dir, queued_items)
    _write_readme(session_dir, list(queued_items.values()))
    pending = len(frames) - len(ready)
    if pending:
        logging.warning(
            "Queued %s frame(s) for manual ChatGPT Images fallback at %s",
            pending,
            os.path.join(_manual_dir(session_dir), "README.md"),
        )
    return sorted(ready, key=lambda x: x["original_frame_index"])
