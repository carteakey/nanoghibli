"""Gemini Batch API path for stylization.

Trade-offs vs the sync path (src/stylizer.py):
- **50% cost** at the same quality.
- **Higher rate limits** (batch has its own bucket).
- **Async**, SLO 24h but usually minutes. The CLI blocks on polling.
- **Not available for Veo** — video generation still uses the sync path.

Batch jobs survive the process that submitted them: the job name is persisted
to `{session_dir}/batch_jobs.json` so a re-invocation with the same
`--session_id` can pick up an in-flight job instead of resubmitting.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from models import FrameInfo, QuotaExceededError

REGISTRY_FILENAME = "batch_jobs.json"

# Terminal Batch API states. Mirrors Google's documented enum.
_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class BatchJobFailed(Exception):
    """Raised when a batch job reaches a non-success terminal state."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _registry_path(session_dir: str) -> str:
    return os.path.join(session_dir, REGISTRY_FILENAME)


def load_registry(session_dir: str) -> Dict[str, Dict[str, Any]]:
    path = _registry_path(session_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_registry(session_dir: str, registry: Dict[str, Dict[str, Any]]) -> None:
    path = _registry_path(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def compute_batch_key(model_id: str, stylize_items: List[Dict]) -> str:
    """Stable key for a batch: model + sorted frame paths. Re-submitting the
    same set of frames under the same model will hit the same registry entry."""
    h = hashlib.md5()
    h.update(model_id.encode())
    for item in sorted(stylize_items, key=lambda x: x["key"]):
        h.update(item["key"].encode())
        h.update(item["frame_path"].encode())
        h.update(item["prompt"].encode())
    return f"{model_id}:{h.hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def _frame_to_request(item: Dict, temperature: float, top_p: float, top_k: int) -> Dict:
    """Build one JSONL line for a single stylization request."""
    with open(item["frame_path"], "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")
    # Per the Batch API image-gen example, image generation requires
    # responseModalities=["TEXT", "IMAGE"].
    return {
        "key": item["key"],
        "request": {
            "contents": [
                {
                    "parts": [
                        {"text": item["prompt"]},
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                    ],
                    "role": "user",
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "responseModalities": ["TEXT", "IMAGE"],
            },
        },
    }


def _write_jsonl(stylize_items: List[Dict], temperature: float, top_p: float, top_k: int,
                 out_path: str) -> int:
    """Write the batch input JSONL. Returns number of requests written."""
    n = 0
    with open(out_path, "w") as f:
        for item in stylize_items:
            f.write(json.dumps(_frame_to_request(item, temperature, top_p, top_k)) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_stylize_batch(
    client: genai.Client,
    stylize_items: List[Dict],
    model_id: str,
    session_dir: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    display_name: Optional[str] = None,
) -> str:
    """Submit a stylization batch. Returns the batch job name ('batches/...').

    Uploads a JSONL file to the File API (recommended path for image requests
    per the Google doc — inline is capped at 20MB and 10 images saturate that).
    """
    os.makedirs(session_dir, exist_ok=True)
    jsonl_path = os.path.join(session_dir, "batch_input.jsonl")
    n = _write_jsonl(stylize_items, temperature, top_p, top_k, jsonl_path)
    logging.info(f"Batch: wrote {n} requests to {jsonl_path}")

    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name or f"nanoghibli-batch-{int(time.time())}",
            mime_type="jsonl",
        ),
    )
    logging.info(f"Batch: uploaded input file {uploaded.name}")

    job = client.batches.create(
        model=model_id,
        src=uploaded.name,
        config={"display_name": display_name or f"nanoghibli-batch-{int(time.time())}"},
    )
    logging.info(f"Batch: created job {job.name} (model={model_id})")
    return job.name


def get_or_submit_job(
    client: genai.Client,
    stylize_items: List[Dict],
    model_id: str,
    session_dir: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[str, bool]:
    """Return (job_name, was_resumed). If a prior registry entry exists for
    this exact batch and the job is still alive or succeeded, reuse it.
    Otherwise submit fresh."""
    registry = load_registry(session_dir)
    key = compute_batch_key(model_id, stylize_items)
    existing = registry.get(key)
    if existing and existing.get("job_name"):
        state = existing.get("state")
        if state in (None, "JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"):
            logging.info(f"Batch: resuming existing job {existing['job_name']} (state={state})")
            return existing["job_name"], True
        logging.info(
            f"Batch: registry shows prior job {existing['job_name']} terminated "
            f"as {state} — submitting a fresh job."
        )

    job_name = submit_stylize_batch(
        client, stylize_items, model_id, session_dir, temperature, top_p, top_k,
    )
    registry[key] = {
        "job_name": job_name,
        "state": "JOB_STATE_PENDING",
        "submitted_at": time.time(),
        "n_requests": len(stylize_items),
        "model_id": model_id,
    }
    save_registry(session_dir, registry)
    return job_name, False


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def poll_batch_job(
    client: genai.Client,
    job_name: str,
    session_dir: Optional[str] = None,
    poll_interval: int = 30,
    max_wait_hours: float = 24.0,
) -> Any:
    """Block until the batch reaches a terminal state. Returns the final job
    object. Updates the registry on each poll so a Ctrl-C leaves accurate state.
    On non-success terminal states raises BatchJobFailed; on quota-shaped
    errors raises QuotaExceededError so main.py's handler triggers.
    """
    deadline = time.time() + max_wait_hours * 3600
    registry_key = None
    registry: Dict[str, Dict[str, Any]] = {}
    if session_dir:
        registry = load_registry(session_dir)
        for k, v in registry.items():
            if v.get("job_name") == job_name:
                registry_key = k
                break

    while True:
        if time.time() > deadline:
            raise BatchJobFailed(f"Batch job {job_name} exceeded {max_wait_hours}h wait.")
        try:
            job = client.batches.get(name=job_name)
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "per day" in msg:
                raise QuotaExceededError(f"Quota hit while polling batch {job_name}: {e}")
            logging.warning(f"Batch poll error (will retry): {e}")
            time.sleep(poll_interval)
            continue

        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if session_dir and registry_key is not None:
            registry[registry_key]["state"] = state
            save_registry(session_dir, registry)

        if state in _TERMINAL_STATES:
            logging.info(f"Batch {job_name} reached terminal state: {state}")
            if state != "JOB_STATE_SUCCEEDED":
                err = getattr(job, "error", None)
                raise BatchJobFailed(f"Batch {job_name} ended in {state}: {err}")
            return job

        logging.info(f"Batch {job_name} state={state}; sleeping {poll_interval}s...")
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def _decode_inline_image(part: Dict) -> Optional[bytes]:
    """Extract raw image bytes from one response part. Handles both the
    snake_case (inline_data) and camelCase (inlineData) key styles that
    appear depending on how the JSONL was produced."""
    inline = part.get("inlineData") or part.get("inline_data")
    if not inline:
        return None
    data = inline.get("data")
    if not data:
        return None
    return base64.b64decode(data)


def process_batch_results(
    client: genai.Client,
    job: Any,
    stylize_items: List[Dict],
    output_dir: str,
    cache_dir: str,
    model_id: str,
) -> List[FrameInfo]:
    """Download the batch result file, parse each response, write images to
    `output_dir/stylized_{index:06d}.png` and to the content-addressable cache
    at `cache_dir/{frame_hash}_{model_slug}.png` (same naming as the sync
    stylizer so reruns can hit the cache).
    """
    from stylizer import get_file_hash  # reuse to keep cache keys aligned

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    model_slug = "pro" if "pro" in model_id else "flash"
    by_key = {item["key"]: item for item in stylize_items}

    # Results are in a JSONL file; download and parse line-by-line.
    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    if not file_name:
        raise BatchJobFailed(
            f"Batch job {getattr(job, 'name', '?')} has no dest.file_name — "
            "cannot retrieve results. (Inline-response batches not supported here.)"
        )

    logging.info(f"Batch: downloading results from {file_name}")
    raw = client.files.download(file=file_name)
    content = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw

    results: List[FrameInfo] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            logging.warning(f"Batch: skipping unparseable result line: {line[:120]}")
            continue

        key = rec.get("key")
        if key not in by_key:
            logging.warning(f"Batch: result has unknown key {key}")
            continue
        item = by_key[key]

        if rec.get("error"):
            logging.warning(f"Batch: request {key} failed: {rec['error']}")
            continue

        response = rec.get("response") or {}
        candidates = response.get("candidates") or []
        if not candidates:
            logging.warning(f"Batch: no candidates in response for {key}")
            continue
        parts = (candidates[0].get("content") or {}).get("parts") or []

        img_bytes = None
        for part in parts:
            img_bytes = _decode_inline_image(part)
            if img_bytes:
                break
        if not img_bytes:
            logging.warning(f"Batch: no inline image in response for {key}")
            continue

        orig_index = item["original_frame_index"]
        out_name = f"stylized_{orig_index:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        # Content-addressable cache (matches sync stylizer naming).
        frame_hash = get_file_hash(item["frame_path"])
        cache_path = os.path.join(cache_dir, f"{frame_hash}_{model_slug}.png")
        with open(cache_path, "wb") as f:
            f.write(img_bytes)

        results.append({"path": out_path, "original_frame_index": orig_index})

    results.sort(key=lambda x: x["original_frame_index"])
    logging.info(f"Batch: processed {len(results)}/{len(stylize_items)} results")
    return results


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def run_stylize_batch(
    client: genai.Client,
    stylize_items: List[Dict],
    model_id: str,
    output_dir: str,
    cache_dir: str,
    session_dir: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    poll_interval: int = 30,
    max_wait_hours: float = 24.0,
) -> List[FrameInfo]:
    """End-to-end: submit-or-resume → poll → process. main.py calls this once
    per stylizer model per run."""
    job_name, was_resumed = get_or_submit_job(
        client, stylize_items, model_id, session_dir, temperature, top_p, top_k,
    )
    job = poll_batch_job(
        client, job_name, session_dir=session_dir,
        poll_interval=poll_interval, max_wait_hours=max_wait_hours,
    )
    return process_batch_results(
        client, job, stylize_items, output_dir, cache_dir, model_id,
    )
