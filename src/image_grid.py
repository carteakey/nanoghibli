from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions
from google.genai import types
from PIL import Image

from model_catalog import (
    model_cache_slug,
    model_label,
    normalize_image_grid_models,
)
from models import IMAGE_OUTPUT_PRICE, TOKEN_RATES, UsageMetrics


DEFAULT_PROMPT_LIBRARY = "prompts/image_grid.yaml"
DEFAULT_OUTPUT_ROOT = "data/output/image_grid"
JUDGE_MODEL = "gemini-3.1-pro-preview"
MANUAL_MODEL_ID = "manual-chatgpt-images"


@dataclass
class GridPrompt:
    id: str
    category: str
    prompt: str
    style_target: str
    aspect_ratio: str
    image_size: str
    text_targets: List[str]
    subject_requirements: List[str]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def stable_key(*parts: str) -> str:
    h = hashlib.md5()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def load_prompt_library(path: str) -> List[GridPrompt]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults", {})
    prompts = []
    for item in raw.get("prompts", []):
        prompts.append(GridPrompt(
            id=slugify(item["id"]),
            category=item.get("category", "uncategorized"),
            prompt=item["prompt"].strip(),
            style_target=item.get("style_target", defaults.get("style_target", "")).strip(),
            aspect_ratio=str(item.get("aspect_ratio", defaults.get("aspect_ratio", "1:1"))),
            image_size=str(item.get("image_size", defaults.get("image_size", "1K"))),
            text_targets=list(item.get("text_targets", [])),
            subject_requirements=list(item.get("subject_requirements", [])),
        ))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_generation_prompt(p: GridPrompt) -> str:
    requirements = "; ".join(p.subject_requirements)
    text_targets = ", ".join(p.text_targets) if p.text_targets else "none"
    return (
        f"{p.prompt}\n\n"
        f"Visual style target: {p.style_target}.\n"
        f"Required visible elements: {requirements}.\n"
        f"Required exact readable text: {text_targets}.\n"
        "Keep the composition clear and useful for model comparison. "
        "Do not add extra text unless exact text is requested."
    )


def output_path_for(base_dir: str, prompt_id: str, model_id: str) -> str:
    return os.path.join(
        base_dir,
        "images",
        prompt_id,
        f"{model_cache_slug(model_id)}.png",
    )


def manual_path_for(base_dir: str, prompt_id: str) -> str:
    return os.path.join(
        base_dir,
        "manual_uploads",
        f"{prompt_id}__chatgpt.png",
    )


def save_first_gemini_content_image(response: Any, out_path: str) -> bool:
    for part in getattr(response, "parts", []) or []:
        if getattr(part, "thought", False):
            continue
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None:
            image = part.as_image()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            image.save(out_path)
            return True
    return False


def save_first_imagen_image(response: Any, out_path: str) -> bool:
    generated_images = getattr(response, "generated_images", None) or []
    if not generated_images:
        return False
    generated = generated_images[0]
    image = getattr(generated, "image", None)
    if image is None:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(image, "save"):
        image.save(out_path)
        return True
    image_bytes = getattr(image, "image_bytes", None)
    if image_bytes:
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return True
    return False


def generate_image(
    client: genai.Client,
    prompt: GridPrompt,
    model_id: str,
    out_path: str,
    metrics: UsageMetrics,
    max_retries: int,
) -> Dict[str, Any]:
    full_prompt = build_generation_prompt(prompt)
    if os.path.exists(out_path):
        return {"status": "cached", "path": out_path}

    for attempt in range(max_retries):
        try:
            if model_id.startswith("imagen-"):
                config_kwargs = {
                    "number_of_images": 1,
                    "aspect_ratio": prompt.aspect_ratio,
                    "output_mime_type": "image/png",
                }
                if "fast" not in model_id:
                    config_kwargs["image_size"] = prompt.image_size
                response = client.models.generate_images(
                    model=model_id,
                    prompt=full_prompt,
                    config=types.GenerateImagesConfig(**config_kwargs),
                )
                saved = save_first_imagen_image(response, out_path)
                if saved:
                    metrics.add_image(model_id)
                    return {"status": "generated", "path": out_path}
            else:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[full_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        temperature=0.7,
                        top_p=0.95,
                    ),
                )
                metrics.add_usage(response, model_id)
                saved = save_first_gemini_content_image(response, out_path)
                if saved:
                    metrics.add_image(model_id)
                    return {"status": "generated", "path": out_path}
            return {"status": "no_image", "path": ""}
        except exceptions.ResourceExhausted as e:
            wait_time = 60
            logging.warning("Rate/quota hit for %s on %s: %s", prompt.id, model_id, e)
            time.sleep(wait_time)
        except Exception as e:
            logging.warning(
                "Generation failed for %s on %s attempt %s/%s: %s",
                prompt.id,
                model_id,
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
    return {"status": "failed", "path": ""}


def extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def score_image(
    client: genai.Client,
    prompt: GridPrompt,
    model_id: str,
    image_path: str,
    metrics: UsageMetrics,
    max_retries: int,
) -> Dict[str, Any]:
    score_path = f"{os.path.splitext(image_path)[0]}.score.json"
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            return json.load(f)

    rubric = {
        "prompt_id": prompt.id,
        "model_id": model_id,
        "subject_adherence": "0-10: required objects, counts, setting, and constraints are satisfied",
        "text_rendering": "0-10: requested text is present, exact, and readable; 10 if no text was requested and no stray text appears",
        "editability": "0-10: clean composition with separable subjects and usable blank/foreground space where requested",
        "style_drift": "0-10: visual style matches the requested style target without unwanted realism or unrelated aesthetics",
        "overall": "0-10: practical winner for this prompt",
    }
    judge_prompt = (
        "Score this generated image for an internal image-model comparison. "
        "Return only valid JSON with numeric scores and a short notes string.\n\n"
        f"Original prompt: {prompt.prompt}\n"
        f"Style target: {prompt.style_target}\n"
        f"Required elements: {prompt.subject_requirements}\n"
        f"Required exact text: {prompt.text_targets}\n"
        f"Rubric: {json.dumps(rubric)}\n"
        "JSON shape: {\"subject_adherence\": 0, \"text_rendering\": 0, "
        "\"editability\": 0, \"style_drift\": 0, \"overall\": 0, \"notes\": \"...\"}"
    )

    for attempt in range(max_retries):
        try:
            image = Image.open(image_path)
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=[judge_prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            metrics.add_usage(response, JUDGE_MODEL)
            score = extract_json_object(response.text or "")
            score["judge_model"] = JUDGE_MODEL
            with open(score_path, "w") as f:
                json.dump(score, f, indent=2)
            return score
        except Exception as e:
            logging.warning(
                "Scoring failed for %s/%s attempt %s/%s: %s",
                prompt.id,
                model_id,
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
    return {
        "subject_adherence": None,
        "text_rendering": None,
        "editability": None,
        "style_drift": None,
        "overall": None,
        "notes": "scoring failed",
        "judge_model": JUDGE_MODEL,
    }


def write_manual_instructions(base_dir: str, prompts: Iterable[GridPrompt]) -> None:
    manual_dir = os.path.join(base_dir, "manual_uploads")
    os.makedirs(manual_dir, exist_ok=True)
    instructions_path = os.path.join(manual_dir, "README.md")
    with open(instructions_path, "w") as f:
        f.write("# Manual ChatGPT Image Uploads\n\n")
        f.write("Create each image from the matching prompt using your ChatGPT subscription, then save it here with the exact filename below.\n\n")
        for p in prompts:
            f.write(f"## {p.id}\n\n")
            f.write(f"Filename: `{p.id}__chatgpt.png`\n\n")
            f.write("Prompt:\n\n")
            f.write(f"{build_generation_prompt(p)}\n\n")


def build_manual_rows(base_dir: str, prompts: Iterable[GridPrompt]) -> List[Dict[str, Any]]:
    rows = []
    for p in prompts:
        path = manual_path_for(base_dir, p.id)
        rows.append({
            "prompt_id": p.id,
            "category": p.category,
            "model_id": MANUAL_MODEL_ID,
            "model_label": "CHATGPT-MANUAL",
            "status": "ready" if os.path.exists(path) else "pending_manual_upload",
            "image_path": path if os.path.exists(path) else "",
            "estimated_cost_usd": 0.0,
        })
    return rows


def estimated_image_cost(model_id: str) -> float:
    return IMAGE_OUTPUT_PRICE.get(model_id, 0.0)


def write_manifest(base_dir: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(base_dir, exist_ok=True)
    jsonl_path = os.path.join(base_dir, "manifest.jsonl")
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = os.path.join(base_dir, "leaderboard.csv")
    fieldnames = [
        "prompt_id", "category", "model_label", "model_id", "status",
        "subject_adherence", "text_rendering", "editability", "style_drift",
        "overall", "estimated_cost_usd", "image_path", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_html_report(base_dir: str, rows: List[Dict[str, Any]]) -> None:
    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    html = [
        "<!doctype html><meta charset='utf-8'><title>NanoGhibli Image Grid</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;background:#f7f5ef;color:#222}"
        "h1{font-size:24px}h2{margin-top:32px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}"
        ".card{background:white;border:1px solid #ddd;border-radius:8px;padding:10px}.card img{width:100%;aspect-ratio:1/1;object-fit:cover;background:#eee}"
        ".meta{font-size:12px;line-height:1.4}.score{font-weight:600}</style>",
        "<h1>NanoGhibli Image Model Comparison</h1>",
    ]
    for prompt_id, prompt_rows in by_prompt.items():
        html.append(f"<h2>{prompt_id}</h2><div class='grid'>")
        for row in prompt_rows:
            img = row.get("image_path", "")
            rel = os.path.relpath(img, base_dir) if img else ""
            html.append("<div class='card'>")
            if rel and os.path.exists(img):
                html.append(f"<img src='{rel}' alt='{row['model_label']}'>")
            else:
                html.append("<div style='aspect-ratio:1/1;background:#eee;display:grid;place-items:center'>pending</div>")
            html.append(
                f"<div class='meta'><div><b>{row['model_label']}</b></div>"
                f"<div class='score'>overall: {row.get('overall', '')}</div>"
                f"<div>subject: {row.get('subject_adherence', '')} text: {row.get('text_rendering', '')}</div>"
                f"<div>edit: {row.get('editability', '')} style: {row.get('style_drift', '')}</div>"
                f"<div>cost: ${float(row.get('estimated_cost_usd') or 0):.3f}</div>"
                f"<div>{row.get('notes', '')}</div></div>"
            )
            html.append("</div>")
        html.append("</div>")
    with open(os.path.join(base_dir, "report.html"), "w") as f:
        f.write("\n".join(html))


def run_grid(args: argparse.Namespace) -> None:
    load_dotenv()
    prompts = load_prompt_library(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    models = normalize_image_grid_models([args.models] if args.models else [])
    if args.include_manual and MANUAL_MODEL_ID not in models:
        pass

    session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, session_id)
    os.makedirs(base_dir, exist_ok=True)
    write_manual_instructions(base_dir, prompts)

    client: Optional[genai.Client] = None
    needs_api = not args.prepare_only and not (args.score_existing and args.skip_score)
    if needs_api and (not os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY is required unless --prepare_only is used.")
    if needs_api:
        client = genai.Client()

    metrics = UsageMetrics(model_tier="image_grid")
    rows: List[Dict[str, Any]] = []
    for p in prompts:
        for model_id in models:
            out_path = output_path_for(base_dir, p.id, model_id)
            row = {
                "prompt_id": p.id,
                "category": p.category,
                "model_id": model_id,
                "model_label": model_label(model_id),
                "status": "prepared",
                "image_path": out_path,
                "estimated_cost_usd": estimated_image_cost(model_id),
            }
            if args.score_existing:
                row["status"] = "ready" if os.path.exists(out_path) else "missing"
                row["image_path"] = out_path if os.path.exists(out_path) else ""
                if row["image_path"] and not args.skip_score:
                    assert client is not None
                    score = score_image(client, p, model_id, row["image_path"], metrics, args.max_retries)
                    row.update(score)
            elif not args.prepare_only:
                assert client is not None
                logging.info("Generating %s with %s", p.id, model_id)
                result = generate_image(client, p, model_id, out_path, metrics, args.max_retries)
                row["status"] = result["status"]
                row["image_path"] = result.get("path") or ""
                if row["image_path"] and not args.skip_score:
                    score = score_image(client, p, model_id, row["image_path"], metrics, args.max_retries)
                    row.update(score)
            rows.append(row)

    if args.include_manual:
        manual_rows = build_manual_rows(base_dir, prompts)
        if not args.skip_score and not args.prepare_only:
            assert client is not None
            prompt_by_id = {p.id: p for p in prompts}
            for row in manual_rows:
                if row["status"] == "ready":
                    score = score_image(
                        client,
                        prompt_by_id[row["prompt_id"]],
                        MANUAL_MODEL_ID,
                        row["image_path"],
                        metrics,
                        args.max_retries,
                    )
                    row.update(score)
        rows.extend(manual_rows)

    write_manifest(base_dir, rows)
    write_html_report(base_dir, rows)
    shutil.copy(args.prompts, os.path.join(base_dir, "prompt_library.yaml"))
    print(metrics)
    print(f"Report: {os.path.join(base_dir, 'report.html')}")
    print(f"Leaderboard: {os.path.join(base_dir, 'leaderboard.csv')}")
    print(f"Manual upload instructions: {os.path.join(base_dir, 'manual_uploads', 'README.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an image model comparison grid.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPT_LIBRARY, help="YAML prompt library.")
    parser.add_argument("--models", default=None, help="Comma-separated models. Defaults to Nano Banana Pro, Nano Banana 2, Imagen 4 Fast/Standard/Ultra.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N prompts.")
    parser.add_argument("--prepare_only", action="store_true", help="Create report shell and manual upload instructions without API calls.")
    parser.add_argument("--score_existing", action="store_true", help="Do not generate new images; rebuild reports and optionally score files already on disk.")
    parser.add_argument("--skip_score", action="store_true", help="Generate images without judge scoring.")
    parser.add_argument("--include_manual", action="store_true", default=True, help="Include manual ChatGPT image placeholders.")
    parser.add_argument("--no_manual", dest="include_manual", action="store_false", help="Do not include manual ChatGPT placeholders.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        run_grid(args)
    except Exception as e:
        logging.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
