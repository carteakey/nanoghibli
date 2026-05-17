import hashlib
import re
from typing import Dict, Iterable, List


STYLIZER_ALIASES: Dict[str, str] = {
    "nano-banana": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "flash": "gemini-3.1-flash-image-preview",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    "pro": "gemini-3-pro-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "pro-2k": "gemini-3-pro-image-preview-2k",
    "nano-banana-pro-2k": "gemini-3-pro-image-preview-2k",
    "gemini-3-pro-image-preview-2k": "gemini-3-pro-image-preview-2k",
    "gpt-image-1.5": "gpt-image-1.5",
    "gpt-image-1.5-high-fidelity": "gpt-image-1.5",
    "chatgpt": "manual-chatgpt-images",
    "chatgpt-manual": "manual-chatgpt-images",
    "manual-chatgpt": "manual-chatgpt-images",
    "manual-chatgpt-images": "manual-chatgpt-images",
}

IMAGE_GRID_ALIASES: Dict[str, str] = {
    "nano-banana": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "flash-image": "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "pro-image": "gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "imagen-4-fast": "imagen-4.0-fast-generate-001",
    "imagen-fast": "imagen-4.0-fast-generate-001",
    "imagen-4.0-fast-generate-001": "imagen-4.0-fast-generate-001",
    "imagen-4": "imagen-4.0-generate-001",
    "imagen-4-standard": "imagen-4.0-generate-001",
    "imagen-standard": "imagen-4.0-generate-001",
    "imagen-4.0-generate-001": "imagen-4.0-generate-001",
    "imagen-4-ultra": "imagen-4.0-ultra-generate-001",
    "imagen-ultra": "imagen-4.0-ultra-generate-001",
    "imagen-4.0-ultra-generate-001": "imagen-4.0-ultra-generate-001",
}


def normalize_stylizer_model(value: str) -> str:
    key = value.strip().lower()
    if key in STYLIZER_ALIASES:
        return STYLIZER_ALIASES[key]
    raise ValueError(f"Unknown stylizer model: {value}")


def normalize_stylizer_models(values: Iterable[str]) -> List[str]:
    models = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                models.append(normalize_stylizer_model(part))
    if not models:
        models.append(STYLIZER_ALIASES["flash"])
    return models


def normalize_image_grid_model(value: str) -> str:
    key = value.strip().lower()
    if key in IMAGE_GRID_ALIASES:
        return IMAGE_GRID_ALIASES[key]
    raise ValueError(f"Unknown image grid model: {value}")


def normalize_image_grid_models(values: Iterable[str]) -> List[str]:
    models = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                models.append(normalize_image_grid_model(part))
    if not models:
        models = [
            IMAGE_GRID_ALIASES["nano-banana-pro"],
            IMAGE_GRID_ALIASES["nano-banana-2"],
            IMAGE_GRID_ALIASES["imagen-4-fast"],
            IMAGE_GRID_ALIASES["imagen-4-standard"],
            IMAGE_GRID_ALIASES["imagen-4-ultra"],
        ]
    return models


def model_cache_slug(model_id: str) -> str:
    if model_id == "gemini-2.5-flash-image":
        return "nano_banana"
    if model_id == "gemini-3.1-flash-image-preview":
        return "flash"
    if model_id == "gemini-3-pro-image-preview":
        return "pro"
    if model_id == "gemini-3-pro-image-preview-2k":
        return "pro_2k"
    if model_id == "gpt-image-1.5":
        return "gpt_image_1_5_high"
    if model_id == "manual-chatgpt-images":
        return "chatgpt_manual"
    if model_id == "imagen-4.0-fast-generate-001":
        return "imagen_4_fast"
    if model_id == "imagen-4.0-generate-001":
        return "imagen_4_standard"
    if model_id == "imagen-4.0-ultra-generate-001":
        return "imagen_4_ultra"
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def model_label(model_id: str) -> str:
    if model_id == "gemini-2.5-flash-image":
        return "NANO-BANANA"
    if model_id == "gemini-3.1-flash-image-preview":
        return "FLASH"
    if model_id == "gemini-3-pro-image-preview":
        return "PRO"
    if model_id == "gemini-3-pro-image-preview-2k":
        return "PRO-2K"
    if model_id == "gpt-image-1.5":
        return "GPT-IMAGE-1.5-HIGH"
    if model_id == "manual-chatgpt-images":
        return "CHATGPT-MANUAL"
    if model_id == "imagen-4.0-fast-generate-001":
        return "IMAGEN-4-FAST"
    if model_id == "imagen-4.0-generate-001":
        return "IMAGEN-4-STANDARD"
    if model_id == "imagen-4.0-ultra-generate-001":
        return "IMAGEN-4-ULTRA"
    return model_cache_slug(model_id).upper()


def model_provider(model_id: str) -> str:
    if model_id.startswith("gemini-"):
        return "google"
    if model_id.startswith("imagen-"):
        return "google"
    if model_id.startswith("gpt-image-"):
        return "openai"
    if model_id.startswith("manual-"):
        return "manual"
    return "unknown"


def rotation_signature(model_ids: List[str]) -> str:
    if len(model_ids) == 1:
        return model_cache_slug(model_ids[0])
    digest = hashlib.md5(",".join(model_ids).encode()).hexdigest()[:8]
    return f"rot_{digest}"
