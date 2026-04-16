import logging
import threading
from typing import TypedDict, List, Dict, Tuple

# Pricing reference: api-pricing.md (Google Gemini API, paid tier, standard pricing,
# prompts <= 200k tokens). Update alongside that doc.

# (input_per_1M, output_per_1M). Output price already includes thinking tokens
# per Google's pricing page.
TOKEN_RATES: Dict[str, Tuple[float, float]] = {
    "gemini-3.1-flash-lite-preview":  (0.25, 1.50),
    "gemini-3.1-flash-image-preview": (0.50, 3.00),
    "gemini-3-pro-image-preview":     (2.00, 12.00),
    "gemini-3-flash-preview":         (0.50, 3.00),
}

# Per-image output price at 1K or 2K resolution. The pipeline caps frames at
# 1920x1080 (extractor.py), so 1K/2K tier applies; 4K is not reachable.
IMAGE_OUTPUT_PRICE: Dict[str, float] = {
    "gemini-3.1-flash-image-preview": 0.067,
    "gemini-3-pro-image-preview":     0.134,
}

# Veo: per-second price at 720p/1080p ("Standard" resolution in the pricing doc).
VEO_RATE_PER_SEC: Dict[str, float] = {
    "veo-3.1-fast-generate-preview": 0.15,
    "veo-3.1-generate-preview":      0.40,
    "veo-3.0-fast-generate-001":     0.15,
    "veo-3.0-generate-001":          0.40,
    "veo-2.0-generate-001":          0.35,
}

# Gemini Batch API bills at 50% of interactive rates. Apply to both token and
# image costs for any call routed through client.batches.*.
BATCH_MULT: float = 0.5
_BATCH_SUFFIX: str = ":batch"


def _bucket_key(model_id: str, is_batch: bool) -> str:
    """Return the dict key we store costs under. Batch calls get a ':batch'
    suffix so sync and batch costs stay separated per model."""
    return f"{model_id}{_BATCH_SUFFIX}" if is_batch else model_id


def _split_key(key: str) -> Tuple[str, bool]:
    """Inverse of _bucket_key. Returns (base_model_id, is_batch)."""
    if key.endswith(_BATCH_SUFFIX):
        return key[: -len(_BATCH_SUFFIX)], True
    return key, False


class UsageMetrics:
    def __init__(self, model_tier: str = "flash"):
        self.model_tier = model_tier
        self._lock = threading.Lock()
        self.tokens: Dict[str, Dict[str, int]] = {}
        self.images_by_model: Dict[str, int] = {}
        self.videos_by_model: Dict[str, Dict[str, float]] = {}
        self.descriptions_generated = 0

    def add_usage(self, response, model_id: str, is_batch: bool = False):
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            logging.debug(f"No usage_metadata on response from {model_id}; tokens not counted.")
            return
        prompt = getattr(meta, "prompt_token_count", 0) or 0
        candidates = getattr(meta, "candidates_token_count", 0) or 0
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        cached = getattr(meta, "cached_content_token_count", 0) or 0
        key = _bucket_key(model_id, is_batch)
        with self._lock:
            bucket = self.tokens.setdefault(
                key, {"input": 0, "output": 0, "thoughts": 0, "cached": 0}
            )
            bucket["input"] += prompt
            bucket["output"] += candidates
            bucket["thoughts"] += thoughts
            bucket["cached"] += cached

    def add_image(self, model_id: str, is_batch: bool = False):
        key = _bucket_key(model_id, is_batch)
        with self._lock:
            self.images_by_model[key] = self.images_by_model.get(key, 0) + 1

    def add_description(self):
        with self._lock:
            self.descriptions_generated += 1

    def add_video(self, model_id: str, seconds: float, is_batch: bool = False):
        # Veo is not supported by the Batch API today, so is_batch is accepted
        # for API symmetry but should never be True in practice.
        key = _bucket_key(model_id, is_batch)
        with self._lock:
            bucket = self.videos_by_model.setdefault(key, {"count": 0, "seconds": 0.0})
            bucket["count"] += 1
            bucket["seconds"] += seconds

    @property
    def total_images(self) -> int:
        return sum(self.images_by_model.values())

    @property
    def total_video_seconds(self) -> float:
        return sum(b["seconds"] for b in self.videos_by_model.values())

    @property
    def total_videos(self) -> int:
        return int(sum(b["count"] for b in self.videos_by_model.values()))

    @property
    def total_input_tokens(self) -> int:
        return sum(b["input"] for b in self.tokens.values())

    @property
    def total_output_tokens(self) -> int:
        # Thinking is billed at the output rate; report them together.
        return sum(b["output"] + b["thoughts"] for b in self.tokens.values())

    def _compute_cost(self) -> float:
        total = 0.0
        for key, bucket in self.tokens.items():
            base, is_batch = _split_key(key)
            rates = TOKEN_RATES.get(base)
            if rates is None:
                logging.debug(f"No token rate known for {key}; excluded from cost.")
                continue
            in_rate, out_rate = rates
            mult = BATCH_MULT if is_batch else 1.0
            total += (bucket["input"] / 1_000_000 * in_rate) * mult
            total += ((bucket["output"] + bucket["thoughts"]) / 1_000_000 * out_rate) * mult
        for key, n in self.images_by_model.items():
            base, is_batch = _split_key(key)
            rate = IMAGE_OUTPUT_PRICE.get(base)
            if rate is None:
                logging.debug(f"No image rate known for {key}; excluded from cost.")
                continue
            mult = BATCH_MULT if is_batch else 1.0
            total += n * rate * mult
        for key, bucket in self.videos_by_model.items():
            base, is_batch = _split_key(key)
            rate = VEO_RATE_PER_SEC.get(base)
            if rate is None:
                logging.debug(f"No video rate known for {key}; excluded from cost.")
                continue
            mult = BATCH_MULT if is_batch else 1.0
            total += bucket["seconds"] * rate * mult
        return total

    def __str__(self):
        total_cost = self._compute_cost()
        lines = [
            "",
            "=" * 44,
            f"   ESTIMATED SESSION COST ({self.model_tier.upper()})",
            "=" * 44,
            f"Descriptions generated : {self.descriptions_generated}",
            f"Images stylized        : {self.total_images}",
            f"Veo videos             : {self.total_videos} ({self.total_video_seconds:.1f}s)",
        ]
        if self.tokens:
            lines.append("-" * 44)
            lines.append("Tokens by model:")
            for key, bucket in self.tokens.items():
                base, is_batch = _split_key(key)
                tag = " [batch 50%]" if is_batch else ""
                line = f"  {base}{tag}: {bucket['input']} in / {bucket['output']} out"
                if bucket["thoughts"]:
                    line += f" (+{bucket['thoughts']} thought)"
                if bucket["cached"]:
                    line += f" [{bucket['cached']} cached]"
                lines.append(line)
        lines.append("-" * 44)
        lines.append(f"Total Est. Cost: ${total_cost:.4f}")
        lines.append("=" * 44)
        return "\n".join(lines)


class FrameInfo(TypedDict):
    path: str
    original_frame_index: int


class QuotaExceededError(Exception):
    pass


class Scene(TypedDict, total=False):
    scene_index: int
    start_frame: int
    end_frame: int
    frames: List[FrameInfo]
    stylized_frames: List[FrameInfo]
    description: str
