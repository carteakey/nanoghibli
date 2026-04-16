import os
import shutil
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.api_core import exceptions
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

from models import FrameInfo, UsageMetrics, QuotaExceededError

GHIBLI_PROMPT = (
    "Modify this image into a Studio Ghibli homage. "
    "Keep the exact same structural composition, objects, and subjects, but render it in a beautiful, "
    "hand-drawn anime style reminiscent of classic 90s animation. "
    "Use lush, vibrant watercolor backgrounds, distinct clean character/object outlines, and soft, magical lighting. "
    "This is strictly an homage to visualize how things would look in that style. "
    "CRITICAL: You must follow the source image exactly. Do not hallucinate, invent, or add any new elements, objects, text, or details that are not explicitly present in the original image. Do not fill in any blanks."
)

def get_file_hash(file_path: str) -> str:
    """Returns MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_scene_description(client: genai.Client, frame_path: str, metrics: UsageMetrics = None) -> str:
    """
    Asks Gemini to briefly describe the scene to use as a context prompt.
    """
    try:
        image = Image.open(frame_path)
        prompt = "Describe the main subjects, setting, and lighting in this image in one short sentence (max 15 words). Focus on what is visible."
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, image]
        )
        if metrics:
            metrics.add_usage(response, "gemini-3.1-flash-lite-preview")
            metrics.add_description()
        description = response.text.strip()
        logging.info(f"Generated scene description: {description}")
        return description
    except Exception as e:
        logging.warning(f"Failed to generate scene description: {e}")
        return ""

def process_single_frame(client: genai.Client, frame_info: FrameInfo, output_dir: str, model_id: str = "gemini-3.1-flash-image-preview", cache_dir: str = "data/cache/stylized", max_retries: int = 5, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, scene_description: str = "", metrics: UsageMetrics = None) -> Optional[FrameInfo]:
    input_path = frame_info["path"]
    orig_index = frame_info.get("original_frame_index", 0)
    
    # 1. Check Global Cache
    # Append model_id type to hash to ensure different models have different cache entries
    frame_hash = get_file_hash(input_path)
    model_slug = "pro" if "pro" in model_id else "flash"
    cache_path = os.path.join(cache_dir, f"{frame_hash}_{model_slug}.png")
    out_name = f"stylized_{orig_index:06d}.png"
    out_path = os.path.join(output_dir, out_name)
    
    if os.path.exists(cache_path):
        if not os.path.exists(out_path):
            shutil.copy(cache_path, out_path)
        return {
            "path": out_path,
            "original_frame_index": orig_index
        }
        
    if os.path.exists(out_path):
        return {
            "path": out_path,
            "original_frame_index": orig_index
        }
        
    # Build smart prompt
    context_prefix = f"The scene contains: {scene_description}. " if scene_description else ""
    full_prompt = f"{context_prefix}{GHIBLI_PROMPT}"

    for attempt in range(max_retries):
        try:
            image = Image.open(input_path)
            response = client.models.generate_content(
                model=model_id,
                contents=[full_prompt, image],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    http_options={'timeout': 60000}, # 60 seconds timeout
                )
            )
            if metrics:
                metrics.add_usage(response, model_id)
                metrics.add_image(model_id)
            
            for part in response.parts:
                if getattr(part, 'thought', False):
                    continue
                if part.inline_data is not None:
                    out_img = part.as_image()
                    out_img.save(out_path)
                    # Save to cache for future runs
                    if not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    out_img.save(cache_path)
                    
                    return {
                        "path": out_path,
                        "original_frame_index": orig_index
                    }
                    
            logging.warning(f"No image returned for {input_path} on attempt {attempt+1}")
        except exceptions.ResourceExhausted as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "per day" in error_msg:
                logging.error(f"Daily quota hit for Stylizer: {e}. Exiting so you can resume later.")
                qe = QuotaExceededError("Stylizer daily quota exceeded.")
                if metrics is not None:
                    qe.metrics_snapshot = metrics
                raise qe
            
            wait_time = 60 # Handle RPM limits
            logging.warning(f"Rate limit hit for {input_path} (RPM). Waiting {wait_time}s... Error: {e}")
            time.sleep(wait_time)
        except Exception as e:
            logging.warning(f"Error stylizing {input_path} on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt)) # Standard exponential backoff
            
    logging.error(f"Failed to stylize {input_path} after {max_retries} attempts.")
    return None

def stylize_frames(frames: List[FrameInfo], output_dir: str, model_id: str = "gemini-3.1-flash-image-preview", cache_dir: str = "data/cache/stylized", max_workers: int = 4, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, scene_description: str = "", metrics: UsageMetrics = None) -> List[FrameInfo]:
    """
    Takes a list of frame dicts and stylizes each concurrently using the Gemini API.
    Uses scene_description to maintain consistency.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    client = genai.Client()
    stylized_frames: List[FrameInfo] = []
    failed = 0
    quota_hit: Optional[QuotaExceededError] = None

    logging.info(f"Stylizing {len(frames)} frames with model {model_id} and description: '{scene_description}'")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_frame, client, f, output_dir, model_id, cache_dir,
                5, temperature, top_p, top_k, scene_description, metrics,
            ): f for f in frames
        }
        try:
            for future in tqdm(as_completed(futures), total=len(frames)):
                try:
                    result = future.result()
                except QuotaExceededError as e:
                    # Daily quota — stop eagerly so siblings don't burn calls.
                    quota_hit = e
                    for pending in futures:
                        pending.cancel()
                    break
                if result:
                    stylized_frames.append(result)
                else:
                    failed += 1
        finally:
            # cancel_futures requires Python 3.9+; guard for older runtimes.
            try:
                executor.shutdown(wait=True, cancel_futures=(quota_hit is not None))
            except TypeError:
                executor.shutdown(wait=True)

    if quota_hit is not None:
        raise quota_hit

    if failed:
        logging.warning(
            f"Stylization: {failed}/{len(frames)} frames failed for this scene "
            f"(model={model_id}). Continuing with {len(stylized_frames)} frames."
        )

    stylized_frames.sort(key=lambda x: x["original_frame_index"])
    return stylized_frames
