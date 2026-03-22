import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

from models import FrameInfo

GHIBLI_PROMPT = (
    "Modify this image into a Studio Ghibli homage. "
    "Keep the exact same structural composition, objects, and subjects, but render it in a beautiful, "
    "hand-drawn anime style reminiscent of classic 90s animation. "
    "Use lush, vibrant watercolor backgrounds, distinct clean character/object outlines, and soft, magical lighting. "
    "This is strictly an homage to visualize how things would look in that style. "
    "CRITICAL: You must follow the source image exactly. Do not hallucinate, invent, or add any new elements, objects, text, or details that are not explicitly present in the original image. Do not fill in any blanks."
)

def process_single_frame(client: genai.Client, frame_info: FrameInfo, output_dir: str, max_retries: int = 3) -> Optional[FrameInfo]:
    input_path = frame_info["path"]
    orig_index = frame_info.get("original_frame_index", 0)
    out_name = f"stylized_{orig_index:06d}.png"
    out_path = os.path.join(output_dir, out_name)
    
    if os.path.exists(out_path):
        return {
            "path": out_path,
            "original_frame_index": orig_index
        }
        
    for attempt in range(max_retries):
        try:
            image = Image.open(input_path)
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[GHIBLI_PROMPT, image],
            )
            
            for part in response.parts:
                if getattr(part, 'thought', False):
                    continue
                if part.inline_data is not None:
                    out_img = part.as_image()
                    out_img.save(out_path)
                    return {
                        "path": out_path,
                        "original_frame_index": orig_index
                    }
                    
            logging.warning(f"No image returned for {input_path} on attempt {attempt+1}")
        except Exception as e:
            logging.warning(f"Error stylizing {input_path} on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            
    logging.error(f"Failed to stylize {input_path} after {max_retries} attempts.")
    return None

def stylize_frames(frames: List[FrameInfo], output_dir: str, max_workers: int = 4) -> List[FrameInfo]:
    """
    Takes a list of frame dicts (e.g. [{"path": ..., "original_frame_index": ...}])
    and stylizes each concurrently using the Gemini API.
    Saves outputs in output_dir.
    Returns a list of stylized frame dicts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = genai.Client()
    stylized_frames: List[FrameInfo] = []

    logging.info(f"Stylizing {len(frames)} frames with up to {max_workers} concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_frame, client, f, output_dir): f for f in frames}
        for future in tqdm(as_completed(futures), total=len(frames)):
            result = future.result()
            if result:
                stylized_frames.append(result)

    return stylized_frames
