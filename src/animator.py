import cv2
import os
import logging
from tqdm import tqdm
from typing import List
from PIL import Image

from models import FrameInfo

def create_video_from_frames(frames: List[FrameInfo], output_path: str, fps: float = 30.0):
    """
    Assembles a list of frame dicts into an mp4 video or gif.
    frames: list of dicts [{"path": ..., "original_frame_index": ...}]
    """
    if not frames:
        logging.warning("No frames to assemble.")
        return

    # Sort frames by original index to maintain order
    frames.sort(key=lambda x: x.get("original_frame_index", 0))

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if output_path.lower().endswith(".gif"):
        logging.info(f"Assembling GIF to {output_path} at {fps} FPS...")
        imgs = []
        for frame_info in tqdm(frames):
            img_path = frame_info["path"]
            try:
                img = Image.open(img_path)
                # Convert to RGB if not already
                if img.mode != "RGB":
                    img = img.convert("RGB")
                imgs.append(img)
            except Exception as e:
                logging.warning(f"Could not read frame {img_path}: {e}")
        
        if imgs:
            duration = int(1000 / fps)
            # Use adaptive palette for better quality
            imgs_quantized = []
            for img in imgs:
                try:
                    # Try using LIBIMAGEQUANT for best quality if available
                    imgs_quantized.append(img.quantize(colors=256, method=Image.Quantize.LIBIMAGEQUANT))
                except (ValueError, AttributeError):
                    # Fallback to standard MEDIANCUT
                    imgs_quantized.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
            
            imgs_quantized[0].save(
                output_path,
                save_all=True,
                append_images=imgs_quantized[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            logging.info("GIF assembly complete.")
        return

    # Read the first image to get dimensions
    first_frame_path = frames[0]["path"]
    first_img = cv2.imread(first_frame_path)
    
    if first_img is None:
        raise ValueError(f"Could not read the first frame at {first_frame_path}")
        
    height, width, layers = first_img.shape
    size = (width, height)

    # Use mp4v codec for standard mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    logging.info(f"Assembling video to {output_path} at {fps} FPS...")
    for frame_info in tqdm(frames):
        img_path = frame_info["path"]
        img = cv2.imread(img_path)
        if img is not None:
            # Resize if necessary to match the first frame, though Gemini typically 
            # maintains aspect ratio. Better safe than sorry if dimensions differ slightly.
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size)
            out.write(img)
        else:
            logging.warning(f"Could not read frame to write {img_path}")

    out.release()
    logging.info("Video assembly complete.")
