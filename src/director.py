import os
import time
import json
import logging
import subprocess
from google import genai
from google.genai import types

from models import UsageMetrics

def create_lowres_proxy(video_path: str, output_path: str):
    """
    Creates a small, low-bitrate version of the video for AI analysis.
    """
    if os.path.exists(output_path):
        return output_path
        
    logging.info(f"Creating low-res proxy for analysis: {output_path}")
    try:
        # Scale to 480p, no audio, high compression
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "scale=-2:480",
            "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
            "-an", 
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        logging.warning(f"FFmpeg proxy creation failed: {e}. Using original file.")
        return video_path

def get_video_script(client: genai.Client, video_path: str, session_dir: str, metrics: UsageMetrics = None) -> list:
    """
    Uploads the video to Gemini and generates a content-aware edit script.
    """
    proxy_path = os.path.join(session_dir, "analysis_proxy.mp4")
    proxy_path = create_lowres_proxy(video_path, proxy_path)

    # 1. Upload to File API
    logging.info(f"Uploading {proxy_path} to Gemini for analysis...")
    video_file = client.files.upload(file=proxy_path)
    
    # Poll for completion
    while video_file.state.name == "PROCESSING":
        logging.info("Gemini is analyzing the video file...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        logging.error("Gemini video processing failed.")
        return []

    # 2. Ask the Director
    prompt = (
        "You are an expert film editor and Studio Ghibli director. "
        "Watch this video and generate a structured 'Director's Script' for a trailer. "
        "Identify the most visually iconic and narratively important segments. "
        "Return a JSON list of objects, each containing: "
        " - start_time: float (seconds) "
        " - end_time: float (seconds) "
        " - type: string (one of 'dialogue', 'action', 'landscape', 'portrait', 'transition') "
        " - description: string (brief 15-word visual description focusing on subjects and lighting) "
        " - importance: integer (1-10, where 10 is critical for the trailer) "
        "\nRules: "
        "1. Capture dialogue segments accurately. "
        "2. Ensure scenes are at least 1.5 seconds long. "
        "3. Focus on clarity and visual beauty. "
        "4. Output ONLY valid JSON."
    )

    # Director runs once per video and drives what becomes the trailer, so
    # spend on a real video-understanding model rather than the cheapest tier.
    director_model = "gemini-3-flash-preview"

    logging.info(f"Requesting Edit Script from Gemini Director ({director_model})...")
    try:
        response = client.models.generate_content(
            model=director_model,
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2 # Lower temperature for structural accuracy
            )
        )

        if metrics:
            metrics.add_usage(response, director_model)

        script = json.loads(response.text)
        logging.info(f"Director identified {len(script)} key segments.")
        return script

    except Exception as e:
        logging.error(f"Failed to generate Director Script: {e}")
        if 'response' in locals():
            logging.debug(f"Raw Response: {response.text}")
        return []
    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception as e:
            logging.debug(f"Could not delete uploaded file {video_file.name}: {e}")
