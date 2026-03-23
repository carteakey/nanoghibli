import os
import argparse
import glob
import uuid
import subprocess
import json
import logging
import yaml
import re
import shutil
from google import genai
from dotenv import load_dotenv

from extractor import extract_scenes_from_video, get_photos_from_directory, extract_frames_from_script
from stylizer import stylize_frames, get_scene_description
from animator import create_video_from_frames
from veo_animator import generate_scene_video
from director import get_video_script
from models import UsageMetrics

def get_slug(path: str) -> str:
    """Creates a URL-friendly slug from a filename."""
    name = os.path.basename(path).split('.')[0]
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="NanoGhibli: Convert videos/photos to Ghibli-style trailers.")
    parser.add_argument("--input", required=True, nargs='+', help="Path(s) to input video file or photo directory.")
    parser.add_argument("--mode", required=True, choices=["video", "photo"], help="Mode of operation: 'video' or 'photo'.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--session_id", default=None, help="Session ID to resume or use for output folder.")
    parser.add_argument("--fps", type=float, help="Target FPS for the output video.")
    parser.add_argument("--threshold", type=float, help="Motion threshold for video frame extraction.")
    parser.add_argument("--scene_threshold", type=float, help="Scene change threshold.")
    parser.add_argument("--max_duration", type=float, help="Maximum duration (in seconds) to process from the video.")
    parser.add_argument("--use_veo", action="store_true", help="Use Veo 3.1 to generate a fluid video for each scene.")
    parser.add_argument("--use_director", action="store_true", help="Use Gemini to first analyze the video and create an intelligent edit script.")
    parser.add_argument("--output_format", choices=["mp4", "gif"], help="Output format for the final video.")
    parser.add_argument("--skip_stylize", action="store_true", help="Skip extraction and stylization and proceed directly to assembly.")
    parser.add_argument("--skip_video", action="store_true", help="Skip video generation (Veo) and assembly. Useful for pre-production stylization.")
    parser.add_argument("--skip_black_frames", action="store_true", help="Skip black frames during extraction.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")
    
    # Model parameters
    parser.add_argument("--temperature", type=float, help="Temperature for stylizer model.")
    parser.add_argument("--top_p", type=float, help="Top_p for stylizer model.")
    parser.add_argument("--top_k", type=int, help="Top_k for stylizer model.")
    
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    config = load_config(args.config)
    
    # Merge config and args (args override config)
    fps = args.fps or config.get("processing", {}).get("fps", 12.0)
    threshold = args.threshold or config.get("processing", {}).get("motion_threshold", 20.0)
    scene_threshold = args.scene_threshold or config.get("processing", {}).get("scene_threshold", 35.0)
    max_duration = args.max_duration or config.get("processing", {}).get("max_duration")
    output_format = args.output_format or config.get("processing", {}).get("output_format", "mp4")
    use_veo = args.use_veo or config.get("processing", {}).get("use_veo", False)
    use_director = args.use_director or config.get("processing", {}).get("use_director", False)
    skip_black_frames = args.skip_black_frames or config.get("processing", {}).get("skip_black_frames", False)
    max_workers = config.get("processing", {}).get("max_workers", 4)
    
    # Stylizer params
    temp = args.temperature or config.get("models", {}).get("stylizer", {}).get("temperature", 0.7)
    top_p = args.top_p or config.get("models", {}).get("stylizer", {}).get("top_p", 0.95)
    top_k = args.top_k or config.get("models", {}).get("stylizer", {}).get("top_k", 40)

    # Load environment variables
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        logging.error("GEMINI_API_KEY environment variable not set. Please set it in .env or your shell.")
        return

    client = genai.Client()
    metrics = UsageMetrics()
    
    cache_base = "data/cache"
    stylized_cache = os.path.join(cache_base, "stylized")
    segments_cache = os.path.join(cache_base, "segments")
    os.makedirs(stylized_cache, exist_ok=True)
    os.makedirs(segments_cache, exist_ok=True)

    for input_path in args.input:
        input_slug = get_slug(input_path)
        session_id = args.session_id if args.session_id else f"{uuid.uuid4().hex[:8]}_{input_slug}"
        logging.info(f"=== Processing: {input_path} (Session ID: {session_id}) ===")

        # Create directories for this session
        base_dir = os.path.join("data/output", session_id)
        frames_dir = os.path.join(base_dir, "extracted_frames")
        stylized_dir = os.path.join(base_dir, "stylized_frames")
        veo_dir = os.path.join(base_dir, "veo_segments")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(stylized_dir, exist_ok=True)
        if use_veo:
            os.makedirs(veo_dir, exist_ok=True)

        ext = "gif" if output_format == "gif" else "mp4"
        final_output = os.path.join(base_dir, f"trailer.{ext}")
        veo_final_output = os.path.join(base_dir, f"trailer_veo.{ext}")

        scenes = []
        video_fps = 30.0

        if not args.skip_stylize:
            # === Phase 0: The Director (Optional) ===
            script = None
            if args.mode == "video" and use_director:
                script_file = os.path.join(base_dir, "director_script.json")
                if os.path.exists(script_file):
                    logging.info(f"Loading existing Director Script from {script_file}...")
                    with open(script_file, "r") as f:
                        script = json.load(f)
                else:
                    logging.info("=== Phase 0: The Director Phase ===")
                    script = get_video_script(client, input_path, base_dir, metrics=metrics)
                    if script:
                        with open(script_file, "w") as f:
                            json.dump(script, f, indent=2)

            logging.info("=== Phase 1: Frame Extraction ===")
            if args.mode == "video":
                if not os.path.isfile(input_path):
                    logging.error(f"Input video file not found: {input_path}")
                    continue
                
                if script:
                    scenes, video_fps = extract_frames_from_script(input_path, frames_dir, script)
                else:
                    scenes, video_fps = extract_scenes_from_video(input_path, frames_dir, threshold, scene_threshold, max_duration, skip_black_frames)
                
                logging.info(f"Original video FPS was: {video_fps}. Target FPS is {fps}.")
            elif args.mode == "photo":
                if not os.path.isdir(input_path):
                    logging.error(f"Input directory not found: {input_path}")
                    continue
                scenes, video_fps = get_photos_from_directory(input_path)

            if not scenes:
                logging.warning(f"No scenes/frames extracted for {input_path}. Skipping.")
                continue

            logging.info("=== Phase 2: Stylization (Nano Banana 2) ===")
            
            # Stylize scene by scene to maintain consistency and generate descriptions
            all_stylized_frames = []
            for scene in scenes:
                # 1. SMART PROMPT: Generate scene description if not exists
                if not scene.get("description"):
                    # Use the first frame of the scene as a representative for description
                    rep_frame = scene["frames"][0]["path"]
                    scene["description"] = get_scene_description(client, rep_frame, metrics=metrics)
                
                # Check for existing stylized frames for this scene
                scene_frames_needed = []
                scene_stylized = []
                for f in scene["frames"]:
                    expected_out_name = f"stylized_{f['original_frame_index']:06d}.png"
                    expected_out_path = os.path.join(stylized_dir, expected_out_name)
                    if os.path.exists(expected_out_path):
                        scene_stylized.append({
                            "path": expected_out_path,
                            "original_frame_index": f['original_frame_index']
                        })
                    else:
                        scene_frames_needed.append(f)
                
                if scene_frames_needed:
                    logging.info(f"Stylizing scene {scene['scene_index']} ({len(scene_frames_needed)} frames)...")
                    new_stylized = stylize_frames(
                        scene_frames_needed, 
                        stylized_dir, 
                        cache_dir=stylized_cache,
                        max_workers=max_workers, 
                        temperature=temp, 
                        top_p=top_p, 
                        top_k=top_k,
                        scene_description=scene.get("description", ""),
                        metrics=metrics
                    )
                    scene_stylized.extend(new_stylized)
                
                scene_stylized.sort(key=lambda x: x["original_frame_index"])
                all_stylized_frames.extend(scene_stylized)
                scene["stylized_frames"] = scene_stylized
            
            # Save updated metadata with descriptions
            scenes_file = os.path.join(frames_dir, "scenes.json")
            with open(scenes_file, "w") as f:
                json.dump({"fps": video_fps, "scenes": scenes}, f, indent=2)

        if args.skip_video:
            logging.info("=== Pre-Production Finished! Stylized frames are cached. ===")
            continue

        logging.info("=== Phase 3: Assembly & Output ===")
        if args.mode == "video":
            if use_veo:
                logging.info(f"Generating Veo segments for {len(scenes)} scenes...")
                sync_segments = []
                
                for scene in scenes:
                    idx = scene["scene_index"]
                    scene_stylized = scene["stylized_frames"]
                    
                    if not scene_stylized:
                        logging.warning(f"Skipping scene {idx}, no stylized frames.")
                        continue
                    
                    # 1. Calculate durations
                    orig_start = scene.get("start_frame", 0) / video_fps
                    orig_end = scene.get("end_frame", 0) / video_fps
                    orig_duration = orig_end - orig_start
                    
                    # SEMANTIC NAMING: slug_start_end.mp4
                    seg_id = f"{input_slug}_{int(orig_start*1000):06d}_{int(orig_end*1000):06d}"
                    seg_path = os.path.join(veo_dir, f"{seg_id}.mp4")
                    sync_seg_path = os.path.join(veo_dir, f"{seg_id}_sync.mp4")
                    global_sync_path = os.path.join(segments_cache, f"{seg_id}_sync.mp4")
                    
                    # Check Global Cache First
                    if os.path.exists(global_sync_path):
                        logging.info(f"Reusing synced segment from library: {seg_id}")
                        if not os.path.exists(sync_seg_path):
                            shutil.copy(global_sync_path, sync_seg_path)
                        sync_segments.append(sync_seg_path)
                        continue

                    # Determine Veo request duration
                    if orig_duration <= 4.0:
                        dur_str = "4"
                    elif orig_duration <= 6.0:
                        dur_str = "6"
                    else:
                        dur_str = "8"
                        
                    # Generate Veo if not exists
                    if not os.path.exists(seg_path):
                        logging.info(f"Generating Scene {idx}: original {orig_duration:.2f}s -> requesting {dur_str}s Veo")
                        generate_scene_video(
                            scene_stylized, 
                            seg_path, 
                            duration_seconds=dur_str,
                            scene_description=scene.get("description", ""),
                            metrics=metrics
                        )
                    
                    # 2. SYNC LOGIC: Slice audio and conform video
                    if os.path.exists(seg_path) and not os.path.exists(sync_seg_path):
                        logging.info(f"Synchronizing Scene {idx} to exactly {orig_duration:.2f}s...")
                        try:
                            # Slice original audio and conform video in one command
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", seg_path, # The Veo video
                                "-ss", f"{orig_start:.3f}", "-t", f"{orig_duration:.3f}", "-i", input_path, # Source audio
                                "-filter_complex", f"[0:v]setpts=({orig_duration}/4)*PTS[v]", # Adjust based on default Veo 4s base
                                "-map", "[v]", "-map", "1:a:0",
                                "-c:v", "libx264", "-c:a", "aac", "-shortest",
                                sync_seg_path
                            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            # Save to global cache
                            shutil.copy(sync_seg_path, global_sync_path)
                            
                        except Exception as e:
                            logging.error(f"Failed to sync scene {idx}: {e}")
                            if os.path.exists(seg_path):
                                sync_seg_path = seg_path
                    
                    if os.path.exists(sync_seg_path):
                        sync_segments.append(sync_seg_path)
                
                # 3. Final Concat
                if sync_segments:
                    concat_list_path = os.path.join(veo_dir, "concat_list.txt")
                    with open(concat_list_path, "w") as f:
                        for seg in sync_segments:
                            f.write(f"file '{os.path.abspath(seg)}'\n")
                    
                    logging.info(f"Concatenating {len(sync_segments)} synced segments into final trailer...")
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", veo_final_output],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        logging.info(f"Done! Perfectly Synced Trailer saved to: {veo_final_output}")
                    except Exception as e:
                        logging.error(f"Final concatenation failed: {e}")
                else:
                    logging.error("No valid segments to assemble.")

            else:
                # Assembly without Veo
                all_stylized_ordered = []
                for scene in scenes:
                    all_stylized_ordered.extend(scene["stylized_frames"])
                
                create_video_from_frames(all_stylized_ordered, final_output, fps)
                logging.info(f"Initial assembled video saved to: {final_output}. Now adding original audio...")
                
                final_with_audio_path = final_output.replace(f".{ext}", f"_with_audio.{ext}")
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", 
                            "-i", final_output, 
                            "-i", input_path, 
                            "-c:v", "copy", 
                            "-c:a", "aac", 
                            "-map", "0:v:0", 
                            "-map", "1:a:0", 
                            "-shortest", 
                            final_with_audio_path
                        ],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    logging.info(f"Done! Standard Video with original audio saved to: {final_with_audio_path}")
                except subprocess.CalledProcessError:
                    logging.error("Error adding audio with ffmpeg. The silent video is available.")
                except FileNotFoundError:
                    logging.error("ffmpeg not found. The silent video is available.")

        else:
            logging.info(f"Done! Stylized photos saved to: {stylized_dir}")

    # Print final session costs
    print(metrics)

if __name__ == "__main__":
    main()
