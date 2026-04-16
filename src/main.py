import os
import sys
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
from stylizer import stylize_frames, get_scene_description, GHIBLI_PROMPT
from animator import create_video_from_frames
from veo_animator import generate_scene_video
from director import get_video_script
from models import UsageMetrics, QuotaExceededError
import batch_stylizer

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


def pick(cli_val, cfg_val, default):
    """CLI wins if the user set it, else config, else default.
    Unlike `a or b`, 0 / 0.0 count as user-provided values.
    """
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default

def main():
    """Public entry point. Handles QuotaExceededError cleanly so users get a
    resume hint instead of an ugly traceback."""
    try:
        _run_main()
    except QuotaExceededError as e:
        logging.error(
            f"Daily quota exhausted: {e}. "
            "Partial state is saved in data/output/<session_id>/ — rerun "
            "with the same --session_id after quota resets to resume."
        )
        # If the inner code attached metrics to the exception, print them.
        metrics = getattr(e, "metrics_snapshot", None)
        if metrics is not None:
            print(metrics)
        sys.exit(2)


def _run_main():
    parser = argparse.ArgumentParser(description="NanoGhibli: Convert videos/photos to Ghibli-style trailers.")
    parser.add_argument("--input", required=True, nargs='+', help="Path(s) to input video file or photo directory.")
    parser.add_argument("--mode", required=True, choices=["video", "photo"], help="Mode of operation: 'video' or 'photo'.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--session_id", default=None, help="Session ID to resume or use for output folder.")
    parser.add_argument("--fps", type=float, help="Target FPS for the output video.")
    parser.add_argument("--threshold", type=float, help="Motion threshold for video frame extraction.")
    parser.add_argument("--scene_threshold", type=float, help="Scene change threshold.")
    parser.add_argument("--max_duration", type=float, help="Maximum duration (in seconds) to process from the video.")
    parser.add_argument("--batch", action="store_true", help="Route stylization through the Gemini Batch API (50%% cost, async, 24h SLO). Veo stays synchronous.")
    parser.add_argument("--use_veo", action="store_true", help="Use Veo 3.1 to generate a fluid video for each scene.")
    parser.add_argument("--use_director", action="store_true", help="Use Gemini to first analyze the video and create an intelligent edit script.")
    parser.add_argument("--output_format", choices=["mp4", "gif"], help="Output format for the final video.")
    parser.add_argument("--skip_stylize", action="store_true", help="Skip extraction and stylization and proceed directly to assembly.")
    parser.add_argument("--skip_video", action="store_true", help="Skip video generation (Veo) and assembly. Useful for pre-production stylization.")
    parser.add_argument("--skip_black_frames", action="store_true", help="Skip black frames during extraction.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")
    
    # Model parameters
    parser.add_argument("--stylizer_model", choices=["flash", "pro"], default="flash", help="Choose between high-efficiency (flash) and high-fidelity (pro) stylizer models.")
    parser.add_argument("--temperature", type=float, help="Temperature for stylizer model.")
    parser.add_argument("--top_p", type=float, help="Top_p for stylizer model.")
    parser.add_argument("--top_k", type=int, help="Top_k for stylizer model.")
    
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    config = load_config(args.config)
    
    # Merge config and args. Use pick() rather than `or` so that 0 / 0.0 / False
    # passed on the CLI are respected as explicit user values.
    proc = config.get("processing", {})
    stylizer_cfg = config.get("models", {}).get("stylizer", {})

    fps = pick(args.fps, proc.get("fps"), 12.0)
    threshold = pick(args.threshold, proc.get("motion_threshold"), 20.0)
    scene_threshold = pick(args.scene_threshold, proc.get("scene_threshold"), 35.0)
    max_duration = pick(args.max_duration, proc.get("max_duration"), None)
    output_format = pick(args.output_format, proc.get("output_format"), "mp4")
    # Boolean flags: argparse store_true defaults to False, so use `or` semantics
    # deliberately here (either source can enable).
    use_veo = args.use_veo or bool(proc.get("use_veo", False))
    use_director = args.use_director or bool(proc.get("use_director", False))
    skip_black_frames = args.skip_black_frames or bool(proc.get("skip_black_frames", False))
    batch_enabled = args.batch or bool(proc.get("batch_enabled", False))
    batch_poll_interval = int(proc.get("batch_poll_interval", 30))
    batch_max_wait_hours = float(proc.get("batch_max_wait_hours", 24.0))
    max_workers = proc.get("max_workers", 4)

    temp = pick(args.temperature, stylizer_cfg.get("temperature"), 0.7)
    top_p = pick(args.top_p, stylizer_cfg.get("top_p"), 0.95)
    top_k = pick(args.top_k, stylizer_cfg.get("top_k"), 40)

    # Load environment variables
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        logging.error("GEMINI_API_KEY environment variable not set. Please set it in .env or your shell.")
        return

    client = genai.Client()
    
    # Model Selection
    if args.stylizer_model == "pro":
        stylizer_model_id = "gemini-3-pro-image-preview"
        model_tier = "pro"
    else:
        stylizer_model_id = "gemini-3.1-flash-image-preview"
        model_tier = "flash"

    metrics = UsageMetrics(model_tier=model_tier)

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

            mode_label = "BATCH" if batch_enabled else args.stylizer_model.upper()
            logging.info(f"=== Phase 2: Stylization ({mode_label}) ===")

            # Stylize scene by scene to maintain consistency and generate descriptions.
            # We hot-write scenes.json after each scene so a mid-loop crash or
            # quota hit leaves durable progress for the next --session_id rerun.
            scenes_file = os.path.join(frames_dir, "scenes.json")

            def save_scenes_progress():
                with open(scenes_file, "w") as f:
                    json.dump({"fps": video_fps, "scenes": scenes}, f, indent=2)

            # Pass 1: generate descriptions (fast Flash-Lite, keeps sync for
            # both paths) and determine which frames still need stylization.
            # Split into per-scene dicts so the batch path can key results back.
            scene_buckets = []  # list of {"scene": ..., "need": [...], "done": [...]}
            for scene in scenes:
                if not scene.get("description"):
                    rep_frame = scene["frames"][0]["path"]
                    scene["description"] = get_scene_description(
                        client, rep_frame, metrics=metrics
                    )
                need = []
                done = []
                for f in scene["frames"]:
                    expected_out_name = f"stylized_{f['original_frame_index']:06d}.png"
                    expected_out_path = os.path.join(stylized_dir, expected_out_name)
                    if os.path.exists(expected_out_path):
                        done.append({
                            "path": expected_out_path,
                            "original_frame_index": f["original_frame_index"],
                        })
                    else:
                        need.append(f)
                scene_buckets.append({"scene": scene, "need": need, "done": done})
            save_scenes_progress()  # persist descriptions before expensive work

            if batch_enabled:
                # Flatten all needed frames into one batch submission.
                stylize_items = []
                for b in scene_buckets:
                    scene = b["scene"]
                    desc = scene.get("description", "")
                    context_prefix = f"The scene contains: {desc}. " if desc else ""
                    full_prompt = f"{context_prefix}{GHIBLI_PROMPT}"
                    for f in b["need"]:
                        stylize_items.append({
                            "key": f"frame_{f['original_frame_index']:06d}",
                            "frame_path": f["path"],
                            "original_frame_index": f["original_frame_index"],
                            "prompt": full_prompt,
                        })

                if stylize_items:
                    logging.info(
                        f"Submitting batch of {len(stylize_items)} frames "
                        f"across {len(scene_buckets)} scenes..."
                    )
                    batch_results = batch_stylizer.run_stylize_batch(
                        client, stylize_items,
                        model_id=stylizer_model_id,
                        output_dir=stylized_dir,
                        cache_dir=stylized_cache,
                        session_dir=base_dir,
                        temperature=temp, top_p=top_p, top_k=top_k,
                        poll_interval=batch_poll_interval,
                        max_wait_hours=batch_max_wait_hours,
                    )
                    # Count token+image costs for the batch (50% tier).
                    # The batch SDK response doesn't give per-request usage_metadata
                    # in a reliable shape; we bill per-image at the batch rate.
                    for _ in batch_results:
                        metrics.add_image(stylizer_model_id, is_batch=True)
                    # Map back by original_frame_index so each scene gets its frames.
                    by_idx = {r["original_frame_index"]: r for r in batch_results}
                    for b in scene_buckets:
                        extra = [by_idx[f["original_frame_index"]] for f in b["need"]
                                 if f["original_frame_index"] in by_idx]
                        b["done"].extend(extra)

            else:
                # Sync path: per-scene stylize_frames(), hot-write scenes.json
                # after each scene so a mid-loop quota hit leaves durable state.
                for b in scene_buckets:
                    scene = b["scene"]
                    if b["need"]:
                        logging.info(
                            f"Stylizing scene {scene['scene_index']} "
                            f"({len(b['need'])} frames) using {args.stylizer_model.upper()}..."
                        )
                        new_stylized = stylize_frames(
                            b["need"], stylized_dir,
                            model_id=stylizer_model_id, cache_dir=stylized_cache,
                            max_workers=max_workers, temperature=temp,
                            top_p=top_p, top_k=top_k,
                            scene_description=scene.get("description", ""),
                            metrics=metrics,
                        )
                        b["done"].extend(new_stylized)
                    b["done"].sort(key=lambda x: x["original_frame_index"])
                    scene["stylized_frames"] = b["done"]
                    save_scenes_progress()

            # Finalize scene["stylized_frames"] for both paths, flush once more.
            all_stylized_frames = []
            for b in scene_buckets:
                b["done"].sort(key=lambda x: x["original_frame_index"])
                b["scene"]["stylized_frames"] = b["done"]
                all_stylized_frames.extend(b["done"])
            save_scenes_progress()

        else:
            # Load existing scenes
            scenes_file = os.path.join(frames_dir, "scenes.json")
            if os.path.exists(scenes_file):
                with open(scenes_file, "r") as f:
                    data = json.load(f)
                    scenes = data["scenes"]
                    video_fps = data["fps"]
            else:
                logging.error(f"Could not find {scenes_file} to resume scenes for {input_path}.")
                continue
                
            stylized_files = sorted(glob.glob(os.path.join(stylized_dir, "*.png")))
            stylized_frames = [{"path": p, "original_frame_index": int(os.path.basename(p).split('_')[1].split('.')[0])} for p in stylized_files]
            if not stylized_frames:
                 logging.warning(f"No existing stylized frames found for {input_path}. Skipping.")
                 continue
            
            # Map back to scenes
            stylized_dict = {f["original_frame_index"]: f["path"] for f in stylized_frames}
            for scene in scenes:
                scene["stylized_frames"] = []
                for f in scene["frames"]:
                    if f["original_frame_index"] in stylized_dict:
                        scene["stylized_frames"].append({
                            "path": stylized_dict[f["original_frame_index"]],
                            "original_frame_index": f["original_frame_index"]
                        })

        if args.skip_video:
            logging.info("=== Pre-Production Finished! Stylized frames are cached. ===")
            continue

        logging.info("=== Phase 3: Assembly & Output ===")
        if args.mode == "video":
            if use_veo:
                logging.info(f"Generating Veo segments for {len(scenes)} scenes...")
                sync_segments = []

                # Per-segment state file. Keyed by seg_id so reruns can skip
                # scenes already marked 'synced' without re-checking files.
                # States: "pending" -> "veo_done" -> "synced" -> "failed".
                veo_progress_path = os.path.join(base_dir, "veo_progress.json")
                if os.path.exists(veo_progress_path):
                    with open(veo_progress_path, "r") as pf:
                        veo_progress = json.load(pf)
                else:
                    veo_progress = {}

                def save_veo_progress():
                    with open(veo_progress_path, "w") as pf:
                        json.dump(veo_progress, pf, indent=2)

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

                    # SEMANTIC NAMING: slug_model_start_end.mp4
                    # Include stylizer tier in the key so flash/pro runs don't
                    # reuse each other's Veo outputs (different stylized frames
                    # produce different-looking Veo videos).
                    seg_id = f"{input_slug}_{model_tier}_{int(orig_start*1000):06d}_{int(orig_end*1000):06d}"
                    seg_path = os.path.join(veo_dir, f"{seg_id}.mp4")
                    sync_seg_path = os.path.join(veo_dir, f"{seg_id}_sync.mp4")
                    global_sync_path = os.path.join(segments_cache, f"{seg_id}_sync.mp4")

                    # Short-circuit: prior run already marked this scene synced.
                    prog = veo_progress.get(seg_id, {})
                    if prog.get("state") == "synced" and os.path.exists(sync_seg_path):
                        logging.info(f"Scene {idx} already synced in prior run — reusing.")
                        sync_segments.append(sync_seg_path)
                        continue

                    # Check Global Cache First
                    if os.path.exists(global_sync_path):
                        logging.info(f"Reusing synced segment from library: {seg_id}")
                        if not os.path.exists(sync_seg_path):
                            shutil.copy(global_sync_path, sync_seg_path)
                        sync_segments.append(sync_seg_path)
                        veo_progress[seg_id] = {"state": "synced", "sync_path": sync_seg_path}
                        save_veo_progress()
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
                        veo_progress[seg_id] = {"state": "pending", "dur_str": dur_str}
                        save_veo_progress()
                        logging.info(f"Generating Scene {idx}: original {orig_duration:.2f}s -> requesting {dur_str}s Veo")
                        generate_scene_video(
                            scene_stylized,
                            seg_path,
                            duration_seconds=dur_str,
                            scene_description=scene.get("description", ""),
                            metrics=metrics
                        )
                        if os.path.exists(seg_path):
                            veo_progress[seg_id] = {"state": "veo_done", "veo_path": seg_path, "dur_str": dur_str}
                            save_veo_progress()

                    # 2. SYNC LOGIC: Slice audio and conform video
                    if os.path.exists(seg_path) and not os.path.exists(sync_seg_path):
                        logging.info(f"Synchronizing Scene {idx} to exactly {orig_duration:.2f}s...")
                        try:
                            # Slice original audio and conform Veo video to the
                            # original scene's duration. dur_str is the length
                            # Veo actually produced (4 / 6 / 8s).
                            veo_src_seconds = float(dur_str)
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", seg_path, # The Veo video
                                "-ss", f"{orig_start:.3f}", "-t", f"{orig_duration:.3f}", "-i", input_path, # Source audio
                                "-filter_complex", f"[0:v]setpts=({orig_duration}/{veo_src_seconds})*PTS[v]",
                                "-map", "[v]", "-map", "1:a:0",
                                "-c:v", "libx264", "-c:a", "aac", "-shortest",
                                sync_seg_path
                            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                            # Save to global cache
                            shutil.copy(sync_seg_path, global_sync_path)
                            veo_progress[seg_id] = {"state": "synced", "veo_path": seg_path, "sync_path": sync_seg_path}
                            save_veo_progress()

                        except Exception as e:
                            logging.error(f"Failed to sync scene {idx}: {e}")
                            veo_progress[seg_id] = {"state": "failed", "reason": str(e), "veo_path": seg_path}
                            save_veo_progress()
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
