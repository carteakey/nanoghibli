import os
import argparse
import glob
import uuid
import subprocess
import json
from dotenv import load_dotenv

from extractor import extract_scenes_from_video, get_photos_from_directory
from stylizer import stylize_frames
from animator import create_video_from_frames
from veo_animator import generate_scene_video

def main():
    parser = argparse.ArgumentParser(description="NanoGhibli: Convert videos/photos to Ghibli-style trailers.")
    parser.add_argument("--input", required=True, help="Path to input video file or photo directory.")
    parser.add_argument("--mode", required=True, choices=["video", "photo"], help="Mode of operation: 'video' or 'photo'.")
    parser.add_argument("--session_id", default=None, help="Session ID to resume or use for output folder.")
    parser.add_argument("--fps", type=float, default=12.0, help="Target FPS for the output video.")
    parser.add_argument("--threshold", type=float, default=20.0, help="Motion threshold for video frame extraction.")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum duration (in seconds) to process from the video.")
    parser.add_argument("--use_veo", action="store_true", help="Use Veo 3.1 to generate a fluid video for each scene.")
    parser.add_argument("--skip_stylize", action="store_true", help="Skip extraction and stylization and proceed directly to assembly using existing frames.")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set. Please set it in .env or your shell.")
        return

    session_id = args.session_id if args.session_id else uuid.uuid4().hex[:8]
    print(f"=== Session ID: {session_id} ===")

    # Create directories for this session
    base_dir = os.path.join("data/output", session_id)
    frames_dir = os.path.join(base_dir, "extracted_frames")
    stylized_dir = os.path.join(base_dir, "stylized_frames")
    veo_dir = os.path.join(base_dir, "veo_segments")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(stylized_dir, exist_ok=True)
    if args.use_veo:
        os.makedirs(veo_dir, exist_ok=True)

    final_output = os.path.join(base_dir, "trailer.mp4")
    veo_final_output = os.path.join(base_dir, "trailer_veo.mp4")

    scenes = []
    fps = args.fps
    video_fps = 30.0

    if not args.skip_stylize:
        print("\n=== Phase 1: Frame Extraction ===")
        if args.mode == "video":
            if not os.path.isfile(args.input):
                print(f"Error: Input video file not found: {args.input}")
                return
            scenes, video_fps = extract_scenes_from_video(args.input, frames_dir, args.threshold, 35.0, args.max_duration)
            print(f"Original video FPS was: {video_fps}. Target FPS is {fps}.")
        elif args.mode == "photo":
            if not os.path.isdir(args.input):
                print(f"Error: Input directory not found: {args.input}")
                return
            scenes, video_fps = get_photos_from_directory(args.input)

        if not scenes:
            print("No scenes/frames extracted. Exiting.")
            return

        print("\n=== Phase 2: Stylization (Nano Banana 2) ===")
        frames_to_stylize = []
        for scene in scenes:
            for f in scene["frames"]:
                frames_to_stylize.append(f)
                
        existing_stylized = sorted(glob.glob(os.path.join(stylized_dir, "*.png")))
        stylized_frames = []
        
        if len(existing_stylized) >= len(frames_to_stylize):
            print(f"Found {len(existing_stylized)} existing stylized frames. Skipping stylization...")
            stylized_frames = [{"path": p, "original_frame_index": int(os.path.basename(p).split('_')[1].split('.')[0])} for p in existing_stylized]
        else:
            frames_needed = []
            for f in frames_to_stylize:
                expected_out_name = f"stylized_{f['original_frame_index']:06d}.png"
                expected_out_path = os.path.join(stylized_dir, expected_out_name)
                if os.path.exists(expected_out_path):
                    stylized_frames.append({
                        "path": expected_out_path,
                        "original_frame_index": f['original_frame_index']
                    })
                else:
                    frames_needed.append(f)
                    
            if frames_needed:
                print(f"Stylizing {len(frames_needed)} remaining frames...")
                new_stylized = stylize_frames(frames_needed, stylized_dir)
                stylized_frames.extend(new_stylized)
            
            stylized_frames.sort(key=lambda x: x["original_frame_index"])
    else:
        # Load existing scenes
        scenes_file = os.path.join(frames_dir, "scenes.json")
        if os.path.exists(scenes_file):
            with open(scenes_file, "r") as f:
                data = json.load(f)
                scenes = data["scenes"]
                video_fps = data["fps"]
        else:
            print(f"Error: Could not find {scenes_file} to resume scenes.")
            return
            
        stylized_files = sorted(glob.glob(os.path.join(stylized_dir, "*.png")))
        stylized_frames = [{"path": p, "original_frame_index": int(os.path.basename(p).split('_')[1].split('.')[0])} for p in stylized_files]
        if not stylized_frames:
             print("No existing stylized frames found. Exiting.")
             return

    # Map stylized paths back to scenes
    stylized_dict = {f["original_frame_index"]: f["path"] for f in stylized_frames}
    for scene in scenes:
        scene["stylized_frames"] = []
        for f in scene["frames"]:
            if f["original_frame_index"] in stylized_dict:
                scene["stylized_frames"].append({
                    "path": stylized_dict[f["original_frame_index"]],
                    "original_frame_index": f["original_frame_index"]
                })

    print("\n=== Phase 3: Assembly & Output ===")
    if args.mode == "video":
        if args.use_veo:
            print(f"Generating Veo segments for {len(scenes)} scenes...")
            veo_segments = []
            
            for scene in scenes:
                idx = scene["scene_index"]
                scene_stylized = scene["stylized_frames"]
                
                if not scene_stylized:
                    print(f"Skipping scene {idx}, no stylized frames.")
                    continue
                
                # Calculate duration based on original frame count
                orig_frames = scene["end_frame"] - scene["start_frame"]
                orig_sec = orig_frames / video_fps if video_fps else 4.0
                
                if orig_sec <= 4.0:
                    dur_str = "4"
                elif orig_sec <= 6.0:
                    dur_str = "6"
                else:
                    dur_str = "8"
                    
                seg_name = f"veo_scene_{idx:04d}.mp4"
                seg_path = os.path.join(veo_dir, seg_name)
                veo_segments.append(seg_path)
                
                if os.path.exists(seg_path):
                    print(f"Segment {seg_name} already exists. Skipping.")
                    continue
                
                print(f"Generating Scene {idx}/{len(scenes)}: original length {orig_sec:.1f}s -> requesting {dur_str}s Veo video")
                generate_scene_video(scene_stylized, seg_path, duration_seconds=dur_str)
            
            # Concat all segments using ffmpeg
            concat_list_path = os.path.join(veo_dir, "concat_list.txt")
            with open(concat_list_path, "w") as f:
                for seg in veo_segments:
                    f.write(f"file '{os.path.abspath(seg)}'\n")
            
            print(f"\nConcatenating segments into final Veo trailer...")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", veo_final_output],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"Initial concatenated video saved. Now adding original audio...")

                final_with_audio_path = veo_final_output.replace(".mp4", "_with_audio.mp4")
                
                # Command to extract audio from original input and lay it over the generated video.
                # Since Veo video durations might not match exactly, we'll use -shortest to trim to the shortest stream.
                subprocess.run(
                    [
                        "ffmpeg", "-y", 
                        "-i", veo_final_output, 
                        "-i", args.input, 
                        "-c:v", "copy", 
                        "-c:a", "aac", 
                        "-map", "0:v:0", 
                        "-map", "1:a:0", 
                        "-shortest", 
                        final_with_audio_path
                    ],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"Done! Veo Video with original audio saved to: {final_with_audio_path}")

            except subprocess.CalledProcessError as e:
                print(f"Error concatenating videos with ffmpeg. Check if ffmpeg is installed properly.")
                print("You can manually concatenate the segments located in:", veo_dir)
            except FileNotFoundError:
                print(f"ffmpeg not found. Please install ffmpeg to concatenate the video segments.")
                print("Segments are located in:", veo_dir)

        else:
            create_video_from_frames(stylized_frames, final_output, args.fps)
            print(f"\nDone! Video output saved to: {final_output}")
    else:
        print(f"\nDone! Stylized photos saved to: {stylized_dir}")

if __name__ == "__main__":
    main()