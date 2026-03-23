import os
import cv2
import glob
import json
import logging
from typing import Tuple, List
from scenedetect import detect, ContentDetector
from tqdm import tqdm

from models import Scene, FrameInfo

def find_best_start_frame(cap, start_f, end_f, max_scan=15):
    """
    Scans the first few frames of a scene to find the first one that isn't too dark or empty.
    Returns the frame index.
    """
    best_f = start_f
    max_brightness = -1
    
    for f in range(start_f, min(start_f + max_scan, end_f)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        # If we find a frame with decent brightness (> 30), we take it
        if brightness > 30:
            return f
            
        if brightness > max_brightness:
            max_brightness = brightness
            best_f = f
            
    return best_f

def extract_scenes_from_video(video_path: str, output_dir: str, motion_threshold: float = 20.0, scene_threshold: float = 35.0, max_duration: float = None, skip_black_frames: bool = False) -> Tuple[List[Scene], float]:
    """
    Extract frames from a video and group them into scenes based on motion/scene changes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenes_file = os.path.join(output_dir, "scenes.json")
    if os.path.exists(scenes_file):
        logging.info(f"Loading existing scenes from {scenes_file}...")
        with open(scenes_file, "r") as f:
            data = json.load(f)
            return data["scenes"], data["fps"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0 # Default fallback

    max_frames = None
    if max_duration is not None:
        max_frames = int(fps * max_duration)

    logging.info(f"Detecting scene boundaries in {video_path}...")
    # Use ContentDetector with the provided scene_threshold
    scene_list = detect(video_path, ContentDetector(threshold=scene_threshold))

    # If no scenes were detected (e.g., single continuous shot), manually create one scene
    if not scene_list:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = max_frames if max_frames else 1000
        scene_list_frames = [(0, total_frames)]
    else:
        # Convert FrameTimecode to frame numbers
        scene_list_frames = [(s[0].get_frames(), s[1].get_frames()) for s in scene_list]

    # Filter/truncate scenes if max_frames is set
    filtered_scene_list = []
    if max_frames is not None:
        for start_f, end_f in scene_list_frames:
            if start_f >= max_frames:
                break
            if end_f > max_frames:
                end_f = max_frames
            filtered_scene_list.append((start_f, end_f))
        scene_list_frames = filtered_scene_list

    logging.info(f"Extracting frames from {len(scene_list_frames)} scenes...")
    saved_count = 0
    scenes: List[Scene] = []

    for i, (start_f, end_f) in enumerate(tqdm(scene_list_frames, desc="Processing Scenes")):
        # SEEK FOR LIGHT: Find the actual best start frame for this scene
        actual_start_f = find_best_start_frame(cap, start_f, end_f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_f)
        
        current_scene: Scene = {
            "scene_index": i,
            "start_frame": start_f,
            "end_frame": end_f,
            "frames": []
        }
        
        prev_frame_gray = None
        for frame_count in range(actual_start_f, end_f):
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for faster motion detection processing
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if skip_black_frames and gray.mean() < 5.0:
                continue

            is_keyframe = False

            if prev_frame_gray is None:
                # First frame of the scene is always a keyframe
                is_keyframe = True
            else:
                diff = cv2.absdiff(prev_frame_gray, gray)
                mean_diff = diff.mean()
                if mean_diff > motion_threshold:
                    is_keyframe = True

            if is_keyframe:
                frame_name = f"frame_{frame_count:06d}.jpg"
                out_path = os.path.join(output_dir, frame_name)
                
                # Memory Optimization: Resize if dimensions exceed 1920x1080
                h, w = frame.shape[:2]
                max_w, max_h = 1920, 1080
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    logging.debug(f"Resized frame {frame_count} from {w}x{h} to {new_w}x{new_h}")

                cv2.imwrite(out_path, frame)
                
                current_scene["frames"].append({
                    "path": out_path,
                    "original_frame_index": frame_count
                })
                saved_count += 1
                prev_frame_gray = gray

        if current_scene["frames"]:
            scenes.append(current_scene)

    cap.release()
    logging.info(f"Extracted {saved_count} frames across {len(scenes)} scenes.")
    
    # Save metadata
    with open(scenes_file, "w") as f:
        json.dump({"fps": fps, "scenes": scenes}, f, indent=2)
        
    return scenes, fps

def extract_frames_from_script(video_path: str, output_dir: str, script: list) -> Tuple[List[Scene], float]:
    """
    Extracts frames based on a Director's Script.
    Sampling density scales with 'importance' and 'type'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 24.0

    scenes: List[Scene] = []
    saved_count = 0

    logging.info(f"Director-Led Extraction: Processing {len(script)} segments...")

    for i, segment in enumerate(tqdm(script, desc="Extracting Director Scenes")):
        start_time = segment.get("start_time", 0.0)
        end_time = segment.get("end_time", start_time + 2.0)
        importance = segment.get("importance", 5)
        seg_type = segment.get("type", "action")
        description = segment.get("description", "")

        # Adaptive sampling: 
        # Dialogue/Importance 10 -> high density (up to 2 FPS)
        # Landscape/Importance 1 -> low density (0.5 FPS)
        if seg_type == "dialogue" or importance >= 8:
            sampling_rate = 1.0 # 1 frame per second
        elif seg_type == "landscape" or importance <= 3:
            sampling_rate = 0.25 # 1 frame per 4 seconds
        else:
            sampling_rate = 0.5 # 1 frame per 2 seconds

        duration = end_time - start_time
        num_frames_to_extract = max(1, int(duration * sampling_rate))
        
        # Fixed interval extraction within segment
        interval = (end_time - start_time) / num_frames_to_extract if num_frames_to_extract > 1 else 0
        
        current_scene: Scene = {
            "scene_index": i,
            "start_frame": int(start_time * fps),
            "end_frame": int(end_time * fps),
            "description": description,
            "frames": []
        }

        for j in range(num_frames_to_extract):
            target_time = start_time + (j * interval)
            frame_idx = int(target_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            # Memory optimization
            h, w = frame.shape[:2]
            if w > 1920 or h > 1080:
                scale = min(1920/w, 1080/h)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            frame_name = f"dir_scene_{i:03d}_frame_{j:02d}.jpg"
            out_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_path, frame)

            current_scene["frames"].append({
                "path": out_path,
                "original_frame_index": frame_idx
            })
            saved_count += 1

        if current_scene["frames"]:
            scenes.append(current_scene)

    cap.release()
    logging.info(f"Director extracted {saved_count} frames across {len(scenes)} segments.")
    
    # Save metadata
    scenes_file = os.path.join(output_dir, "scenes.json")
    with open(scenes_file, "w") as f:
        json.dump({"fps": fps, "scenes": scenes}, f, indent=2)

    return scenes, fps

def get_photos_from_directory(input_dir: str) -> Tuple[List[Scene], float]:
    """
    Retrieve a list of image paths from a directory.
    Treats all photos as a single scene for simplicity, or one scene per photo.
    Let's do one scene per photo.
    """
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    photos = []
    for ext in supported_extensions:
        photos.extend(glob.glob(os.path.join(input_dir, ext)))
        photos.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    photos.sort()
    logging.info(f"Found {len(photos)} photos in {input_dir}.")
    
    scenes: List[Scene] = []
    for i, p in enumerate(photos):
        scenes.append({
            "scene_index": i,
            "start_frame": i,
            "end_frame": i + 1,
            "frames": [{"path": p, "original_frame_index": i}]
        })
    return scenes, 1.0 # 1 fps dummy
