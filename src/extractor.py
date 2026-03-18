import os
import cv2
import glob
import json
from scenedetect import detect, ContentDetector

def extract_scenes_from_video(video_path: str, output_dir: str, motion_threshold: float = 20.0, scene_threshold: float = 35.0, max_duration: float = None):
    """
    Extract frames from a video and group them into scenes based on motion/scene changes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenes_file = os.path.join(output_dir, "scenes.json")
    if os.path.exists(scenes_file):
        print(f"Loading existing scenes from {scenes_file}...")
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

    print(f"Detecting scene boundaries in {video_path}...")
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

    print(f"Found {len(scene_list_frames)} scenes using PySceneDetect.")

    saved_count = 0
    scenes = []

    for i, (start_f, end_f) in enumerate(scene_list_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        current_scene = {
            "scene_index": i,
            "start_frame": start_f,
            "end_frame": end_f,
            "frames": []
        }
        
        prev_frame_gray = None
        for frame_count in range(start_f, end_f):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    print(f"Extracted {saved_count} frames across {len(scenes)} scenes.")
    
    # Save metadata
    with open(scenes_file, "w") as f:
        json.dump({"fps": fps, "scenes": scenes}, f, indent=2)
        
    return scenes, fps

def get_photos_from_directory(input_dir: str):
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
    print(f"Found {len(photos)} photos in {input_dir}.")
    
    scenes = []
    for i, p in enumerate(photos):
        scenes.append({
            "scene_index": i,
            "start_frame": i,
            "end_frame": i + 1,
            "frames": [{"path": p, "original_frame_index": i}]
        })
    return scenes, 1.0 # 1 fps dummy
