import os
import cv2
import glob
import json

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

    frame_count = 0
    saved_count = 0
    prev_frame_gray = None

    scenes = []
    current_scene = None

    while True:
        if max_frames is not None and frame_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        is_scene_cut = False
        is_keyframe = False

        if prev_frame_gray is None:
            is_scene_cut = True
            is_keyframe = True
        else:
            diff = cv2.absdiff(prev_frame_gray, gray)
            mean_diff = diff.mean()

            if mean_diff > scene_threshold:
                is_scene_cut = True
                is_keyframe = True
            elif mean_diff > motion_threshold:
                is_keyframe = True

        if is_scene_cut:
            if current_scene is not None:
                if len(current_scene["frames"]) < 3:
                    # Not enough frames, just continue adding to the current scene instead of cutting
                    is_scene_cut = False
                else:
                    current_scene["end_frame"] = frame_count
                    scenes.append(current_scene)
                    current_scene = None

            if is_scene_cut:
                current_scene = {
                    "scene_index": len(scenes),
                    "start_frame": frame_count,
                    "frames": []
                }

        if is_keyframe and current_scene is not None:
            frame_name = f"frame_{frame_count:06d}.jpg"
            out_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_path, frame)
            
            current_scene["frames"].append({
                "path": out_path,
                "original_frame_index": frame_count
            })
            saved_count += 1
            prev_frame_gray = gray

        frame_count += 1

    if current_scene is not None:
        current_scene["end_frame"] = frame_count
        scenes.append(current_scene)

    cap.release()
    print(f"Extracted {saved_count} frames across {len(scenes)} scenes (total {frame_count} frames).")
    
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
