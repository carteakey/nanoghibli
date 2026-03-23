from typing import TypedDict, List

class FrameInfo(TypedDict):
    path: str
    original_frame_index: int

class Scene(TypedDict, total=False):
    scene_index: int
    start_frame: int
    end_frame: int
    frames: List[FrameInfo]
    stylized_frames: List[FrameInfo]
    description: str
