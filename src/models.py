from typing import TypedDict, List

class UsageMetrics:
    def __init__(self, model_tier="flash"):
        self.input_tokens = 0
        self.output_tokens = 0
        self.images_processed = 0
        self.videos_generated = 0
        self.descriptions_generated = 0
        self.model_tier = model_tier

    def add_usage(self, response):
        if hasattr(response, 'usage_metadata'):
            self.input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
            self.output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)

    def __str__(self):
        # Estimated pricing (approximate)
        # Gemini 3.1 Flash-Lite: $0.25 / 1M input, $1.50 / 1M output
        # Gemini 3.1 Flash Image: ~$0.067 per 1K image
        # Gemini 3 Pro Image: ~$0.135 per 1K/2K image
        # Veo 3.1 Fast: $0.15 per second (assuming 4s videos = $0.60 per video)
        
        flash_cost = (self.input_tokens / 1_000_000 * 0.25) + (self.output_tokens / 1_000_000 * 1.50)
        
        if self.model_tier == "pro":
            image_cost = self.images_processed * 0.135
        else:
            image_cost = self.images_processed * 0.067
            
        veo_cost = self.videos_generated * 0.60
        total_cost = flash_cost + image_cost + veo_cost
        
        return (
            f"\n{'='*30}\n"
            f"   ESTIMATED SESSION COST ({self.model_tier.upper()})\n"
            f"{'='*30}\n"
            f"Descriptions: {self.descriptions_generated}\n"
            f"Images Stylized: {self.images_processed}\n"
            f"Veo Videos: {self.videos_generated}\n"
            f"Tokens: {self.input_tokens} In / {self.output_tokens} Out\n"
            f"{'-'*30}\n"
            f"Total Est. Cost: ${total_cost:.4f}\n"
            f"{'='*30}\n"
        )

class FrameInfo(TypedDict):
    path: str
    original_frame_index: int

class QuotaExceededError(Exception):
    pass

class Scene(TypedDict, total=False):
    scene_index: int
    start_frame: int
    end_frame: int
    frames: List[FrameInfo]
    stylized_frames: List[FrameInfo]
    description: str
