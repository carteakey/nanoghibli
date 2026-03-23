# NanoGhibli v2.2

NanoGhibli is an intelligent, multimodal AI pipeline that transforms raw video footage into Studio Ghibli-inspired trailers. By combining high-level cinematic analysis with artistic style transfer and generative animation, it creates cohesive, narrative-driven trailers that feel like hand-drawn masterpieces.

## The v2.2 "Director" Pipeline

NanoGhibli v2.2 moves beyond simple computer vision to a **"Director-First"** architecture:

1.  **Phase 0: The Director Phase (`--use_director`)**: Gemini 3.1 Flash "watches" a low-res proxy of your video. It generates a structured **JSON Edit Script** that identifies dialogue, action, and landscapes, assigning importance scores and visual descriptions to each scene.
2.  **Phase 1: Adaptive Extraction**: Instead of fixed frame rates, the pipeline adjusts its sampling density based on the Director's script (e.g., 1.0 FPS for intense dialogue, 0.25 FPS for vast landscapes). This ensures crucial details are captured while minimizing API costs.
3.  **Phase 2: Global Cached Stylization**: Frames are stylized using **Gemini 3.1 Flash Image**. Every frame is MD5-hashed and stored in a global cache (`data/cache/stylized/`). If a frame has been stylized in any previous session, it is reused instantly.
4.  **Phase 3: Perfect Sync Animation**: Stylized frames are bridge-animated using **Veo 3.1 Fast**. The resulting clips are conformed via FFmpeg `setpts` to match the original movie's timing exactly, ensuring dialogue and audio remain perfectly synced.
5.  **Phase 4: Atomic Assembly**: Synced segments are concatenated and merged with the original audio track to produce the final trailer.

## Key Features

- **Multimodal Understanding**: Uses Gemini 3.1's massive context window to "understand" the narrative flow before processing.
- **Content-Addressable Library**: Semantic naming and hashing mean you never pay for the same stylization twice.
- **Perfect Audio Sync**: Per-scene atomic muxing and temporal conforming keep the soundtrack locked to the visuals.
- **Pre-Production Mode (`--skip_video`)**: Finish all expensive vision work and descriptions before committing to the 4-video-per-day Veo limit.
- **Cost Transparency**: Detailed session cost estimation provided at the end of every run.

## Setup

1. **Install Dependencies:**
   Ensure you have Python 3 and `ffmpeg` installed. 
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys:**
   Add your key to a `.env` file:
   ```bash
   GEMINI_API_KEY=your_key
   ```

## Usage

### The "Director" Production Run
This is the recommended way to run the pipeline for maximum quality and consistency.
```bash
python src/main.py \
  --mode video \
  --input my_movie.mp4 \
  --use_director \
  --use_veo \
  --session_id final_trailer
```

### Pre-Production (Caching Only)
Use this to prepare all stylized frames when you are out of Veo video generation quota.
```bash
python src/main.py \
  --mode video \
  --input my_movie.mp4 \
  --use_director \
  --skip_video \
  --session_id preprod_run
```

## Architectural Decision: Global vs. Chunked Director

**Decision:** The Director Phase utilizes a single, global video analysis rather than chunking the video into smaller segments.

**Rationale:**
- **Narrative Context:** Gemini 3.1's 1M token context window allows it to analyze up to 15-20 minutes of footage in a single pass. This provides "Editor's Intuition," allowing the model to understand the relationship between early setups and late payoffs in a trailer.
- **Simplicity:** A single global script avoids the complexity of merging overlapping timestamps and deduplicating scenes at chunk boundaries.
- **Cohesion:** Visual anchors generated globally remain more consistent than those generated in isolated 10-second chunks.

*Note: For processing full-length feature films (90+ minutes), a sliding-window chunking strategy is planned for v3.0.*
