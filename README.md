# NanoGhibli v2.2

NanoGhibli is an intelligent, multimodal AI pipeline that transforms raw video footage into Studio Ghibli-inspired trailers. By combining high-level cinematic analysis with artistic style transfer and generative animation, it creates cohesive, narrative-driven trailers that feel like hand-drawn masterpieces.

## The v2.2 "Director" Pipeline

NanoGhibli v2.2 moves beyond simple computer vision to a **"Director-First"** architecture:

1.  **Phase 0: The Director Phase (`--use_director`)**: Gemini 3.1 Flash "watches" a low-res proxy of your video. It generates a structured **JSON Edit Script** that identifies dialogue, action, and landscapes, assigning importance scores and visual descriptions to each scene.
2.  **Phase 1: Adaptive Extraction**: Instead of fixed frame rates, the pipeline adjusts its sampling density based on the Director's script (e.g., 1.0 FPS for intense dialogue, 0.25 FPS for vast landscapes). This ensures crucial details are captured while minimizing API costs.
3.  **Phase 2: Global Cached Stylization**: Frames are stylized using **Gemini 3.1 Flash Image** or **Gemini 3 Pro Image**. Every frame is MD5-hashed and stored in a global cache (`data/cache/stylized/`). If a frame has been stylized in any previous session, it is reused instantly.
4.  **Phase 3: Perfect Sync Animation**: Stylized frames are bridge-animated using **Veo 3.1 Fast**. The resulting clips are conformed via FFmpeg `setpts` to match the original movie's timing exactly, ensuring dialogue and audio remain perfectly synced.
5.  **Phase 4: Atomic Assembly**: Synced segments are concatenated and merged with the original audio track to produce the final trailer.

## Key Features

- **Dual-Tier Stylization**: Choose between **Nano Banana 2** (Flash) for high-efficiency previews and **Nano Banana Pro** (Gemini 3 Pro) for professional-grade reasoning and artistic fidelity.
- **Multimodal Understanding**: Uses Gemini 3.1's massive context window to "understand" the narrative flow before processing.
- **Content-Addressable Library**: Semantic naming and hashing mean you never pay for the same stylization twice. Cache is model-aware (Flash and Pro versions are cached separately).
- **Perfect Audio Sync**: Per-scene atomic muxing and temporal conforming keep the soundtrack locked to the visuals.
- **Pre-Production Mode (`--skip_video`)**: Finish all expensive vision work and descriptions before committing to the 4-video-per-day Veo limit.
- **Batch API Stylization (`--batch`)**: Route stylization through the Gemini Batch API for **50% cost** and higher rate limits. Async with a 24h SLO (usually minutes). Veo stays synchronous — batch doesn't support video.
- **Quota-Safe Resume**: Every phase writes progress to disk (`scenes.json`, `veo_progress.json`, `batch_jobs.json`). If Veo hits its 10/day cap mid-run, rerun with the same `--session_id X` and the pipeline picks up exactly where it stopped.
- **Cost Transparency**: Detailed session cost estimation at the end of every run, accounting for per-model rates, batch tier, and actual Veo seconds generated.

## Setup

1. **Install Dependencies:**
   NanoGhibli supports CPython 3.11.x. The checked-in `requirements.lock`
   pins the complete runtime and test dependency graph for that interpreter;
   install it from a clean checkout rather than resolving the unconstrained
   input list directly.

   On Ubuntu/Debian, install the native FFmpeg prerequisite first:
   ```bash
   sudo apt-get update
   sudo apt-get install --no-install-recommends -y ffmpeg
   ```
   On macOS, use `brew install ffmpeg`.

   ```bash
   python3.11 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install --requirement requirements.lock
   ```

2. **API Keys:**
   Add your key to a `.env` file:
   ```bash
   GEMINI_API_KEY=your_key
   ```

### Development and verification

The test suite is account-free: it uses temporary files and provider mocks,
and `tests/conftest.py` fails any attempted socket connection. Keep Gemini and
OpenAI credentials unset while running it. The same syntax, import, native
prerequisite, and test gates run for every push and pull request in
`.github/workflows/ci.yml`.

From a clean checkout, run the exact local gates with:

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  ffmpeg -version

env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  .venv/bin/python -c 'import cv2; print("OpenCV", cv2.__version__)'

env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  .venv/bin/python -m compileall -q src tests

env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  PYTHONPATH=src .venv/bin/python - <<'PY'
import importlib

for name in (
    "animator", "batch_stylizer", "director", "extractor", "image_grid",
    "main", "manual_chatgpt", "model_catalog", "models", "pull_batch_results",
    "sequence_trailer", "still_trailer", "stylizer", "submit_stills_batch",
    "trailer_frame_audit", "veo_animator",
):
    importlib.import_module(name)
print("production imports ok")
PY

env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  PYTHONPATH=src .venv/bin/python -m pytest --strict-config --strict-markers -q
```

`requirements.txt` is the human-edited input list. If it changes, regenerate
the lock with the pinned toolchain target and review the resulting diff:

```bash
uv pip compile --universal --python-version 3.11 --no-annotate \
  --output-file requirements.lock requirements.txt
```

The CI job installs FFmpeg with the runner's package manager, installs the
locked Python dependencies, checks OpenCV/FFmpeg availability, and does not
upload caches, input media, or generated outputs.

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

### Batch API Stylization (`--batch`)
Stylize at **50% cost** by routing frames through the Gemini Batch API. The CLI
blocks on polling (SLO 24h, typical completion: a few minutes). Veo stays
synchronous because the Batch API does not support video generation.
```bash
python src/main.py \
  --mode video \
  --input my_movie.mp4 \
  --batch \
  --use_veo \
  --session_id batch_trailer
```
The batch job name is persisted to `data/output/<session_id>/batch_jobs.json`.
Kill and restart the CLI — on the next run it picks up the same job rather than
resubmitting.

### Resuming After a Quota Hit
Every phase hot-writes state to disk:
- `scenes.json` — descriptions + stylized frame paths, updated per scene.
- `veo_progress.json` — per-segment Veo state (`pending`/`veo_done`/`synced`/`failed`).
- `batch_jobs.json` — active/terminated batch jobs keyed by input content.

If Veo hits its **10-per-day** cap (or stylizer hits daily quota), the CLI logs:
```
Daily quota exhausted: …
Rerun with --session_id <id> after quota resets to resume.
```
Tomorrow, rerun with the same `--session_id` and the same inputs — completed
scenes are skipped, pending Veo segments are picked up, and a running batch is
polled to completion instead of resubmitted.

### Batch Ghiblifying Photos
Transform a folder of static images into Studio Ghibli-style art.
```bash
python src/main.py \
  --mode photo \
  --input data/input/my_photos/ \
  --stylizer_model pro \
  --session_id ghibli_collection
```
- **Stylizer Models**: Use `--stylizer_model pro` for the highest quality art (Nano Banana Pro), `--stylizer_model flash` for Nano Banana 2, or `--stylizer_model nano-banana` for the older Gemini 2.5 Flash Image bucket when you need a larger daily image pool.
- **Stylizer Rotation**: Use `--stylizer_models flash,pro-2k,gpt-image-1.5-high-fidelity,pro` to round-robin image generation across model-specific daily buckets. OpenAI rotation entries require `OPENAI_API_KEY`; Gemini entries require `GEMINI_API_KEY`.
- **Manual ChatGPT Fallback**: Add `chatgpt-manual` to the rotation when API buckets are constrained, e.g. `--stylizer_models flash,pro-2k,chatgpt-manual`. The pipeline writes a queue to `data/output/<session_id>/manual_chatgpt/README.md`; save completed ChatGPT Images results into the listed `uploads/` filenames and rerun the same `--session_id` to import them into `stylized_frames/` and the global cache.
- **Smart Anchoring**: Each image is analyzed by Gemini 3.1 Flash Lite to generate a unique visual description before stylization, ensuring lighting and subjects are respected.
- **Global Cache**: Images already stylized in any previous run will be instantly pulled from `data/cache/stylized/` at $0 cost.
- **Output**: Results are saved in `data/output/<session_id>/stylized_frames/`.

### Still-Trailer Assembly and Audit
For trailer experiments built from per-frame stills, use the sequence assembler
to preserve exact timing while replacing missing/title/black frames from the
source. For model outputs that sometimes invent their own black bars, crop the
generated matte away, crop to one fixed active aspect, and place every generated
frame into the same active window. This preserves stable bars without creating a
zoom pulse.
```bash
python src/sequence_trailer.py \
  --preset clean_still_trailer \
  --image_dir data/output/<session_id>/stylized_frames \
  --audio_source data/input/trailer.mp4 \
  --output "data/output/<session_id>/Loganime test v1.mp4" \
  --input_fps 12 \
  --output_fps 12 \
  --expected_frames 1290 \
  --width 1920 \
  --height 1080 \
  --source_image_dir data/input/<trailer>/trailer_12fps_motion_frames \
  --title_source_ranges 49-63 \
  --tail_source_range 1130-1289 \
  --overlay_letterbox_pixels 129 \
  --fill_missing
```

The `clean_still_trailer` preset generalizes the Logan workflow fixes:
generated frames use a stable cover fit, source fallback frames crop embedded
source bars before cover fitting, true-black source frames are preserved instead
of filled from neighboring generated frames, and title/tail source ranges are
merged into the source override list. Use `--overlay_letterbox_pixels` when a
reference frame establishes an exact bar height; this overlays fixed bars after
all fitting so mixed source/generated frames do not pulse or drift.

For the intentional hybrid/glitch version where missing stylized frames flash
back to the real trailer, replace `--fill_missing` with
`--source_flash_for_missing`.

Audit source vs. stylized stills and generate contact sheets:
```bash
python src/trailer_frame_audit.py \
  --source_dir data/input/<trailer>/trailer_12fps_motion_frames \
  --stylized_dir data/output/<session_id>/stylized_frames \
  --output_dir data/output/<session_id>/audit \
  --expected_frames 1290 \
  --contact_ranges 0-69,1118-1209,1200-1289
```

### Image Model Comparison Grid
Run the same prompt library through Nano Banana Pro, Nano Banana 2, and Imagen
4 Fast/Standard/Ultra. The tool writes generated images, a CSV leaderboard, an
HTML report, and a manual upload lane for ChatGPT images created from your
subscription.

Prepare the grid shell and ChatGPT upload instructions without spending API:
```bash
python src/image_grid.py --prepare_only --session_id image_grid_test
```

Run the API-backed grid:
```bash
python src/image_grid.py --session_id image_grid_test
```

Manual ChatGPT comparison flow:
1. Open `data/output/image_grid/<session_id>/manual_uploads/README.md`.
2. Generate each listed prompt in ChatGPT Images.
3. Save each file using the exact requested name, e.g. `kitchen_sign__chatgpt.png`.
4. Rerun scoring/report generation:
   ```bash
   python src/image_grid.py --session_id image_grid_test --score_existing
   ```

The default prompt library lives at `prompts/image_grid.yaml`. Edit it to
add prompt IDs, categories, exact text targets, and subject requirements.

## Architectural Decision: Global vs. Chunked Director

**Decision:** The Director Phase utilizes a single, global video analysis rather than chunking the video into smaller segments.

**Rationale:**
- **Narrative Context:** Gemini 3.1's 1M token context window allows it to analyze up to 15-20 minutes of footage in a single pass. This provides "Editor's Intuition," allowing the model to understand the relationship between early setups and late payoffs in a trailer.
- **Simplicity:** A single global script avoids the complexity of merging overlapping timestamps and deduplicating scenes at chunk boundaries.
- **Cohesion:** Visual anchors generated globally remain more consistent than those generated in isolated 10-second chunks.

*Note: For processing full-length feature films (90+ minutes), a sliding-window chunking strategy is planned for v3.0.*

## Planning

Committed implementation work is tracked in the
[NanoGhibli Linear project](https://linear.app/carteakey/project/nanoghibli-7904a51da414).
Use `TODO.md` only for speculative ideas and research questions; do not mirror
committed Linear issues there.
