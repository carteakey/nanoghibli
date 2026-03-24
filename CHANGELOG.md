# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-03-23
### Added
- **Dual-Tier Stylization:** Support for both `flash` (Nano Banana 2) and `pro` (Nano Banana Pro) models via the `--stylizer_model` flag.
- **Nano Banana Pro Support:** Integration of `gemini-3-pro-image-preview` for high-fidelity reasoning and professional-grade Studio Ghibli art.
- **Model-Aware Caching:** Global cache now separates assets by model tier, ensuring Flash and Pro versions of the same frame are cached independently.
- **Content-Addressable Library:** Implemented MD5-based global frame caching in `data/cache/stylized/`. Stylizing the same frame twice now costs $0 across all sessions.
- **Semantic Segment Naming:** Synced video segments are now named by video slug and timestamps (`dune_005000_010000_sync.mp4`), enabling cross-run asset sharing and effortless merging.
- **Pre-Production Mode:** Added `--skip_video` flag to allow users to finish all stylization and caching before committing to expensive video generation runs.
- **Maximum Context VEO:** Updated VEO prompts to reference the entire sequence of stylized frames, improving narrative flow in generated scenes.
- **Cost Reporting:** Real-time session cost estimation printed at the end of every run, with tier-specific pricing ($0.05 for Pro images).

## [2.1.0] - 2026-03-23
### Added
- **Perfect Audio Sync:** Moved from global muxing to per-scene atomic audio slicing. Audio is now perfectly aligned with dialogue timestamps.
- **Temporal Conforming:** Implemented FFmpeg `setpts` filtering to conform Veo generated videos (4s/8s) exactly to the original scene duration from the script.

## [2.0.0] - 2026-03-23
### Added
- **Multimodal Director Phase:** Added a pre-scan phase where Gemini 3.1 Flash "watches" the video to create an intelligent Edit Script.
- **Content-Aware Extraction:** Replaced blind motion detection with adaptive sampling based on scene type (e.g., higher FPS for dialogue, lower for landscapes).
- **Intelligent Anchoring:** Director-generated descriptions are now used as the primary visual anchors for the entire pipeline, ensuring narrative and visual continuity.
- **Model Upgrade:** Fully migrated all vision and analysis tasks to Gemini 3.1 (Flash Lite and Flash Image Preview).
- **Video Proxies:** Implemented high-speed FFmpeg proxy creation to minimize upload times and token costs for long video analysis.

## [1.1.0] - 2026-03-23
### Added
- **Smart Prompting:** Added a "Director" phase that uses Gemini 2.0 Flash to generate a 15-word description of each scene before stylization.
- **Scene Consistency:** The generated scene description is now injected into the Ghibli prompt for every frame in that scene, significantly improving character and environmental detail consistency.
- **Seek for Light:** Implemented intelligent start-frame detection that scans the beginning of each scene to skip black or empty frames, ensuring Veo starts with high-quality visual context.
- **Metadata Persistence:** Scene descriptions are now saved to `scenes.json` to support session resumes.
- **Enhanced VEO Context:** The VEO animator now incorporates the scene description into its cinematic generation prompt.

## [1.0.0] - 2026-03-23
### Added
- **Batch Processing:** Support for multiple video or photo directory inputs in a single run via CLI.
- **Comprehensive Error Handling:** Robust handling for API rate limits (`429 RESOURCE_EXHAUSTED`) with exponential backoff and network timeouts.
- **Memory Optimization:** Automatic frame resizing for high-resolution (4K/1080p) videos to reduce memory footprint.
- **Configuration System:** Introduced `config.yaml` for central management of model parameters (temperature, top_p, etc.) and processing defaults.
- **Progress Tracking:** Integrated `tqdm` progress bars across extraction, stylization, and VEO generation phases.
- **GIF Output Support:** Added high-quality GIF generation with adaptive palette support via the `--output_format` flag.
- **Unit Testing Suite:** Added comprehensive tests for `extractor.py` and `animator.py` in the `tests/` directory.

## [0.1.0] - 2026-03-17
### Added
- `extractor.py` to handle video frame extraction based on motion thresholds and photo loading.
- `stylizer.py` integrating the Google GenAI SDK (Nano Banana 2/Gemini 3.1 Flash Image Preview) for Ghibli-style frame transformation.
- `animator.py` to assemble stylized frames back into an MP4 video using OpenCV.
- `main.py` CLI interface supporting both `--mode video` and `--mode photo`.
- `REQUIREMENTS.md` documenting the core project goals.
- `.gitignore` specifically tuned for Python, virtual environments, and generated data/video files.
