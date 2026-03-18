# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial VEO model implementation stub (`src/veo_animator.py`).

## [0.1.0] - 2026-03-17
### Added
- `extractor.py` to handle video frame extraction based on motion thresholds and photo loading.
- `stylizer.py` integrating the Google GenAI SDK (Nano Banana 2/Gemini 3.1 Flash Image Preview) for Ghibli-style frame transformation.
- `animator.py` to assemble stylized frames back into an MP4 video using OpenCV.
- `main.py` CLI interface supporting both `--mode video` and `--mode photo`.
- `REQUIREMENTS.md` documenting the core project goals.
- `.gitignore` specifically tuned for Python, virtual environments, and generated data/video files.
