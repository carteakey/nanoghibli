# TODO

This file is for speculative ideas and research questions. Committed work is
tracked in the [NanoGhibli Linear project](https://linear.app/carteakey/project/nanoghibli-7904a51da414).

## High Priority
- [x] Implement VEO model integration for smoother animation transitions between stylized frames.
- [x] Add comprehensive error handling for API rate limits and network timeouts.
- [x] Allow batch processing of multiple videos or photo directories in one run.
- [x] Implement Dynamic Scene Selection (The Director Phase) via Multimodal Analysis.

## Medium Priority
- [x] Add a progress bar to the frame extraction phase for large videos.
- [x] Expose Gemini model parameters (like temperature) via CLI arguments for advanced tweaking.
- [x] Optimize memory usage when processing high-resolution input files.
- [x] Add session cost tracking and usage metrics reporting.

## Low Priority / Polish
- [x] Add unit tests for `extractor.py` and `animator.py`.
- [x] Support generating GIFs alongside MP4 outputs.
- [x] Create a dedicated configuration file (e.g., `config.yaml`) for default settings.

## Future Research

- Evaluate whether 90+ minute source videos need a sliding-window Director
  strategy, and how overlapping windows could preserve narrative context and
  visual anchors without duplicating scenes or multiplying API cost.
