# NanoGhibli

NanoGhibli is a Python-based pipeline that automatically converts videos or folders of photos into beautifully stylized, Studio Ghibli-inspired trailers. 

It accomplishes this by using computer vision to detect scenes and keyframes, prompting the **Nano Banana 2 (Gemini 2.5 Flash Image)** model for artistic stylization, and utilizing the **Veo 3.1 Fast** model for cinematic scene interpolation. It even multiplexes the original audio track back onto the final generation.

## Features

- **Scene Detection:** Uses histogram correlation and motion thresholds via OpenCV to accurately split input videos into distinct scenes.
- **Stylization:** Converts extracted frames into vibrant watercolor Ghibli-style art using the Nano Banana 2 API.
- **Cinematic Animation:** Generates fluid transitions for each scene using the Veo 3.1 Fast preview model.
- **Resumable Pipeline:** Saves progress locally to a unique `session_id`. If you hit an API limit, simply re-run the command to pick up exactly where you left off without wasting quota.
- **Audio Syncing:** Automatically extracts the original audio and overlays it onto the synthesized video.

## Setup

1. **Install Dependencies:**
   Ensure you have Python 3 and `ffmpeg` installed on your system. 
   Then, install the Python requirements:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install opencv-python google-genai pillow python-dotenv tqdm
   ```

2. **API Keys:**
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage Examples

Run the scripts from the root directory within your activated virtual environment.

### 1. Process a full video with Veo Interpolation
Detects scenes, stylizes frames, animates the scenes with Veo, and outputs the final video with original audio synced.

```bash
python src/main.py \
  --mode video \
  --input path/to/your_video.mp4 \
  --use_veo \
  --session_id my_awesome_video
```

### 2. Process just a 30-second snippet
Perfect for testing out prompts or saving API costs before running an entire movie.

```bash
python src/main.py \
  --mode video \
  --input path/to/your_video.mp4 \
  --max_duration 30 \
  --use_veo \
  --session_id short_test
```

### 3. Resume an interrupted session
If your script fails (e.g., due to a `429 RESOURCE_EXHAUSTED` rate limit error from the Veo API), simply run the exact same command. The script will detect the existing `session_id`, skip the expensive extraction and stylization steps, and resume generating the remaining Veo scenes!

```bash
python src/main.py \
  --mode video \
  --input path/to/your_video.mp4 \
  --max_duration 30 \
  --use_veo \
  --session_id short_test
```

### 4. Simple Assembly (No Veo)
If you don't want to use the heavy Veo 3.1 video generation model, you can run the pipeline without the `--use_veo` flag. This will simply stitch the stylized keyframes together like a traditional flipbook animation.

```bash
python src/main.py \
  --mode video \
  --input path/to/your_video.mp4 \
  --session_id flipbook_test
```

### 5. Stylize a folder of Photos
You can also run NanoGhibli on a folder of static images to batch stylize them!

```bash
python src/main.py \
  --mode photo \
  --input data/input/photos_folder/
```

Outputs will be saved in `data/output/<session_id>/`.
