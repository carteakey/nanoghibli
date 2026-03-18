import os
from google import genai
from PIL import Image
from tqdm import tqdm

# We will read GEMINI_API_KEY from environment (handled in main.py via dotenv)

GHIBLI_PROMPT = (
    "Modify this image into a Studio Ghibli homage. "
    "Keep the exact same structural composition, objects, and subjects, but render it in a beautiful, "
    "hand-drawn anime style reminiscent of classic 90s animation. "
    "Use lush, vibrant watercolor backgrounds, distinct clean character/object outlines, and soft, magical lighting. "
    "This is strictly an homage to visualize how things would look in that style. "
    "CRITICAL: You must follow the source image exactly. Do not hallucinate, invent, or add any new elements, objects, text, or details that are not explicitly present in the original image. Do not fill in any blanks."
)

def stylize_frames(frames, output_dir: str):
    """
    Takes a list of frame dicts (e.g. [{"path": ..., "original_frame_index": ...}])
    and stylizes each using the Gemini API.
    Saves outputs in output_dir.
    Returns a list of stylized frame dicts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = genai.Client()
    
    stylized_frames = []

    print(f"Stylizing {len(frames)} frames...")
    
    for i, frame_info in enumerate(tqdm(frames)):
        input_path = frame_info["path"]
        orig_index = frame_info.get("original_frame_index", i)
        
        try:
            image = Image.open(input_path)
            
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[GHIBLI_PROMPT, image],
            )
            
            # The response could contain multiple parts, we look for the image
            saved = False
            for part in response.parts:
                if getattr(part, 'thought', False):
                    continue # Skip thoughts
                
                if part.inline_data is not None:
                    out_img = part.as_image()
                    out_name = f"stylized_{orig_index:06d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    out_img.save(out_path)
                    
                    stylized_frames.append({
                        "path": out_path,
                        "original_frame_index": orig_index
                    })
                    saved = True
                    break # Assuming one image per response
            
            if not saved:
                print(f"Warning: No image returned for {input_path}")
                
        except Exception as e:
            print(f"Error stylizing {input_path}: {e}")

    return stylized_frames
