import os
import time
import base64
import argparse
import mimetypes

from google import genai
from google.genai import types

# ------------ Helpers ------------
def read_script(script_arg: str) -> str:
    """if string, use it; if file, read it"""
    if os.path.isfile(script_arg):
        with open(script_arg, "r", encoding="utf-8") as f:
            return f.read()
    return script_arg

def guess_mime(path: str, default="image/jpeg") -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or default

# ------------ Argparse ------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Call VEO-3 API with a script and an edited image to generate a video, then download to local file."
    )
    ap.add_argument("--api-key", required=True, help="Google API key for genai.")
    ap.add_argument("--script", required=True,
                    help="Script content OR a path to a script file. If a file exists at this path, its content is used.")
    ap.add_argument("--image-path", required=True, help="Path to the edited image (jpeg/png/webp).")
    ap.add_argument("--output", default="veo3_with_image_input.mp4",
                    help="Output video filename (default: veo3_with_image_input.mp4)")
    ap.add_argument("--model", default="veo-3.0-generate-preview",
                    help="Model name (default: veo-3.0-generate-preview)")
    ap.add_argument("--aspect-ratio", default="16:9",
                    help="Aspect ratio (default: 16:9)")
    ap.add_argument("--num-videos", type=int, default=1,
                    help="Number of videos to generate (default: 1)")
    ap.add_argument("--person-generation", default="allow_adult",
                    choices=["allow_adult", "block_all"],
                    help="Person generation policy (default: allow_adult)")
    ap.add_argument("--poll-interval", type=int, default=10,
                    help="Seconds between polling attempts (default: 10)")
    ap.add_argument("--poll-timeout", type=int, default=1800,
                    help="Maximum seconds to wait for completion (default: 1800)")
    return ap.parse_args()

# ------------ Main ------------
def main():
    args = parse_args()

    # 1) Client
    client = genai.Client(api_key=args.api_key)

    # 2) Script prompt
    user_script = read_script(args.script)
    prompt_template = """
Follow this script: [script]


** End of script
Remember:
1. No camera move
2. show lip sync
3. don't show any caption
"""
    prompt = prompt_template.replace("[script]", user_script)

    # 3) Image bytes + mime
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()
    mime_type = guess_mime(args.image_path, default="image/jpeg")

    # 4) Submit job
    op = client.models.generate_videos(
        model=args.model,
        prompt=prompt,
        image=types.Image(image_bytes=image_bytes, mime_type=mime_type),
        config=types.GenerateVideosConfig(
            aspect_ratio=args.aspect_ratio,
            number_of_videos=args.num_videos,
            person_generation=args.person_generation,
        ),
    )

    # 5) Polling
    start = time.time()
    while not op.done:
        elapsed = int(time.time() - start)
        if elapsed > args.poll_timeout:
            raise TimeoutError(f"Timeout waiting for video generation (> {args.poll_timeout}s)")
        print(f"Waiting for video generation to complete... (elapsed: {elapsed}s)")
        time.sleep(args.poll_interval)
        op = client.operations.get(op)

    # 6) Download
    if not getattr(op, "response", None) or not getattr(op.response, "generated_videos", None):
        raise RuntimeError("Operation finished but no generated videos found in response.")
    if not op.response.generated_videos:
        raise RuntimeError("Empty generated_videos list.")

    video = op.response.generated_videos[0]
    # NOTE: google-genai SDK returns a File-like handle in video.video
    # Use the client helper to download, then save to disk.
    file_obj = client.files.download(file=video.video)  # returns a BytesIO-like object
    # some versions return wrapper（with .save），some return bytes
    out_path = args.output
    try:
        # Newer SDKs often expose a .save() method
        video.video.save(out_path)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: write bytes from download result
        # file_obj might have .read()，or might be bytes
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        with open(out_path, "wb") as f:
            f.write(data)

    print(f"Generated video saved to {out_path}")

if __name__ == "__main__":
    main()
