from openai import OpenAI
import os
import subprocess
from glob import glob
import base64
import json
import shutil
import time
import argparse

# ---------- OpenAI Client ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")

# ---------- Export (downstream) default paths ----------
DEFAULT_EXPORT_DIR = "export_for_nano"

# ---------- Utilities ----------
def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_text(path, content):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content is not None else "")

def copy_image(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)

# ---------- Step 1: Extract frames ----------
def extract_frames(video_path, output_folder, fps=1):
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{output_folder}/frame_%04d.jpg"
    ]

    subprocess.run(cmd, check=True)
    print(f"Frames saved to {output_folder}")

# ---------- Step 2: Image selection (single-pick each round) ----------
def select_best_frame(images, selection_prompt):
    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in images]
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert visual assistant."},
            {"role": "user", "content": [{"type": "text", "text": selection_prompt}] + image_contents}
        ],
        max_tokens=500
    )
    content = response.choices[0].message.content or ""
    selected_file = None
    for img in images:
        filename = os.path.basename(img)
        if filename in content:
            selected_file = img
            break
    if selected_file:
        print(f"Selected frame (raw): {selected_file}")
        return selected_file
    else:
        raise ValueError("Could not find selected frame filename in GPT-4V response.")

def pick_frames_progressive(topk, images, base_prompt_single, max_attempts=9):
    kept = []
    attempt = 0
    dynamic_prompt = base_prompt_single

    while len(kept) < topk and attempt < max_attempts:
        attempt += 1
        print(f"\n🔁 Progressive pick attempt {attempt}/{max_attempts}")
        try:
            picked = select_best_frame(images, dynamic_prompt)
        except ValueError:
            dynamic_prompt += "\nNote: Please reply with an exact filename (e.g., frame_0007.jpg)."
            continue

        # avoid picking the same frame
        if picked in kept:
            banned = ", ".join(os.path.basename(p) for p in kept)
            dynamic_prompt += f"\nDo NOT select these filenames again: {banned}"
            print(f"[SKIP] Model repeated a kept frame: {os.path.basename(picked)}")
            continue

        ok, fb = evaluate_output(picked, prompt_type="character")
        if ok:
            kept.append(picked)
            print(f"✅ Accepted {len(kept)}/{topk}: {os.path.basename(picked)}")
            # upload banned list
            banned = ", ".join(os.path.basename(p) for p in kept)
            dynamic_prompt += f"\nDo NOT select these filenames again: {banned}"
        else:
            print(f"❌ Candidate failed evaluation: {os.path.basename(picked)}\n{fb}")
            dynamic_prompt += f"\n[Previous feedback]: {fb}\nPlease pick a different frame addressing the issues above."

    return kept

# ---------- Step 3: Describe image ----------
def describe_image(image_file, prompt):
    b64_image = encode_image_base64(image_file)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for visual description."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# ---------- Step 4: Summarize long video ----------
def load_transcript(transcript_file):
    with open(transcript_file, "r", encoding="utf-8") as f:
        return f.read()

def summarize_long_video(keyframe_images, transcript, prompt):
    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in keyframe_images]
    messages = [
        {"role": "system", "content": "You are a helpful assistant for video summarization."},
        {"role": "user", "content": [
            {"type": "text", "text": f"{prompt}\n\nHere is the transcript:\n{transcript}"}] + image_contents}
    ]
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        max_tokens=1500
    )
    return response.choices[0].message.content

# ---------- Step 5: Generate ad script ----------
def generate_ad_script(object_desc, character_desc, long_video_summary, object_image, character_image, long_video_frames, additional_prompt=""):
    b64_object_image = encode_image_base64(object_image)
    b64_character_image = encode_image_base64(character_image)
    b64_long_video_frames = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in long_video_frames]
    
    prompt = f"""
    Here is the product description: {object_desc}
    Here is the main character description: {character_desc}
    Here is the long video summary: {long_video_summary}

    {additional_prompt}

    Please write a creative, natural advertising script less than 100 words that integrates the product smoothly into the long video's context,
    matching the scene, emotion, and character style. The script may include scenes, actions, and words.

    Script example: 
    **[Scene begins in the vastness of space, an astronaut (as shown in image) floats]**

    **Astronaut says:** In the cosmos, every second matters.

    **[Close-up on the watch glistening.]**

    **Astronaut says:** Face this unwavering precision. 

    **Astronaut says:** Timeless durability, bridging worlds and hearts.

    Please write a script that contains 2 scenes!!!
    """

    messages = [
        {"role": "system", "content": "You are a professional advertising scriptwriter."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": b64_object_image}},
            {"type": "image_url", "image_url": {"url": b64_character_image}}
        ] + b64_long_video_frames}
    ]
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ---------- Step 6: Relation ----------
def generate_character_object_relation_paragraph(character_desc: str, object_desc: str, ad_script: str) -> str:
    prompt = f"""
            Based on the following information, write a concise paragraph (don't need to be more than 50 words) in English
            that describes the **initial staging of the opening shot** of an advertisement:

            - Specify the physical relation between the character and the object 
            (e.g., holding, leaning against, wearing, standing beside, presenting).
            - Ensure the relation is physically plausible, visually clear, and smoothly transitions into Scene 1 of the script.
            - Do not list points. Output only one cohesive paragraph.
            - In your description, call the character "the character in the first image", and call the object "the object in the second image" or "the [name of the object] in the second image".
            - Don't need to describe the appearance of the character or the object, just describe how they are physically related.
            - A good example: "The character in the first image holds the bvlgari necklace in the second image."

            [Character description]
            {character_desc}

            [Object description]
            {object_desc}

            [Advertising script]
            {ad_script}
            """
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise film staging and cinematography assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return (response.choices[0].message.content or "").strip()

# ---------- Step 7: Evaluation ----------
def evaluate_output(content, prompt_type):
    if prompt_type == "character":
        prompt = """
        Please evaluate the following image of a character.
        Is the character's face clearly visible and does it represent a main character in a film or TV show?
        Reply with "PASS" if it meets the criteria, or "FAIL" followed by suggestions to improve.
        """
        b64_image = encode_image_base64(content)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict reviewer of character portrait quality."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_image}}
                ]}
            ],
            max_tokens=300
        )
    elif prompt_type == "script":
        prompt = """
        Please evaluate the following advertising script:
        Does the script naturally and creatively integrate the product into the story world?
        Is it emotionally consistent with the described character and scene?
        Reply with "PASS" if it meets the criteria, or "FAIL" followed by suggestions to improve.
        """
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict evaluator of advertising copy quality."},
                {"role": "user", "content": f"{prompt}\n\nAdvertising Script:\n{content}"}
            ],
            max_tokens=300
        )
    else:
        raise ValueError("Unknown prompt_type passed to evaluate_output")

    result = response.choices[0].message.content.strip()
    passed = "PASS" in result
    feedback = "" if passed else result
    return passed, feedback

# ---------- Argparse ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Upstream pipeline: progressive select (top-k) character frames, describe images, summarize long video, and generate ad script; then export for downstream.")
    parser.add_argument("--long-video-path", required=True, help="Path to the long video to extract frames from.")
    parser.add_argument("--object-image", required=True, help="Path to the product/object image (will become img2).")
    parser.add_argument("--transcript-file", required=True, help="Path to transcript text file of the long video.")
    parser.add_argument("--output-dir", required=True, help="Directory to store extracted frames and intermediates.")
    # optional
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Directory to export img1/img1_*, img2, prompt for downstream.")
    parser.add_argument("--fps", type=float, default=0.5, help="FPS to extract frames from the long video.")
    parser.add_argument("--topk", type=int, default=1, help="How many character frames to export (1 keeps legacy behavior).")
    parser.add_argument("--max-pick-attempts", type=int, default=9, help="Total attempts for progressive picking (global).")
    return parser.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()

    long_video_path = args.long_video_path
    object_image = args.object_image
    transcript_file = args.transcript_file
    output_dir = args.output_dir
    export_dir = args.export_dir
    topk = max(1, int(args.topk))
    max_pick_attempts = max(1, int(args.max_pick_attempts))

    export_img1_single = os.path.join(export_dir, "img1.jpg")   # legacy
    export_img2 = os.path.join(export_dir, "img2.jpg")
    export_prompt = os.path.join(export_dir, "prompt.txt")
    export_desc_dir = os.path.join(export_dir, "descriptions")
    export_summary_dir = os.path.join(export_dir, "summary")
    export_relation_txt = os.path.join(export_dir, "character_object_relation.txt")
    export_meta_path = os.path.join(export_dir, "meta.json")

    print("✅ Step 1: start")

    # Divide to frames
    long_frames_dir = os.path.join(output_dir, "long_video")
    extract_frames(long_video_path, long_frames_dir, fps=args.fps)
    long_frames = sorted(glob(f"{long_frames_dir}/*.jpg"))

    # Progressive selection (unified for topk==1 or >1)
    base_prompt_single = (
        "From these frames, please select EXACTLY ONE filename (e.g., frame_0003.jpg) "
        "that shows the main character with a face or portrait. "
        "Reply ONLY with the filename, and nothing else."
    )
    candidate_pool = long_frames[:20] if len(long_frames) > 20 else long_frames

    picked_frames = pick_frames_progressive(
        topk=topk,
        images=candidate_pool,
        base_prompt_single=base_prompt_single,
        max_attempts=max_pick_attempts
    )
    if not picked_frames:
        raise RuntimeError("Failed to find any valid character frame after multiple attempts.")
    if len(picked_frames) < topk:
        print(f"[WARN] Only {len(picked_frames)} frame(s) accepted within {max_pick_attempts} attempts (requested topk={topk}).")

    print("✅ Step 2: character frames selected →", [os.path.basename(p) for p in picked_frames])

    # Descriptions (object + first character frame as anchor)
    object_desc = describe_image(object_image, "Please describe the key product or object in this image.")
    print("Object description:\n", object_desc)

    character_desc = describe_image(picked_frames[0], "Please describe the main character in this image.")
    print("Character description:\n", character_desc)
    print("✅ Step 3: descriptions done")

    # Long video summary
    transcript = load_transcript(transcript_file)
    summary_prompt = "Please summarize the main storyline, scene, characters, emotions, and environment from these video keyframes and transcript."
    long_video_summary = summarize_long_video(long_frames, transcript, summary_prompt)

    # Generate ad script (with eval+retry) — still anchor on first frame
    max_script_attempts = 2
    additional_prompt = ""
    ad_script = None
    passed = False

    for i in range(max_script_attempts):
        print(f"\n📝 Attempt {i+1} generating ad script...")
        ad_script = generate_ad_script(
            object_desc, character_desc, long_video_summary,
            object_image, picked_frames[0], long_frames,
            additional_prompt
        )
        passed, feedback = evaluate_output(ad_script, prompt_type="script")
        if passed:
            print("✅ Script passed evaluation!")
            break
        else:
            print("❌ Script failed evaluation:\n", feedback)
            additional_prompt += f"\n[Evaluator feedback]: {feedback}"

    if not ad_script or not passed:
        print("Failed to generate a valid ad script after several attempts.")

    print("\nGenerated Advertising Script:\n", ad_script)
    print("✅ Step 4: script generated")

    # Relation paragraph (also anchor on first frame description)
    print("🎬 Generating character-object relation paragraph...")
    relation_paragraph = generate_character_object_relation_paragraph(
        character_desc=character_desc,
        object_desc=object_desc,
        ad_script=ad_script or ""
    )
    print("Generated relation paragraph:\n", relation_paragraph)

    # ---------- Export for downstream ----------
    ensure_dir(export_dir)

    if topk == 1:
        for p in glob(os.path.join(export_dir, "img1_*.jpg")):
            try:
                os.remove(p)
            except OSError:
                pass

    # Always export legacy single
    copy_image(picked_frames[0], export_img1_single)
    # Multi-frame extra exports
    img1_multi = []
    if topk > 1:
        for i, src in enumerate(picked_frames, start=1):
            dst = os.path.join(export_dir, f"img1_{i:02d}.jpg")
            copy_image(src, dst)
            img1_multi.append(dst)

    copy_image(object_image, export_img2)        # img2
    save_text(export_prompt, ad_script or "")    # prompt
    save_text(export_relation_txt, relation_paragraph)  # relation

    # Optional intermediates
    save_text(os.path.join(export_desc_dir, "object_desc.txt"), object_desc)
    save_text(os.path.join(export_desc_dir, "character_desc.txt"), character_desc)
    save_text(os.path.join(export_summary_dir, "long_video_summary.txt"), long_video_summary)

    meta = {
        "export_time": int(time.time()),
        "export_dir": os.path.abspath(export_dir),
        "img1_path": os.path.abspath(export_img1_single),  # legacy single
        "img1_paths": [os.path.abspath(p) for p in img1_multi],  # multi list (may be empty)
        "img2_path": os.path.abspath(export_img2),
        "prompt_path": os.path.abspath(export_prompt),
        "character_object_relation_path": os.path.abspath(export_relation_txt),
        "character_image_sources": [os.path.abspath(p) for p in picked_frames],
        "object_image_source": os.path.abspath(object_image),
        "long_video_path": os.path.abspath(long_video_path),
        "transcript_file": os.path.abspath(transcript_file),
        "frames_dir": os.path.abspath(long_frames_dir),
        "num_frames": len(long_frames),
        "fps": args.fps,
        "topk": topk
    }
    save_text(export_meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    print("\n🎯 Exported for downstream:")
    print(f"  img1 (legacy) → {os.path.relpath(export_img1_single)}")
    if img1_multi:
        for pth in img1_multi:
            print(f"  img1_* → {os.path.relpath(pth)}")
    print(f"  img2 → {os.path.relpath(export_img2)}")
    print(f"  prompt → {os.path.relpath(export_prompt)}")
    print(f"  character_object_relation → {os.path.relpath(export_relation_txt)}")
    print(f"  meta → {os.path.relpath(export_meta_path)}")

if __name__ == "__main__":
    print("Running main")
    main()
