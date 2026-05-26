from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .model_task_benchmark import _extract_json_block
from .model_runners import GenerationConfig, build_runner
from .planner_veo_e2e import _estimate_stage_costs, _evaluate_video_ad, _load_pricing, _save_json, _usage_to_plain_dict
from .reconstruction_local_eval import combine_skillbench_scores, evaluate_ocr_text_fidelity, evaluate_xclip_alignment
from .reconstruction_metrics import compute_reconstruction_metrics

from demo_pipe1 import encode_image_base64, extract_frames


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> tuple[int, str, str, float]:
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    return proc.returncode, proc.stdout, proc.stderr, time.time() - start


def _extract_eval_frames(video_path: str, out_dir: str, fps: float = 1.0, max_frames: int = 8) -> List[str]:
    extract_frames(video_path, out_dir, fps=fps)
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))[:max_frames]


def _select_frame(client: OpenAI, frames: List[str], prompt: str) -> str:
    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in frames[:10]]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise visual selector for advertisement video frames."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents},
        ],
        max_tokens=300,
    )
    raw = response.choices[0].message.content or ""
    for frame in frames:
        if os.path.basename(frame) in raw:
            return frame
    return frames[min(len(frames) // 2, len(frames) - 1)]


def _select_joint_reference_frame(client: OpenAI, frames: List[str]) -> str:
    return _select_frame(
        client,
        frames,
        (
            "Select the single best frame that jointly shows the main character, the product or branded object, "
            "and their relationship in a visually clear advertisement composition. Reply with the exact filename only."
        ),
    )


def _select_opening_reference_frame(client: OpenAI, frames: List[str], specification: Dict[str, Any]) -> str:
    opening_hook = specification.get("opening_hook") or "the clip opening"
    scene_sequence = " | ".join(str(x) for x in (specification.get("scene_sequence") or []))
    opening_candidates = frames[: min(4, len(frames))] or frames
    return _select_frame(
        client,
        opening_candidates,
        (
            "Select the single best frame that matches the opening shot of the advertisement clip. "
            "Prioritize the earliest frame that visually corresponds to the stated opening hook, even if it does not show the product as clearly as a later frame. "
            f"Opening hook: {opening_hook}. "
            f"Scene sequence: {scene_sequence}. "
            "Reply with the exact filename only."
        ),
    )


def _select_opening_reference_frame_heuristic(frames: List[str]) -> str:
    if not frames:
        raise ValueError("No frames available for opening-frame selection")
    return frames[0]


def _extract_specification(client: OpenAI, reference_video: str, frames: List[str]) -> Dict[str, Any]:
    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in frames[:8]]
    prompt = """
You are given frames from an 8-second advertisement video.
Extract a compact structured specification that can later be used to regenerate the ad.

Return strict JSON with the following fields:
{
  "brand_or_product": "string",
  "target_audience": "string",
  "theme_message": "string",
  "persuasion_strategy": "string",
  "emotional_tone": "string",
  "visual_concepts": ["string", "..."],
  "camera_style": "string",
  "pacing": "string",
  "scene_summary": "string",
  "scene_sequence": [
    "opening beat",
    "middle beat",
    "closing beat"
  ],
  "opening_hook": "string",
  "closing_payoff": "string",
  "reconstruction_goal": "string",
  "style_keywords": ["string", "..."]
}

Important rules:
- Treat the reference clip as a complete closed unit, not as setup for a longer story.
- Preserve the order of events that actually happen in the clip.
- Do not invent missing scenes outside the clip.
- The scene_sequence should describe the actual event order as briefly as possible.
- opening_hook must describe what the clip starts with.
- closing_payoff must describe what the clip ends with.
Keep the specification concise but specific enough to regenerate a semantically similar ad.
""".strip()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You extract structured ad specifications from short reference videos."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents},
        ],
        max_tokens=700,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_block(raw)
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "usage": _usage_to_plain_dict(getattr(response, "usage", None)),
        "raw_response": raw,
        "parsed_specification": parsed,
        "reference_video": os.path.abspath(reference_video),
        "frame_paths": frames,
    }


def _render_prompt_from_spec(spec: Dict[str, Any]) -> str:
    visual_concepts = ", ".join(spec.get("visual_concepts") or [])
    style_keywords = ", ".join(spec.get("style_keywords") or [])
    scene_sequence = " | ".join(str(x) for x in (spec.get("scene_sequence") or []))
    return (
        f"Reconstruct the advertisement described by the following specification. "
        f"Brand or product: {spec.get('brand_or_product')}. "
        f"Target audience: {spec.get('target_audience')}. "
        f"Theme or message: {spec.get('theme_message')}. "
        f"Persuasion strategy: {spec.get('persuasion_strategy')}. "
        f"Emotional tone: {spec.get('emotional_tone')}. "
        f"Key visual concepts: {visual_concepts}. "
        f"Camera style: {spec.get('camera_style')}. "
        f"Pacing: {spec.get('pacing')}. "
        f"Scene summary: {spec.get('scene_summary')}. "
        f"Scene sequence: {scene_sequence}. "
        f"Opening hook: {spec.get('opening_hook')}. "
        f"Closing payoff: {spec.get('closing_payoff')}. "
        f"Reconstruction goal: {spec.get('reconstruction_goal')}. "
        f"Visual style keywords: {style_keywords}. "
        "Create a short two-scene premium advertisement that preserves the same advertising intent, event order, and visual emphasis, beginning from the original opening rather than a later continuation."
    )


def _build_generation_script(spec: Dict[str, Any], runner_name: str, qwen_model_path: Optional[str], gemini_model_name: str) -> tuple[str, Dict[str, Any]]:
    runner = build_runner(runner_name, qwen_model_path=qwen_model_path, gemini_model_name=gemini_model_name)
    system_prompt = (
        "You are a professional advertising video writer. "
        "Write a concise script for a short reconstructed ad. "
        "Keep it visually concrete, premium, and faithful to the supplied specification."
    )
    user_prompt = f"""
Generate a short two-scene ad script from this specification:
{json.dumps(spec, ensure_ascii=False, indent=2)}

Requirements:
- Keep the script under 120 words.
- Preserve the same advertising intent as the specification.
- Reconstruct the same clip content, not a sequel, continuation, or broader campaign variant.
- Preserve the same beginning-middle-end structure and the same order of events in scene_sequence.
- Treat the provided input image as the opening shot anchor, and begin from that opening moment.
- The opening beat must correspond to: {spec.get("opening_hook")}.
- The ending beat must correspond to: {spec.get("closing_payoff")}.
- Do not add new events that are not implied by the reference clip.
- Mention the product or brand naturally.
- Keep a visually coherent two-scene structure.
- Do not output JSON. Output only the ad script text.
""".strip()
    text = runner.generate(system_prompt, user_prompt, GenerationConfig(max_new_tokens=512, temperature=0.0))
    return text.strip(), runner.get_last_generation_meta()


def _evaluate_spec_alignment(client: OpenAI, output_video: str, specification: Dict[str, Any]) -> Dict[str, Any]:
    frame_dir = os.path.join(os.path.dirname(output_video), "spec_alignment_eval_frames")
    frame_paths = _extract_eval_frames(output_video, frame_dir, fps=1.0, max_frames=8)
    spec_text = json.dumps(specification, ensure_ascii=False, indent=2)
    prompt = f"""
You are evaluating whether a generated advertisement video satisfies an extracted reconstruction specification.

Specification:
{spec_text}

Return strict JSON:
{{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "short string"
}}
""".strip()
    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in frame_paths]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of specification-conditioned advertisement video generation."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents},
        ],
        max_tokens=400,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_block(raw)
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "usage": _usage_to_plain_dict(getattr(response, "usage", None)),
        "frame_dir": frame_dir,
        "frame_paths": frame_paths,
        "raw_response": raw,
        "parsed_response": parsed,
    }


def _evaluate_reference_semantic_fidelity(
    client: OpenAI,
    output_video: str,
    reference_video: str,
    specification: Dict[str, Any],
) -> Dict[str, Any]:
    output_frame_dir = os.path.join(os.path.dirname(output_video), "reference_fidelity_output_frames")
    ref_frame_dir = os.path.join(os.path.dirname(output_video), "reference_fidelity_reference_frames")
    output_frames = _extract_eval_frames(output_video, output_frame_dir, fps=1.0, max_frames=6)
    ref_frames = _extract_eval_frames(reference_video, ref_frame_dir, fps=1.0, max_frames=6)
    spec_text = json.dumps(specification, ensure_ascii=False, indent=2)
    prompt = f"""
You are evaluating whether a reconstructed advertisement video preserves the semantic intent of a reference ad video.

Extracted specification:
{spec_text}

The first group of images comes from the reconstructed video.
The second group comes from the reference video.

Return strict JSON:
{{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "short string"
}}
""".strip()
    output_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in output_frames]
    ref_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in ref_frames]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of advertisement-video reconstruction fidelity."},
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": prompt + "\n\nReconstructed video frames:"}]
                    + output_contents
                    + [{"type": "text", "text": "\nReference video frames:"}]
                    + ref_contents
                ),
            },
        ],
        max_tokens=450,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_block(raw)
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "usage": _usage_to_plain_dict(getattr(response, "usage", None)),
        "output_frame_dir": output_frame_dir,
        "reference_frame_dir": ref_frame_dir,
        "raw_response": raw,
        "parsed_response": parsed,
    }


def _run_single_video(
    *,
    video_path: str,
    workspace: str,
    runner: str,
    qwen_model_path: Optional[str],
    gemini_model_name: str,
    google_api_key: str,
    openai_api_key: str,
    max_new_tokens: int,
    temperature: float,
    pricing_file: Optional[str],
    eval_profile: str,
    xclip_command: Optional[str],
) -> Dict[str, Any]:
    video_id = Path(video_path).stem
    run_dir = os.path.join(os.path.abspath(workspace), "agent_env_runs", "reconstruction_spec_runs", f"{video_id}__{runner}")
    _ensure_dir(run_dir)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    client = OpenAI(api_key=openai_api_key)
    pricing_catalog = _load_pricing(pricing_file) if pricing_file and os.path.exists(pricing_file) else {}

    metadata: Dict[str, Any] = {
        "video_id": video_id,
        "runner": runner,
        "reference_video": os.path.abspath(video_path),
        "paths": {
            "run_dir": run_dir,
            "reference_frames_dir": os.path.join(run_dir, "reference_frames"),
            "specification_path": os.path.join(run_dir, "specification.json"),
            "script_path": os.path.join(run_dir, "reconstruction_script.txt"),
            "veo_output": os.path.join(run_dir, "veo_output.mp4"),
        },
        "runtime_sec": {},
        "api_usage": {},
    }

    reference_frames_dir = metadata["paths"]["reference_frames_dir"]
    reference_frames = _extract_eval_frames(video_path, reference_frames_dir, fps=1.0, max_frames=8)
    stage_start = time.time()
    spec_result = _extract_specification(client, video_path, reference_frames)
    metadata["runtime_sec"]["spec_extraction"] = round(time.time() - stage_start, 3)
    specification = spec_result["parsed_specification"]
    specification_path = metadata["paths"]["specification_path"]
    _save_json(specification_path, specification)
    metadata["specification"] = spec_result
    metadata["api_usage"]["spec_extraction"] = {
        "provider": "openai",
        "model": spec_result.get("model"),
        "usage": spec_result.get("usage") or {},
    }

    joint_frame = _select_joint_reference_frame(client, reference_frames)
    opening_frame = _select_opening_reference_frame(client, reference_frames, specification)
    metadata["selected_assets"] = {
        "joint_reference_frame": joint_frame,
        "opening_reference_frame": opening_frame,
        "veo_anchor_frame": opening_frame,
    }

    stage_start = time.time()
    script_text, script_usage = _build_generation_script(specification, runner, qwen_model_path, gemini_model_name)
    metadata["runtime_sec"]["script_generation"] = round(time.time() - stage_start, 3)
    script_path = metadata["paths"]["script_path"]
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    metadata["generated_script"] = script_text
    metadata["api_usage"]["script_generation"] = script_usage

    veo_output = metadata["paths"]["veo_output"]
    cmd = [
        sys.executable,
        os.path.join(os.path.abspath(workspace), "veo3_pipe.py"),
        "--api-key",
        google_api_key,
        "--script",
        script_path,
        "--image-path",
        opening_frame,
        "--output",
        veo_output,
        "--model",
        "veo-3.1-generate-preview",
        "--aspect-ratio",
        "16:9",
        "--num-videos",
        "1",
        "--person-generation",
        "allow_adult",
        "--poll-interval",
        "10",
        "--poll-timeout",
        "1800",
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_api_key
    env["GOOGLE_API_KEY"] = google_api_key
    code, stdout, stderr, elapsed = _run_cmd(cmd, env=env)
    metadata["runtime_sec"]["veo_stage"] = round(elapsed, 3)
    _save_json(
        os.path.join(run_dir, "runner_subprocess_debug.json"),
        {"command": cmd, "returncode": code, "stdout": stdout, "stderr": stderr},
    )
    if code != 0:
        metadata["veo_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": cmd}
        _save_json(metadata_path, metadata)
        raise RuntimeError(f"Pipeline failed for {video_id}: {stderr or stdout}")
    metadata["veo_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": cmd}
    metadata["api_usage"]["veo_stage"] = {
        "provider": "google",
        "model": "veo-3.1-generate-preview",
        "kind": "video_generation",
        "video_seconds": 8.0,
    }

    fidelity_metrics = compute_reconstruction_metrics(veo_output, video_path, samples=8)
    metadata["reference_fidelity_metrics"] = fidelity_metrics

    final_eval: Dict[str, Any]
    spec_eval: Dict[str, Any]
    recon_eval: Dict[str, Any]

    if eval_profile == "legacy_openai":
        stage_start = time.time()
        final_eval = _evaluate_video_ad(veo_output, openai_api_key=openai_api_key, fps=1.0)
        metadata["runtime_sec"]["final_ad_eval"] = round(time.time() - stage_start, 3)
        metadata["final_ad_eval"] = final_eval
        metadata["api_usage"]["final_ad_eval"] = {
            "provider": "openai",
            "model": final_eval.get("model"),
            "usage": final_eval.get("usage") or {},
        }

        stage_start = time.time()
        spec_eval = _evaluate_spec_alignment(client, veo_output, specification)
        metadata["runtime_sec"]["spec_alignment_eval"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        recon_eval = _evaluate_reference_semantic_fidelity(client, veo_output, video_path, specification)
        metadata["runtime_sec"]["reference_semantic_reconstruction_eval"] = round(time.time() - stage_start, 3)

        metadata["spec_alignment_eval"] = spec_eval
        metadata["reference_semantic_reconstruction_eval"] = recon_eval
        metadata["api_usage"]["spec_alignment_eval"] = {
            "provider": "openai",
            "model": spec_eval.get("model"),
            "usage": spec_eval.get("usage") or {},
        }
        metadata["api_usage"]["reference_semantic_reconstruction_eval"] = {
            "provider": "openai",
            "model": recon_eval.get("model"),
            "usage": recon_eval.get("usage") or {},
        }
    else:
        if not xclip_command:
            raise ValueError("xclip_command is required when eval_profile=skillbench_local")

        stage_start = time.time()
        semantic_eval = evaluate_xclip_alignment(
            output_video=veo_output,
            specification=specification,
            command_template=xclip_command,
        )
        metadata["runtime_sec"]["semantic_alignment_eval"] = round(time.time() - stage_start, 3)
        metadata["semantic_alignment_eval"] = semantic_eval

        stage_start = time.time()
        ocr_eval = evaluate_ocr_text_fidelity(output_video=veo_output, specification=specification)
        metadata["runtime_sec"]["ocr_text_eval"] = round(time.time() - stage_start, 3)
        metadata["ocr_text_eval"] = ocr_eval

        combined = combine_skillbench_scores(
            semantic_score=semantic_eval.get("score"),
            fidelity_score=fidelity_metrics.get("composite_score"),
            text_score=ocr_eval.get("score"),
        )
        metadata["skillbench_eval"] = combined
        final_eval = {"parsed_response": {"score": combined["score"], "pass": combined["pass"]}}
        spec_eval = {"parsed_response": {"score": semantic_eval.get("score"), "pass": semantic_eval.get("pass")}}
        recon_eval = {"parsed_response": {"score": combined["components"]["fidelity_score"], "pass": combined["pass"]}}

    metadata["cost_estimate"] = _estimate_stage_costs(metadata.get("api_usage") or {}, pricing_catalog)
    _save_json(metadata_path, metadata)

    result = {
        "video_id": video_id,
        "reference_video": os.path.abspath(video_path),
        "runner": runner,
        "eval_profile": eval_profile,
        "run_dir": run_dir,
        "specification_path": specification_path,
        "video_output": veo_output,
        "reference_fidelity_composite_score": fidelity_metrics.get("composite_score"),
        "metadata": metadata_path,
        "pipeline_runtime_sec": round(elapsed, 3),
    }
    if eval_profile == "legacy_openai":
        result.update(
            {
                "final_ad_eval_score": final_eval["parsed_response"].get("score"),
                "final_ad_eval_pass": final_eval["parsed_response"].get("pass"),
                "spec_alignment_score": spec_eval["parsed_response"].get("score"),
                "spec_alignment_pass": spec_eval["parsed_response"].get("pass"),
                "reference_semantic_score": recon_eval["parsed_response"].get("score"),
                "reference_semantic_pass": recon_eval["parsed_response"].get("pass"),
            }
        )
    else:
        result.update(
            {
                "semantic_alignment_score": metadata["semantic_alignment_eval"].get("score"),
                "semantic_alignment_pass": metadata["semantic_alignment_eval"].get("pass"),
                "ocr_text_score": metadata["ocr_text_eval"].get("score"),
                "ocr_text_pass": metadata["ocr_text_eval"].get("pass"),
                "skillbench_final_score": metadata["skillbench_eval"].get("score"),
                "skillbench_final_pass": metadata["skillbench_eval"].get("pass"),
            }
        )
    return result


def _run_single_specification(
    *,
    specification_json: str,
    reference_video: str,
    workspace: str,
    runner: str,
    qwen_model_path: Optional[str],
    gemini_model_name: str,
    google_api_key: str,
    openai_api_key: str,
    max_new_tokens: int,
    temperature: float,
    pricing_file: Optional[str],
    eval_profile: str,
    xclip_command: Optional[str],
) -> Dict[str, Any]:
    del max_new_tokens, temperature
    spec_path = os.path.abspath(specification_json)
    with open(spec_path, "r", encoding="utf-8") as f:
        specification = json.load(f)

    video_id = Path(reference_video).stem if reference_video else Path(spec_path).stem
    run_dir = os.path.join(os.path.abspath(workspace), "agent_env_runs", "reconstruction_spec_runs", f"{video_id}__{runner}")
    _ensure_dir(run_dir)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    client = OpenAI(api_key=openai_api_key)
    pricing_catalog = _load_pricing(pricing_file) if pricing_file and os.path.exists(pricing_file) else {}

    metadata: Dict[str, Any] = {
        "video_id": video_id,
        "runner": runner,
        "reference_video": os.path.abspath(reference_video),
        "paths": {
            "run_dir": run_dir,
            "reference_frames_dir": os.path.join(run_dir, "reference_frames"),
            "specification_path": os.path.join(run_dir, "specification.json"),
            "script_path": os.path.join(run_dir, "reconstruction_script.txt"),
            "veo_output": os.path.join(run_dir, "veo_output.mp4"),
        },
        "runtime_sec": {},
        "api_usage": {},
    }

    reference_frames_dir = metadata["paths"]["reference_frames_dir"]
    reference_frames = _extract_eval_frames(reference_video, reference_frames_dir, fps=1.0, max_frames=8)
    specification_path = metadata["paths"]["specification_path"]
    _save_json(specification_path, specification)
    metadata["specification"] = {
        "provider": "provided",
        "model": None,
        "usage": {},
        "raw_response": Path(spec_path).read_text(encoding="utf-8"),
        "parsed_specification": specification,
        "reference_video": os.path.abspath(reference_video),
        "frame_paths": reference_frames,
    }

    opening_frame = _select_opening_reference_frame_heuristic(reference_frames)
    metadata["selected_assets"] = {
        "joint_reference_frame": opening_frame,
        "opening_reference_frame": opening_frame,
        "veo_anchor_frame": opening_frame,
    }

    stage_start = time.time()
    script_text, script_usage = _build_generation_script(specification, runner, qwen_model_path, gemini_model_name)
    metadata["runtime_sec"]["script_generation"] = round(time.time() - stage_start, 3)
    script_path = metadata["paths"]["script_path"]
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    metadata["generated_script"] = script_text
    metadata["api_usage"]["script_generation"] = script_usage

    veo_output = metadata["paths"]["veo_output"]
    cmd = [
        sys.executable,
        os.path.join(os.path.abspath(workspace), "veo3_pipe.py"),
        "--api-key",
        google_api_key,
        "--script",
        script_path,
        "--image-path",
        opening_frame,
        "--output",
        veo_output,
        "--model",
        "veo-3.1-generate-preview",
        "--aspect-ratio",
        "16:9",
        "--num-videos",
        "1",
        "--person-generation",
        "allow_adult",
        "--poll-interval",
        "10",
        "--poll-timeout",
        "1800",
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_api_key
    env["GOOGLE_API_KEY"] = google_api_key
    code, stdout, stderr, elapsed = _run_cmd(cmd, env=env)
    metadata["runtime_sec"]["veo_stage"] = round(elapsed, 3)
    _save_json(
        os.path.join(run_dir, "runner_subprocess_debug.json"),
        {"command": cmd, "returncode": code, "stdout": stdout, "stderr": stderr},
    )
    if code != 0:
        metadata["veo_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": cmd}
        _save_json(metadata_path, metadata)
        raise RuntimeError(f"Pipeline failed for {video_id}: {stderr or stdout}")
    metadata["veo_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": cmd}
    metadata["api_usage"]["veo_stage"] = {
        "provider": "google",
        "model": "veo-3.1-generate-preview",
        "kind": "video_generation",
        "video_seconds": 8.0,
    }

    fidelity_metrics = compute_reconstruction_metrics(veo_output, reference_video, samples=8)
    metadata["reference_fidelity_metrics"] = fidelity_metrics

    if eval_profile == "legacy_openai":
        stage_start = time.time()
        final_eval = _evaluate_video_ad(veo_output, openai_api_key=openai_api_key, fps=1.0)
        metadata["runtime_sec"]["final_ad_eval"] = round(time.time() - stage_start, 3)
        metadata["final_ad_eval"] = final_eval
        metadata["api_usage"]["final_ad_eval"] = {
            "provider": "openai",
            "model": final_eval.get("model"),
            "usage": final_eval.get("usage") or {},
        }

        stage_start = time.time()
        spec_eval = _evaluate_spec_alignment(client, veo_output, specification)
        metadata["runtime_sec"]["spec_alignment_eval"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        recon_eval = _evaluate_reference_semantic_fidelity(client, veo_output, reference_video, specification)
        metadata["runtime_sec"]["reference_semantic_reconstruction_eval"] = round(time.time() - stage_start, 3)

        metadata["spec_alignment_eval"] = spec_eval
        metadata["reference_semantic_reconstruction_eval"] = recon_eval
        metadata["api_usage"]["spec_alignment_eval"] = {
            "provider": "openai",
            "model": spec_eval.get("model"),
            "usage": spec_eval.get("usage") or {},
        }
        metadata["api_usage"]["reference_semantic_reconstruction_eval"] = {
            "provider": "openai",
            "model": recon_eval.get("model"),
            "usage": recon_eval.get("usage") or {},
        }
    else:
        if not xclip_command:
            raise ValueError("xclip_command is required when eval_profile=skillbench_local")

        stage_start = time.time()
        semantic_eval = evaluate_xclip_alignment(
            output_video=veo_output,
            specification=specification,
            command_template=xclip_command,
        )
        metadata["runtime_sec"]["semantic_alignment_eval"] = round(time.time() - stage_start, 3)
        metadata["semantic_alignment_eval"] = semantic_eval

        stage_start = time.time()
        ocr_eval = evaluate_ocr_text_fidelity(output_video=veo_output, specification=specification)
        metadata["runtime_sec"]["ocr_text_eval"] = round(time.time() - stage_start, 3)
        metadata["ocr_text_eval"] = ocr_eval

        combined = combine_skillbench_scores(
            semantic_score=semantic_eval.get("score"),
            fidelity_score=fidelity_metrics.get("composite_score"),
            text_score=ocr_eval.get("score"),
        )
        metadata["skillbench_eval"] = combined
        final_eval = {"parsed_response": {"score": combined["score"], "pass": combined["pass"]}}
        spec_eval = {"parsed_response": {"score": semantic_eval.get("score"), "pass": semantic_eval.get("pass")}}
        recon_eval = {"parsed_response": {"score": combined["components"]["fidelity_score"], "pass": combined["pass"]}}

    metadata["cost_estimate"] = _estimate_stage_costs(metadata.get("api_usage") or {}, pricing_catalog)
    _save_json(metadata_path, metadata)

    result = {
        "video_id": video_id,
        "reference_video": os.path.abspath(reference_video),
        "runner": runner,
        "eval_profile": eval_profile,
        "run_dir": run_dir,
        "specification_path": specification_path,
        "video_output": veo_output,
        "reference_fidelity_composite_score": fidelity_metrics.get("composite_score"),
        "metadata": metadata_path,
        "pipeline_runtime_sec": round(elapsed, 3),
    }
    if eval_profile == "legacy_openai":
        result.update(
            {
                "final_ad_eval_score": final_eval["parsed_response"].get("score"),
                "final_ad_eval_pass": final_eval["parsed_response"].get("pass"),
                "spec_alignment_score": spec_eval["parsed_response"].get("score"),
                "spec_alignment_pass": spec_eval["parsed_response"].get("pass"),
                "reference_semantic_score": recon_eval["parsed_response"].get("score"),
                "reference_semantic_pass": recon_eval["parsed_response"].get("pass"),
            }
        )
    else:
        result.update(
            {
                "semantic_alignment_score": metadata["semantic_alignment_eval"].get("score"),
                "semantic_alignment_pass": metadata["semantic_alignment_eval"].get("pass"),
                "ocr_text_score": metadata["ocr_text_eval"].get("score"),
                "ocr_text_pass": metadata["ocr_text_eval"].get("pass"),
                "skillbench_final_score": metadata["skillbench_eval"].get("score"),
                "skillbench_final_pass": metadata["skillbench_eval"].get("pass"),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professor-style reconstruction: video -> specification -> video.")
    parser.add_argument("--input-dir", default="recon_videos", help="Directory of short reference ad videos")
    parser.add_argument("--specification-json", default=None, help="Optional direct-spec mode input JSON")
    parser.add_argument("--reference-video", default=None, help="Reference video used for anchor selection and fidelity evaluation in direct-spec mode")
    parser.add_argument("--runner", required=True, choices=["qwen", "gemini"])
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--gemini-model-name", default="gemini-2.5-flash")
    parser.add_argument("--google-api-key", required=True)
    parser.add_argument("--openai-api-key", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--eval-profile", choices=["legacy_openai", "skillbench_local"], default="legacy_openai")
    parser.add_argument(
        "--xclip-command",
        default=None,
        help="Command template for local X-CLIP evaluation. Available placeholders: {video}, {text_file}, {output_json}",
    )
    parser.add_argument("--pricing-file", default=os.path.join(os.path.dirname(__file__), "benchmark_pricing.json"))
    parser.add_argument("--output", default="agent_env_runs/benchmarks/reconstruction_spec_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if args.specification_json:
        if not args.reference_video:
            raise SystemExit("--reference-video is required when using --specification-json")
        try:
            results.append(
                _run_single_specification(
                    specification_json=args.specification_json,
                    reference_video=os.path.abspath(args.reference_video),
                    workspace=args.workspace,
                    runner=args.runner,
                    qwen_model_path=args.qwen_model_path,
                    gemini_model_name=args.gemini_model_name,
                    google_api_key=args.google_api_key,
                    openai_api_key=args.openai_api_key,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    pricing_file=args.pricing_file,
                    eval_profile=args.eval_profile,
                    xclip_command=args.xclip_command,
                )
            )
        except Exception as exc:
            failures.append({"specification_json": os.path.abspath(args.specification_json), "error": str(exc)})
        input_dir = os.path.abspath(args.specification_json)
        num_videos = 1
    else:
        input_dir = os.path.abspath(args.input_dir)
        videos = sorted(str(p) for p in Path(input_dir).glob("*.mp4"))
        if args.max_videos is not None:
            videos = videos[: args.max_videos]
        if not videos:
            raise SystemExit(f"No .mp4 files found in {input_dir}")
        for video_path in videos:
            try:
                results.append(
                    _run_single_video(
                        video_path=video_path,
                        workspace=args.workspace,
                        runner=args.runner,
                        qwen_model_path=args.qwen_model_path,
                        gemini_model_name=args.gemini_model_name,
                        google_api_key=args.google_api_key,
                        openai_api_key=args.openai_api_key,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        pricing_file=args.pricing_file,
                        eval_profile=args.eval_profile,
                        xclip_command=args.xclip_command,
                    )
                )
            except Exception as exc:
                failures.append({"video": video_path, "error": str(exc)})
        num_videos = len(videos)

    payload = {
        "runner": args.runner,
        "input_dir": input_dir,
        "num_videos": num_videos,
        "num_success": len(results),
        "num_failures": len(failures),
        "results": results,
        "failures": failures,
    }
    abs_output = os.path.abspath(args.output)
    _ensure_dir(os.path.dirname(abs_output) or ".")
    with open(abs_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": abs_output, "num_success": len(results), "num_failures": len(failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
