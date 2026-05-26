from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from openai import OpenAI

from .model_runners import GenerationConfig, RunnerError, build_runner
from .model_task_benchmark import _extract_json_block
from .model_task_examples import TASK1_SYSTEM_PROMPT

import demo_pipe1 as demo_pipe1_module
from demo_pipe1 import (
    describe_image,
    encode_image_base64,
    extract_frames,
    load_transcript,
    pick_frames_progressive,
    summarize_long_video,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_text(path: str, content: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> Tuple[int, str, str, float]:
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    elapsed = time.time() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def _load_pricing(pricing_file: str | None) -> Dict[str, Any]:
    if not pricing_file:
        return {}
    with open(pricing_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _usage_to_plain_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    payload: Dict[str, Any] = {}
    for key in dir(usage):
        if key.startswith("_"):
            continue
        value = getattr(usage, key, None)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
    return payload


class _OpenAIChatRecorder:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._original = client.chat.completions.create
        self.calls: List[Dict[str, Any]] = []

    def _wrapped(self, *args: Any, **kwargs: Any) -> Any:
        response = self._original(*args, **kwargs)
        model = kwargs.get("model") or getattr(response, "model", None)
        usage = _usage_to_plain_dict(getattr(response, "usage", None))
        self.calls.append(
            {
                "provider": "openai",
                "model": model,
                "usage": usage,
            }
        )
        return response

    def summary(self) -> Dict[str, Any]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        per_model: Dict[str, Dict[str, Any]] = {}
        for call in self.calls:
            usage = call.get("usage") or {}
            model = str(call.get("model") or "unknown")
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or (prompt + completion))
            prompt_tokens += prompt
            completion_tokens += completion
            total_tokens += total
            bucket = per_model.setdefault(
                model,
                {"request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            bucket["request_count"] += 1
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["total_tokens"] += total
        return {
            "request_count": len(self.calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "per_model": per_model,
            "calls": self.calls,
        }


@contextmanager
def _record_demo_pipe1_openai_usage() -> Iterator[_OpenAIChatRecorder]:
    recorder = _OpenAIChatRecorder(demo_pipe1_module.client)
    demo_pipe1_module.client.chat.completions.create = recorder._wrapped  # type: ignore[assignment]
    try:
        yield recorder
    finally:
        demo_pipe1_module.client.chat.completions.create = recorder._original  # type: ignore[assignment]


def _estimate_cost_from_usage(model: str, usage: Dict[str, Any], pricing: Dict[str, Any]) -> float | None:
    if not pricing:
        return None
    model_cfg = pricing.get(model)
    if not model_cfg:
        return None
    if model_cfg.get("pricing_mode") != "per_1m_tokens":
        return None
    input_rate = _safe_float(model_cfg.get("input_usd_per_1m"))
    output_rate = _safe_float(model_cfg.get("output_usd_per_1m"))
    if input_rate is None or output_rate is None:
        return None
    prompt_tokens = _safe_float(usage.get("prompt_tokens") or usage.get("prompt_token_count"))
    completion_tokens = _safe_float(usage.get("completion_tokens") or usage.get("candidates_token_count"))
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_cost = (prompt_tokens or 0.0) / 1_000_000.0 * input_rate
    output_cost = (completion_tokens or 0.0) / 1_000_000.0 * output_rate
    return round(prompt_cost + output_cost, 6)


def _estimate_fixed_cost(model: str, pricing: Dict[str, Any], *, image_count: int = 0, seconds: float = 0.0) -> float | None:
    if not pricing:
        return None
    model_cfg = pricing.get(model)
    if not model_cfg:
        return None
    mode = model_cfg.get("pricing_mode")
    if mode == "per_image":
        unit = _safe_float(model_cfg.get("output_usd_per_image"))
        if unit is None:
            return None
        return round(unit * image_count, 6)
    if mode == "per_second":
        unit = _safe_float(model_cfg.get("output_usd_per_second"))
        if unit is None:
            return None
        return round(unit * seconds, 6)
    return None


def _estimate_stage_costs(stage_usage: Dict[str, Any], pricing_catalog: Dict[str, Any]) -> Dict[str, Any]:
    stage_costs: Dict[str, Any] = {}
    total = 0.0
    any_cost = False

    openai_pricing = pricing_catalog.get("openai", {})
    google_pricing = pricing_catalog.get("google", {})

    for stage_name, usage in stage_usage.items():
        stage_cost: float | None = None
        provider = usage.get("provider")
        model = usage.get("model")
        if provider == "openai":
            stage_cost = _estimate_cost_from_usage(str(model), usage.get("usage") or {}, openai_pricing)
        elif provider == "google":
            stage_cost = _estimate_cost_from_usage(str(model), usage.get("usage") or {}, google_pricing)
            if stage_cost is None and usage.get("kind") == "image_generation":
                stage_cost = _estimate_fixed_cost(str(model), google_pricing, image_count=int(usage.get("image_count") or 0))
            if stage_cost is None and usage.get("kind") == "video_generation":
                stage_cost = _estimate_fixed_cost(str(model), google_pricing, seconds=float(usage.get("video_seconds") or 0.0))
        stage_costs[stage_name] = stage_cost
        if stage_cost is not None:
            total += stage_cost
            any_cost = True
    stage_costs["total_estimated_cost_usd"] = round(total, 6) if any_cost else None
    return stage_costs


def _probe_video_duration_sec(video_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        return round(float(proc.stdout.strip()), 3)
    except ValueError:
        return None


def _build_task1_planner_prompt(
    *,
    object_desc: str,
    character_desc: str,
    long_video_summary: str,
    target_audience: str,
    product_name: str,
    source_mode: str,
    external_task_prompt: str | None = None,
) -> str:
    prompt_suffix = (
        "Fit the ad into the supplied story world and preserve scene continuity."
        if source_mode == "long_video"
        else "Build the ad directly from the supplied character and product visuals without relying on an external story clip."
    )
    task_prompt = (
        external_task_prompt.strip()
        if external_task_prompt and external_task_prompt.strip()
        else (
            f"Create a short two-scene premium ad for {product_name}. "
            "Scene 1 should be a hero product shot. "
            f"Scene 2 should transition into a polished payoff shot. {prompt_suffix}"
        )
    )
    return json.dumps(
        {
            "task_type": "task1_open_generation_plan",
            "title": "End-to-end ad video planning with downstream image fusion and VEO generation",
            "prompt": task_prompt,
            "inputs": {
                "object_description": object_desc,
                "character_description": character_desc,
                "long_video_summary": long_video_summary,
                "source_mode": source_mode,
                "target_audience": target_audience,
                "num_scenes": 2,
                "target_edit_properties": {
                    "needs_smooth_transition": True,
                    "camera_motion": "slow_zoom_in",
                    "visual_style": "premium_ad"
                },
            },
            "notes": (
                "Return a plan that can be turned into a two-scene advertising script and an opening-shot staging relation. "
                "The result will later be executed by image fusion and a VEO video generator."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _render_script_from_plan(plan: Dict[str, Any]) -> str:
    scenes = plan.get("scenes") or []
    summary = str(plan.get("summary") or "A short premium ad")
    transition_strategy = str(plan.get("transition_strategy") or "smooth continuity")
    lines = [f"Ad concept: {summary}"]
    for idx, scene in enumerate(scenes[:2], start=1):
        goal = str(scene.get("goal") or "")
        focus = str(scene.get("visual_focus") or "")
        motion = str(scene.get("camera_motion") or "")
        style = ", ".join(str(item) for item in (scene.get("style_keywords") or []))
        lines.append(
            f"Scene {idx}: {goal} Focus on {focus}. Camera motion: {motion}. Visual style: {style}."
        )
    lines.append(f"Transition: {transition_strategy}.")
    return "\n".join(lines)


def _render_relation_from_plan(plan: Dict[str, Any]) -> str:
    scenes = plan.get("scenes") or []
    first_scene = scenes[0] if scenes else {}
    goal = str(first_scene.get("goal") or "").lower()
    if "wear" in goal:
        relation = "The character in the first image wears the object in the second image in the opening hero shot."
    elif "hold" in goal or "present" in goal:
        relation = "The character in the first image holds the object in the second image and presents it clearly to the camera."
    else:
        relation = "The character in the first image presents the object in the second image prominently in the opening ad shot."
    return relation


def _salvage_task1_plan(raw_text: str) -> Dict[str, Any] | None:
    summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', raw_text, re.DOTALL)
    if not summary_match:
        return None

    scenes: List[Dict[str, Any]] = []
    scene_pattern = re.compile(
        r'\{\s*"scene_id"\s*:\s*"(?P<scene_id>[^"]+)"\s*,\s*"goal"\s*:\s*"(?P<goal>[^"]*)"\s*,\s*"visual_focus"\s*:\s*"(?P<visual_focus>[^"]*)"\s*,\s*"camera_motion"\s*:\s*"(?P<camera_motion>[^"]*)"\s*,\s*"style_keywords"\s*:\s*\[(?P<style_keywords>[^\]]*)\]',
        re.DOTALL,
    )
    for match in scene_pattern.finditer(raw_text):
        raw_keywords = match.group("style_keywords")
        style_keywords = re.findall(r'"([^"]+)"', raw_keywords)
        scenes.append(
            {
                "scene_id": match.group("scene_id"),
                "goal": match.group("goal"),
                "visual_focus": match.group("visual_focus"),
                "camera_motion": match.group("camera_motion"),
                "style_keywords": style_keywords,
            }
        )
    if not scenes:
        return None

    transition_match = re.search(r'"transition_strategy"\s*:\s*"([^"]*)"', raw_text, re.DOTALL)
    transition_strategy = transition_match.group(1) if transition_match else "smooth premium transition"
    return {
        "summary": summary_match.group(1),
        "scenes": scenes[:2],
        "planned_actions": ["plan_scenes", "gen_scene_image", "gen_scene_video", "judge_scene", "judge_final", "finish"],
        "transition_strategy": transition_strategy,
        "verification_plan": ["judge_scene", "judge_final"],
        "_salvaged": True,
    }


def _evaluate_video_ad(output_video: str, openai_api_key: str, fps: float = 1.0) -> Dict[str, Any]:
    client = OpenAI(api_key=openai_api_key)
    frames_dir = os.path.join(os.path.dirname(output_video), "eval_frames")
    extract_frames(output_video, frames_dir, fps=fps)
    frame_paths = sorted(str(p) for p in Path(frames_dir).glob("*.jpg"))

    eval_prompt = """
Please review these frames from a generated advertisement video.
Evaluate whether:
1. the product is visible and visually emphasized,
2. the character-product composition looks natural,
3. the frames feel like a coherent premium ad video.
Return strict JSON:
{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "short string"
}
""".strip()

    image_contents = [{"type": "image_url", "image_url": {"url": encode_image_base64(img)}} for img in frame_paths[:8]]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of generated advertising videos."},
            {"role": "user", "content": [{"type": "text", "text": eval_prompt}] + image_contents},
        ],
        max_tokens=400,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = _extract_json_block(raw)
    except Exception as exc:
        raise ValueError(f"Final ad evaluator did not return valid JSON. Raw response: {raw[:1200]}") from exc
    return {
        "raw_response": raw,
        "parsed_response": parsed,
        "usage": _usage_to_plain_dict(getattr(response, "usage", None)),
        "provider": "openai",
        "model": "gpt-4o",
        "frame_dir": frames_dir,
        "frame_paths": frame_paths,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end planner + Nano + VEO pipeline with runtime/eval logging.")
    ap.add_argument("--runner", required=True, choices=["qwen", "gemini"], help="Planner backend")
    ap.add_argument("--qwen-model-path", default=None, help="Required for runner=qwen")
    ap.add_argument("--gemini-model-name", default="gemini-2.5-flash", help="Gemini planner model name")
    ap.add_argument("--workspace", default=".", help="Workspace root")
    ap.add_argument("--long-video-path", default=None, help="Optional long video input for stage 1")
    ap.add_argument("--character-image", default=None, help="Optional direct character image input for stage 1")
    ap.add_argument("--object-image", required=True, help="Product/object image")
    ap.add_argument("--transcript-file", default=None, help="Transcript file for long video")
    ap.add_argument("--google-api-key", required=True, help="Google API key for Nano/VEO")
    ap.add_argument("--openai-api-key", default=None, help="OpenAI API key for stage 1 + eval; defaults to env OPENAI_API_KEY")
    ap.add_argument("--target-audience", default="Young premium consumers who respond to polished, elegant ads.")
    ap.add_argument("--product-name", default="the product")
    ap.add_argument("--task-prompt-override-file", default=None, help="Optional file containing a task-specific planner prompt override.")
    ap.add_argument("--nano-model", default="gemini-2.5-flash-image", help="Nano image-generation model")
    ap.add_argument("--veo-model", default="veo-3.1-generate-preview", help="VEO model name")
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--pricing-file",
        default=os.path.join(os.path.dirname(__file__), "benchmark_pricing.json"),
        help="Optional JSON file with token/image/video pricing rules for cost estimation.",
    )
    ap.add_argument("--run-dir", default=None, help="Optional explicit output run dir")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.long_video_path and not args.character_image:
        raise SystemExit("Provide either --long-video-path or --character-image.")
    if args.long_video_path and not args.transcript_file:
        raise SystemExit("--transcript-file is required when --long-video-path is provided.")
    workspace = os.path.abspath(args.workspace)
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("Missing OpenAI API key. Pass --openai-api-key or set OPENAI_API_KEY.")

    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.join(workspace, "agent_env_runs", "e2e", f"run_{run_id}")
    stage1_output_dir = os.path.join(run_dir, "frames_s1")
    stage1_export_dir = os.path.join(run_dir, "export_for_nano")
    nano_output_dir = os.path.join(run_dir, "fused_outputs")
    nano_export_selected = os.path.join(run_dir, "export_review", "edited_image.jpg")
    veo_output = os.path.join(run_dir, "veo_output.mp4")
    metadata_path = os.path.join(run_dir, "run_metadata.json")

    _ensure_dir(run_dir)
    pricing_catalog = _load_pricing(args.pricing_file) if args.pricing_file and os.path.exists(args.pricing_file) else {}

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "runner": args.runner,
        "qwen_model_path": args.qwen_model_path,
        "gemini_model_name": args.gemini_model_name,
        "nano_model": args.nano_model,
        "veo_model": args.veo_model,
        "pricing_file": args.pricing_file if pricing_catalog else None,
        "paths": {
            "run_dir": run_dir,
            "stage1_output_dir": stage1_output_dir,
            "stage1_export_dir": stage1_export_dir,
            "nano_output_dir": nano_output_dir,
            "nano_export_selected": nano_export_selected,
            "veo_output": veo_output,
        },
        "runtime_sec": {},
        "api_usage": {},
    }

    total_start = time.time()

    # Stage 1
    stage_start = time.time()
    source_mode = "long_video" if args.long_video_path else "character_image"
    long_frames_dir = os.path.join(stage1_output_dir, "long_video")
    with _record_demo_pipe1_openai_usage() as stage1_recorder:
        if args.long_video_path:
            extract_frames(args.long_video_path, long_frames_dir, fps=args.fps)
            long_frames = sorted(str(p) for p in Path(long_frames_dir).glob("*.jpg"))
            base_prompt_single = (
                "From these frames, please select EXACTLY ONE filename (e.g., frame_0003.jpg) "
                "that shows the main character with a face or portrait. "
                "Reply ONLY with the filename, and nothing else."
            )
            candidate_pool = long_frames[:20] if len(long_frames) > 20 else long_frames
            picked_frames = pick_frames_progressive(
                topk=max(1, int(args.topk)),
                images=candidate_pool,
                base_prompt_single=base_prompt_single,
                max_attempts=9,
            )
            if not picked_frames:
                raise RuntimeError("No character frames selected.")
            transcript = load_transcript(args.transcript_file)
            summary_prompt = "Please summarize the main storyline, scene, characters, emotions, and environment from these video keyframes and transcript."
            long_video_summary = summarize_long_video(long_frames, transcript, summary_prompt)
        else:
            picked_frames = [os.path.abspath(args.character_image)]
            long_frames = []
            long_video_summary = (
                "No long-form source video is provided. Build a standalone premium ad directly from the supplied character image and product image."
            )
        object_desc = describe_image(args.object_image, "Please describe the key product or object in this image.")
        character_desc = describe_image(picked_frames[0], "Please describe the main character in this image.")
    stage1_usage_summary = stage1_recorder.summary()
    metadata["runtime_sec"]["stage1_preprocess"] = round(time.time() - stage_start, 3)
    metadata["stage1"] = {
        "source_mode": source_mode,
        "picked_frames": picked_frames,
        "object_desc": object_desc,
        "character_desc": character_desc,
        "long_video_summary": long_video_summary,
        "long_video_path": os.path.abspath(args.long_video_path) if args.long_video_path else None,
        "character_image": os.path.abspath(args.character_image) if args.character_image else None,
        "transcript_file": os.path.abspath(args.transcript_file) if args.transcript_file else None,
    }
    metadata["api_usage"]["stage1_preprocess"] = {
        "provider": "openai",
        "model": "multiple",
        "usage": stage1_usage_summary,
    }

    # Planner
    stage_start = time.time()
    planner_prompt = _build_task1_planner_prompt(
        object_desc=object_desc,
        character_desc=character_desc,
        long_video_summary=long_video_summary,
        target_audience=args.target_audience,
        product_name=args.product_name,
        source_mode=source_mode,
        external_task_prompt=(
            Path(args.task_prompt_override_file).read_text(encoding="utf-8")
            if args.task_prompt_override_file
            else None
        ),
    )
    runner = build_runner(
        args.runner,
        qwen_model_path=args.qwen_model_path,
        gemini_model_name=args.gemini_model_name,
    )
    raw_plan = runner.generate(
        TASK1_SYSTEM_PROMPT,
        planner_prompt,
        GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature),
    )
    planner_usage_meta = runner.get_last_generation_meta()
    try:
        parsed_plan = _extract_json_block(raw_plan)
    except Exception as exc:
        salvaged_plan = _salvage_task1_plan(raw_plan)
        if salvaged_plan is None:
            metadata["runtime_sec"]["planner"] = round(time.time() - stage_start, 3)
            metadata["planner"] = {
                "raw_response": raw_plan,
                "parsed_plan": None,
                "parse_error": str(exc),
            }
            metadata["api_usage"]["planner"] = planner_usage_meta
            _save_json(metadata_path, metadata)
            raise SystemExit(f"planner parse failed. See metadata: {metadata_path}")
        parsed_plan = salvaged_plan
    rendered_script = _render_script_from_plan(parsed_plan)
    rendered_relation = _render_relation_from_plan(parsed_plan)
    _ensure_dir(stage1_export_dir)
    _save_text(os.path.join(stage1_export_dir, "prompt.txt"), rendered_script)
    _save_text(os.path.join(stage1_export_dir, "character_object_relation.txt"), rendered_relation)
    # export images expected downstream
    from shutil import copyfile
    copyfile(picked_frames[0], os.path.join(stage1_export_dir, "img1.jpg"))
    copyfile(args.object_image, os.path.join(stage1_export_dir, "img2.jpg"))
    metadata["runtime_sec"]["planner"] = round(time.time() - stage_start, 3)
    metadata["planner"] = {
        "raw_response": raw_plan,
        "parsed_plan": parsed_plan,
        "parse_salvaged": bool(parsed_plan.get("_salvaged")),
        "rendered_script": rendered_script,
        "rendered_relation": rendered_relation,
    }
    metadata["api_usage"]["planner"] = planner_usage_meta

    # Stage 2 nano
    stage_start = time.time()
    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = args.google_api_key
    env["OPENAI_API_KEY"] = openai_api_key
    nano_cmd = [
        sys.executable,
        os.path.join(workspace, "nano_pipe1.py"),
        "--img1", os.path.join(stage1_export_dir, "img1.jpg"),
        "--img2", os.path.join(stage1_export_dir, "img2.jpg"),
        "--prompt-file", os.path.join(stage1_export_dir, "character_object_relation.txt"),
        "--output-dir", nano_output_dir,
        "--output", nano_export_selected,
        "--model", args.nano_model,
        "--max-attempts", "3",
    ]
    code, stdout, stderr, elapsed = _run_cmd(nano_cmd, env=env)
    metadata["runtime_sec"]["nano_stage"] = round(elapsed, 3)
    metadata["nano_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": nano_cmd}
    metadata["api_usage"]["nano_stage"] = {
        "provider": "google",
        "model": args.nano_model,
        "kind": "image_generation",
        "image_count": 1,
    }
    if code != 0:
        _save_json(metadata_path, metadata)
        raise SystemExit(f"nano stage failed. See metadata: {metadata_path}")

    # Stage 3 veo
    stage_start = time.time()
    veo_cmd = [
        sys.executable,
        os.path.join(workspace, "veo3_pipe.py"),
        "--api-key", args.google_api_key,
        "--script", os.path.join(stage1_export_dir, "prompt.txt"),
        "--image-path", nano_export_selected,
        "--output", veo_output,
        "--model", args.veo_model,
        "--aspect-ratio", "16:9",
        "--num-videos", "1",
        "--person-generation", "allow_adult",
        "--poll-interval", "10",
        "--poll-timeout", "1800",
    ]
    code, stdout, stderr, elapsed = _run_cmd(veo_cmd, env=env)
    metadata["runtime_sec"]["veo_stage"] = round(elapsed, 3)
    metadata["veo_stage"] = {"returncode": code, "stdout": stdout, "stderr": stderr, "command": veo_cmd}
    video_duration_sec = _probe_video_duration_sec(veo_output) if code == 0 else None
    metadata["api_usage"]["veo_stage"] = {
        "provider": "google",
        "model": args.veo_model,
        "kind": "video_generation",
        "video_seconds": video_duration_sec,
    }
    if code != 0:
        _save_json(metadata_path, metadata)
        raise SystemExit(f"veo stage failed. See metadata: {metadata_path}")

    # Eval
    stage_start = time.time()
    eval_result = _evaluate_video_ad(veo_output, openai_api_key=openai_api_key, fps=1.0)
    metadata["runtime_sec"]["eval_stage"] = round(time.time() - stage_start, 3)
    metadata["eval_stage"] = eval_result
    metadata["api_usage"]["eval_stage"] = {
        "provider": "openai",
        "model": eval_result.get("model"),
        "usage": eval_result.get("usage") or {},
    }

    metadata["runtime_sec"]["total"] = round(time.time() - total_start, 3)
    metadata["cost_estimate"] = _estimate_stage_costs(metadata["api_usage"], pricing_catalog)
    _save_json(metadata_path, metadata)

    print(
        json.dumps(
            {
                "run_dir": run_dir,
                "video_output": veo_output,
                "eval_score": eval_result["parsed_response"].get("score"),
                "eval_pass": eval_result["parsed_response"].get("pass"),
                "metadata": metadata_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RunnerError as exc:
        raise SystemExit(f"Runner error: {exc}") from exc
