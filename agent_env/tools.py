from __future__ import annotations

import json
import os
import random
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ad_ranker_bridge import AdRankerJudgeBridge
from .types import ToolResult


@dataclass
class ToolAdapterConfig:
    workspace: str
    artifacts_dir: str = "agent_env_runs"
    ad_ranker_dir: str = "/Users/sxh/Desktop/video-studio/ad_ranker"
    baseline_video_path: Optional[str] = None
    default_customer_profile: str = (
        "A practical online shopper who values clear product presentation, trustworthiness, "
        "and ads that quickly explain why the product is useful."
    )
    enable_ad_ranker_judge: bool = True
    pairwise_strategy: str = "reference"
    pairwise_budget: int = 1


class BaseToolAdapter(ABC):
    def __init__(self, config: ToolAdapterConfig):
        self.config = config
        self.workspace = os.path.abspath(config.workspace)
        self.artifacts_dir = os.path.join(self.workspace, config.artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.ad_ranker_bridge = AdRankerJudgeBridge(
            ad_ranker_dir=config.ad_ranker_dir,
            default_customer_profile=config.default_customer_profile,
            enabled=config.enable_ad_ranker_judge,
        )

    @abstractmethod
    def plan_scenes(self, prompt: str, num_scenes: int) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def generate_scene_image(self, scene_id: str, scene_desc: str, prompt: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def generate_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def zoom_in_clip(self, scene_id: str, video_path: str, zoom_factor: float = 1.15) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def zoom_out_clip(self, scene_id: str, video_path: str, zoom_factor: float = 0.85) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def cut_for_transition(
        self,
        scene_id: str,
        video_path: str,
        cut_preference: Optional[str] = None,
    ) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def cut_for_transition_pair(
        self,
        from_scene_id: str,
        from_video_path: str,
        to_scene_id: str,
        to_video_path: str,
        cut_preference: Optional[str] = None,
    ) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def transition_edit(self, from_scene_id: str, to_scene_id: str, style: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def trim_clip(self, scene_id: str, video_path: str, trim_start_sec: float, trim_end_sec: float) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def stitch_preview(self, scene_videos: List[str], transition_hints: Optional[List[str]] = None) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def replace_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def judge_scene(self, scene_id: str, scene_desc: str, image_path: Optional[str], video_path: Optional[str]) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def judge_final(
        self,
        prompt: str,
        scene_videos: List[str],
        judge_context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        raise NotImplementedError

    def _attach_ad_ranker_metrics(
        self,
        result: ToolResult,
        prompt: str,
        scene_videos: List[str],
        judge_context: Optional[Dict[str, Any]],
    ) -> None:
        context = judge_context or {}
        reference_video = context.get("reference_video") or self.config.baseline_video_path
        customer_profile = context.get("customer_profile") or self.config.default_customer_profile
        candidate_video = context.get("candidate_video") or (scene_videos[-1] if scene_videos else None)
        judge_mode = context.get("judge_mode", "persona_pairwise")
        if judge_mode == "none" or not candidate_video:
            return

        ad_ranker_result = self.ad_ranker_bridge.evaluate(
            prompt=prompt,
            candidate_video=candidate_video,
            reference_video=reference_video,
            customer_profile=customer_profile,
            pairwise_strategy=context.get("pairwise_strategy", self.config.pairwise_strategy),
            pairwise_budget=int(context.get("pairwise_budget", self.config.pairwise_budget)),
        )
        if not ad_ranker_result.get("available"):
            result.meta["ad_ranker"] = ad_ranker_result
            return

        persona_score = float(ad_ranker_result["persona_preference_score"])
        base_final = float(result.metrics.get("final_score", 0.0))
        result.metrics["persona_preference_score"] = round(persona_score, 3)
        result.metrics["final_score"] = round(0.7 * base_final + 0.3 * persona_score, 3)
        result.meta["ad_ranker"] = ad_ranker_result


def _run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _run_subprocess_bytes(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _has_ffmpeg_encoder(encoder_name: str) -> bool:
    proc = _run_subprocess(["ffmpeg", "-hide_banner", "-encoders"])
    if proc.returncode != 0:
        return False
    return encoder_name in proc.stdout


def _preferred_video_encoder() -> str:
    for encoder_name in ("libx264", "libopenh264", "mpeg4"):
        if _has_ffmpeg_encoder(encoder_name):
            return encoder_name
    return "mpeg4"


def _aac_supported() -> bool:
    proc = _run_subprocess(["ffmpeg", "-hide_banner", "-encoders"])
    if proc.returncode != 0:
        return False
    return " aac " in f" {proc.stdout} " or "\naac " in proc.stdout


def _probe_video_dimensions(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        video_path,
    ]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    width_str, height_str = proc.stdout.strip().split("x", 1)
    return int(width_str), int(height_str)


def _probe_video_duration(video_path: str) -> float:
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
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe duration failed")
    return float(proc.stdout.strip())


def _make_mock_video(output_path: str, label: str, duration_sec: int = 3) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_encoder = _preferred_video_encoder()
    half_duration = max(1.0, duration_sec / 2.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=25:duration={half_duration}",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=size=1280x720:rate=25:duration={half_duration}",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        video_encoder,
    ]
    if _aac_supported():
        cmd += ["-c:a", "aac"]
    cmd += [
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={label}",
        output_path,
    ]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg mock video generation failed")


def _build_ad_scene_descriptions(prompt: str, num_scenes: int) -> List[Dict[str, str]]:
    n = max(1, min(8, int(num_scenes)))
    prompt_short = prompt.strip()
    if n == 1:
        return [
            {
                "scene_id": "scene_001",
                "description": (
                    "Opening hero product shot for a premium ad video. "
                    f"Focus on the product as the visual centerpiece with polished lighting, clean composition, "
                    f"subtle camera movement, and brand-forward presentation. Creative brief: {prompt_short[:140]}"
                ),
            }
        ]

    scenes: List[Dict[str, str]] = []
    scene_templates = [
        (
            "Scene 1: Product hero opening shot. Introduce the product with premium cinematic lighting, "
            "tight framing, strong product visibility, elegant motion, and a clean luxury-commercial feel. "
            f"Creative brief: {prompt_short[:140]}"
        ),
        (
            "Scene 2: Lifestyle or payoff shot. Show the product in use or in a polished brand context, "
            "preserve visual continuity, and end with a persuasive ad-style finish that highlights the key value proposition. "
            f"Creative brief: {prompt_short[:140]}"
        ),
        (
            "Scene 3: Benefit-driven follow-up shot. Emphasize texture, usability, and premium detail while keeping the pacing suitable for a short-form advertisement. "
            f"Creative brief: {prompt_short[:140]}"
        ),
    ]

    for i in range(1, n + 1):
        sid = f"scene_{i:03d}"
        if i <= len(scene_templates):
            desc = scene_templates[i - 1]
        else:
            desc = (
                f"Scene {i}: Ad-style continuation shot that maintains product focus, cinematic continuity, "
                f"and persuasive commercial pacing. Creative brief: {prompt_short[:140]}"
            )
        scenes.append({"scene_id": sid, "description": desc})
    return scenes


def _detect_scene_change_candidates(video_path: str, threshold: float = 0.08) -> List[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg scene detection failed")

    candidates: List[float] = []
    for line in proc.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            pts_text = line.split("pts_time:", 1)[1].split(" ", 1)[0].strip()
            candidates.append(float(pts_text))
        except Exception:
            continue
    # preserve order while removing duplicates caused by repeated log lines
    deduped: List[float] = []
    for value in candidates:
        if not deduped or abs(deduped[-1] - value) > 1e-3:
            deduped.append(value)
    return deduped


def _ffmpeg_zoom_in(input_path: str, output_path: str, zoom_factor: float) -> None:
    width, height = _probe_video_dimensions(input_path)
    crop_w = max(2, int(width / zoom_factor))
    crop_h = max(2, int(height / zoom_factor))
    if crop_w % 2:
        crop_w -= 1
    if crop_h % 2:
        crop_h -= 1
    vf = f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/2,scale={width}:{height}"
    video_encoder = _preferred_video_encoder()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if _aac_supported():
        cmd += ["-c:a", "aac"]
    cmd += ["-movflags", "+faststart", output_path]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg zoom-in failed")


def _ffmpeg_zoom_out(input_path: str, output_path: str, zoom_factor: float) -> None:
    width, height = _probe_video_dimensions(input_path)
    scaled_w = max(2, int(width * zoom_factor))
    scaled_h = max(2, int(height * zoom_factor))
    if scaled_w % 2:
        scaled_w -= 1
    if scaled_h % 2:
        scaled_h -= 1
    pad_x = max(0, (width - scaled_w) // 2)
    pad_y = max(0, (height - scaled_h) // 2)
    vf = f"scale={scaled_w}:{scaled_h},pad={width}:{height}:{pad_x}:{pad_y}:black"
    video_encoder = _preferred_video_encoder()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if _aac_supported():
        cmd += ["-c:a", "aac"]
    cmd += ["-movflags", "+faststart", output_path]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg zoom-out failed")


def _decide_cut_point(duration_sec: float, cut_preference: str) -> float:
    preference = cut_preference.lower().strip()
    ratio_map = {
        "early": 0.33,
        "middle": 0.50,
        "late": 0.67,
    }
    ratio = ratio_map.get(preference, 0.50)
    raw_cut = duration_sec * ratio
    return max(0.2, min(duration_sec - 0.2, raw_cut))


def _choose_content_aware_cut_point(
    duration_sec: float,
    candidate_points: List[float],
    cut_preference: Optional[str],
) -> tuple[float, Dict[str, Any]]:
    target_cut = _decide_cut_point(duration_sec, cut_preference) if cut_preference else duration_sec * 0.5
    valid_candidates = [value for value in candidate_points if 0.2 < value < duration_sec - 0.2]
    if valid_candidates:
        chosen = min(valid_candidates, key=lambda value: abs(value - target_cut))
        explanation = {
            "strategy": "scene_change_nearest_preference_window" if cut_preference else "scene_change_midpoint_fallback",
            "target_cut_sec": round(target_cut, 3),
            "candidate_cut_points": [round(value, 3) for value in valid_candidates],
            "used_fallback": False,
            "cut_preference": cut_preference,
        }
        return chosen, explanation

    return target_cut, {
        "strategy": "preference_fallback_without_scene_change" if cut_preference else "midpoint_fallback_without_scene_change",
        "target_cut_sec": round(target_cut, 3),
        "candidate_cut_points": [],
        "used_fallback": True,
        "cut_preference": cut_preference,
    }


def _augment_cut_candidates(duration_sec: float, candidate_points: List[float]) -> List[float]:
    min_margin = 0.2
    max_value = duration_sec - min_margin
    if max_value <= min_margin:
        return []

    augmented: List[float] = []

    # Always include a coarse temporal grid so selection is not forced to a single scene-change point.
    base_ratios = [0.2, 0.33, 0.5, 0.67, 0.8]
    for ratio in base_ratios:
        augmented.append(duration_sec * ratio)

    # Add local windows around detected content boundaries.
    for value in candidate_points:
        augmented.extend([value - 0.24, value - 0.12, value, value + 0.12, value + 0.24])

    clipped = []
    for value in augmented:
        clipped_value = max(min_margin, min(max_value, value))
        clipped.append(round(clipped_value, 3))

    # Deduplicate near-identical points while keeping temporal order.
    unique = sorted(set(clipped))
    merged: List[float] = []
    for value in unique:
        if not merged or abs(value - merged[-1]) > 0.06:
            merged.append(value)
    return merged


def _extract_frame_rgb_signature(video_path: str, timestamp_sec: float) -> List[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        "scale=1:1,format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = _run_subprocess_bytes(cmd)
    if proc.returncode != 0 or len(proc.stdout) < 3:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip() if isinstance(proc.stderr, (bytes, bytearray)) else ""
        raise RuntimeError(stderr or "ffmpeg frame signature failed")
    rgb = proc.stdout[:3]
    return [float(rgb[0]), float(rgb[1]), float(rgb[2])]


def _signature_distance(sig_a: List[float], sig_b: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(sig_a, sig_b))


def _choose_cross_scene_cut_point(
    from_video_path: str,
    to_video_path: str,
    duration_sec: float,
    candidate_points: List[float],
    cut_preference: Optional[str],
) -> tuple[float, Dict[str, Any]]:
    target_cut = _decide_cut_point(duration_sec, cut_preference) if cut_preference else duration_sec * 0.5
    valid_candidates = _augment_cut_candidates(duration_sec, candidate_points)
    if not valid_candidates:
        valid_candidates = [target_cut]

    to_signature = _extract_frame_rgb_signature(to_video_path, 0.05)
    scored_candidates: List[Dict[str, Any]] = []
    for value in valid_candidates:
        sample_time = max(0.0, min(duration_sec - 0.05, value - 0.04))
        from_signature = _extract_frame_rgb_signature(from_video_path, sample_time)
        distance = _signature_distance(from_signature, to_signature)
        preference_penalty = abs(value - target_cut) * 25.0 if cut_preference else 0.0
        total_score = distance + preference_penalty
        scored_candidates.append(
            {
                "cut_point_sec": round(value, 3),
                "frame_distance": round(distance, 3),
                "preference_penalty": round(preference_penalty, 3),
                "total_score": round(total_score, 3),
            }
        )

    chosen = min(scored_candidates, key=lambda item: item["total_score"])
    return float(chosen["cut_point_sec"]), {
        "strategy": "cross_scene_frame_similarity",
        "target_cut_sec": round(target_cut, 3),
        "raw_scene_change_points": [round(value, 3) for value in candidate_points if 0.2 < value < duration_sec - 0.2],
        "candidate_cut_points": [item["cut_point_sec"] for item in scored_candidates],
        "candidate_scores": scored_candidates,
        "used_fallback": not bool(candidate_points),
        "cut_preference": cut_preference,
    }


def _ffmpeg_cut_for_transition(input_path: str, lead_output_path: str, tail_output_path: str, cut_sec: float) -> None:
    duration_sec = _probe_video_duration(input_path)
    tail_duration = max(0.2, duration_sec - cut_sec)
    video_encoder = _preferred_video_encoder()

    lead_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        f"{cut_sec:.3f}",
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if _aac_supported():
        lead_cmd += ["-c:a", "aac"]
    lead_cmd += ["-movflags", "+faststart", lead_output_path]
    lead_proc = _run_subprocess(lead_cmd)
    if lead_proc.returncode != 0:
        raise RuntimeError(lead_proc.stderr.strip() or "ffmpeg lead cut failed")

    tail_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{cut_sec:.3f}",
        "-i",
        input_path,
        "-t",
        f"{tail_duration:.3f}",
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if _aac_supported():
        tail_cmd += ["-c:a", "aac"]
    tail_cmd += ["-movflags", "+faststart", tail_output_path]
    tail_proc = _run_subprocess(tail_cmd)
    if tail_proc.returncode != 0:
        raise RuntimeError(tail_proc.stderr.strip() or "ffmpeg tail cut failed")


def _ffmpeg_stitch_preview(scene_videos: List[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    list_path = os.path.join(os.path.dirname(output_path), "concat_inputs.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for video_path in scene_videos:
            f.write(f"file '{os.path.abspath(video_path)}'\n")

    video_encoder = _preferred_video_encoder()
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if _aac_supported():
        cmd += ["-c:a", "aac"]
    cmd += ["-movflags", "+faststart", output_path]
    proc = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg stitched preview failed")


class MockToolAdapter(BaseToolAdapter):
    def plan_scenes(self, prompt: str, num_scenes: int) -> ToolResult:
        scenes = _build_ad_scene_descriptions(prompt, num_scenes)
        return ToolResult(success=True, meta={"scenes": scenes})

    def generate_scene_image(self, scene_id: str, scene_desc: str, prompt: str) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "image.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"mock image artifact for {scene_id}\n")
            f.write(f"desc={scene_desc}\n")
            f.write(f"prompt={prompt}\n")
        return ToolResult(success=True, artifacts={"image_path": out_path})

    def generate_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "video.mp4")
        _make_mock_video(out_path, label=f"{scene_id}:{scene_desc[:40]}")
        return ToolResult(success=True, artifacts={"video_path": out_path})

    def zoom_in_clip(self, scene_id: str, video_path: str, zoom_factor: float = 1.15) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "video_zoom_in.mp4")
        _ffmpeg_zoom_in(video_path, out_path, zoom_factor=max(1.01, float(zoom_factor)))
        return ToolResult(
            success=True,
            artifacts={"video_path": out_path},
            meta={"qualitative_feedback": f"Applied real ffmpeg zoom-in to {scene_id} with factor {zoom_factor:.2f}."},
        )

    def zoom_out_clip(self, scene_id: str, video_path: str, zoom_factor: float = 0.85) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "video_zoom_out.mp4")
        _ffmpeg_zoom_out(video_path, out_path, zoom_factor=min(0.99, max(0.1, float(zoom_factor))))
        return ToolResult(
            success=True,
            artifacts={"video_path": out_path},
            meta={"qualitative_feedback": f"Applied real ffmpeg zoom-out to {scene_id} with factor {zoom_factor:.2f}."},
        )

    def cut_for_transition(self, scene_id: str, video_path: str, cut_preference: Optional[str] = None) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        duration_sec = _probe_video_duration(video_path)
        candidate_points = _detect_scene_change_candidates(video_path)
        cut_sec, decision_meta = _choose_content_aware_cut_point(duration_sec, candidate_points, cut_preference)
        lead_path = os.path.join(out_dir, "transition_lead.mp4")
        tail_path = os.path.join(out_dir, "transition_tail.mp4")
        _ffmpeg_cut_for_transition(video_path, lead_path, tail_path, cut_sec)
        return ToolResult(
            success=True,
            artifacts={
                "transition_lead_clip_path": lead_path,
                "transition_tail_clip_path": tail_path,
            },
            metrics={
                "cut_point_sec": round(cut_sec, 3),
                "source_duration_sec": round(duration_sec, 3),
            },
            meta={
                "qualitative_feedback": (
                    f"Chose a {cut_preference} cut point at {cut_sec:.2f}s for transition preparation."
                    if cut_preference
                    else f"Chose a content-aware cut point at {cut_sec:.2f}s for transition preparation."
                ),
                **decision_meta,
            },
        )

    def cut_for_transition_pair(
        self,
        from_scene_id: str,
        from_video_path: str,
        to_scene_id: str,
        to_video_path: str,
        cut_preference: Optional[str] = None,
    ) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, from_scene_id)
        os.makedirs(out_dir, exist_ok=True)
        duration_sec = _probe_video_duration(from_video_path)
        candidate_points = _detect_scene_change_candidates(from_video_path)
        cut_sec, decision_meta = _choose_cross_scene_cut_point(
            from_video_path=from_video_path,
            to_video_path=to_video_path,
            duration_sec=duration_sec,
            candidate_points=candidate_points,
            cut_preference=cut_preference,
        )
        lead_path = os.path.join(out_dir, f"transition_to_{to_scene_id}_lead.mp4")
        tail_path = os.path.join(out_dir, f"transition_to_{to_scene_id}_tail.mp4")
        _ffmpeg_cut_for_transition(from_video_path, lead_path, tail_path, cut_sec)
        chosen_score = None
        for item in decision_meta.get("candidate_scores", []):
            if abs(float(item["cut_point_sec"]) - cut_sec) < 1e-3:
                chosen_score = item["total_score"]
                break
        return ToolResult(
            success=True,
            artifacts={
                "transition_lead_clip_path": lead_path,
                "transition_tail_clip_path": tail_path,
            },
            metrics={
                "cut_point_sec": round(cut_sec, 3),
                "source_duration_sec": round(duration_sec, 3),
                "transition_pair_score": float(chosen_score) if chosen_score is not None else 0.0,
            },
            meta={
                "qualitative_feedback": (
                    f"Selected a cut point in {from_scene_id} that best matches the opening of {to_scene_id}"
                    + (f" under {cut_preference} preference." if cut_preference else ".")
                ),
                **decision_meta,
            },
        )

    def transition_edit(self, from_scene_id: str, to_scene_id: str, style: str) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, "transitions")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{from_scene_id}_to_{to_scene_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"mock transition from {from_scene_id} to {to_scene_id}\n")
            f.write(f"style={style}\n")
        return ToolResult(
            success=True,
            artifacts={"transition_path": out_path},
            metrics={"transition_score": round(random.uniform(0.55, 0.95), 3)},
            meta={"qualitative_feedback": f"Applied {style} transition between {from_scene_id} and {to_scene_id}."},
        )

    def trim_clip(self, scene_id: str, video_path: str, trim_start_sec: float, trim_end_sec: float) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "video_trimmed.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"mock trimmed video artifact for {scene_id}\n")
            f.write(f"source={video_path}\n")
            f.write(f"trim_start_sec={trim_start_sec}\n")
            f.write(f"trim_end_sec={trim_end_sec}\n")
        return ToolResult(
            success=True,
            artifacts={"video_path": out_path},
            meta={"qualitative_feedback": f"Trimmed {scene_id} by {trim_start_sec:.2f}s at start and {trim_end_sec:.2f}s at end."},
        )

    def stitch_preview(self, scene_videos: List[str], transition_hints: Optional[List[str]] = None) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, "preview")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "stitched_preview.mp4")
        _ffmpeg_stitch_preview(scene_videos, out_path)
        return ToolResult(
            success=True,
            artifacts={"preview_path": out_path},
            metrics={"continuity_score": round(random.uniform(0.5, 0.95), 3)},
            meta={"qualitative_feedback": "Built a stitched preview for transition and continuity inspection."},
        )

    def replace_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        out_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "video_replaced.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"mock replaced video artifact for {scene_id}\n")
            f.write(f"from_image={image_path}\n")
            f.write(f"desc={scene_desc}\n")
            f.write(f"prompt={prompt}\n")
        return ToolResult(
            success=True,
            artifacts={"video_path": out_path},
            meta={"qualitative_feedback": f"Replaced the video clip for {scene_id} with a refreshed generation."},
        )

    def judge_scene(self, scene_id: str, scene_desc: str, image_path: Optional[str], video_path: Optional[str]) -> ToolResult:
        if not image_path and not video_path:
            return ToolResult(success=False, error="No artifacts to judge")
        score = round(random.uniform(0.4, 0.95), 3)
        feedback = (
            f"Scene {scene_id} is judged with mock quality score {score}. "
            "Check visual clarity, prompt alignment, product visibility, and continuity."
        )
        return ToolResult(success=True, metrics={"auto_score": score}, meta={"qualitative_feedback": feedback})

    def judge_final(
        self,
        prompt: str,
        scene_videos: List[str],
        judge_context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if not scene_videos:
            return ToolResult(success=False, error="No videos available for final judgement")
        quality = random.uniform(0.45, 0.95)
        continuity = random.uniform(0.4, 0.95)
        result = ToolResult(
            success=True,
            metrics={
                "auto_score": round(quality, 3),
                "continuity_score": round(continuity, 3),
                "final_score": round((quality + continuity) / 2.0, 3),
            },
            meta={
                "qualitative_feedback": (
                    "Mock final judge completed. Use this feedback slot for alignment, "
                    "continuity, product visibility, and viewer/persona preference notes."
                )
            },
        )
        self._attach_ad_ranker_metrics(result, prompt, scene_videos, judge_context)
        return result


class ScriptToolAdapter(BaseToolAdapter):
    """
    Thin adapter that can call real scripts via configured command templates.
    Command templates use .format(...) with keys documented per method.
    """

    def __init__(self, config: ToolAdapterConfig, command_templates: Optional[Dict[str, str]] = None):
        super().__init__(config)
        self.command_templates = command_templates or {}

    def _run_template(self, key: str, **kwargs: str) -> ToolResult:
        template = self.command_templates.get(key)
        if not template:
            return ToolResult(success=False, error=f"Missing command template for action: {key}")
        cmd = template.format(**kwargs)
        start = time.time()
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error=f"Command failed ({proc.returncode}): {proc.stderr.strip()}",
                meta={"stdout": proc.stdout, "stderr": proc.stderr, "elapsed_sec": elapsed},
            )
        return ToolResult(success=True, meta={"stdout": proc.stdout, "stderr": proc.stderr, "elapsed_sec": elapsed})

    def _run_template_optional(self, key: str, **kwargs: str) -> ToolResult:
        template = self.command_templates.get(key)
        if not template:
            return ToolResult(success=True, meta={"note": f"No command template configured for {key}; used local stub."})
        return self._run_template(key, **kwargs)

    def plan_scenes(self, prompt: str, num_scenes: int) -> ToolResult:
        result = self._run_template("plan_scenes", prompt=prompt, num_scenes=str(num_scenes))
        if not result.success:
            return result
        # Fallback: create trivial plan if command did not emit a plan file.
        result.meta["scenes"] = _build_ad_scene_descriptions(prompt, num_scenes)
        return result

    def generate_scene_image(self, scene_id: str, scene_desc: str, prompt: str) -> ToolResult:
        result = self._run_template("generate_scene_image", scene_id=scene_id, scene_desc=scene_desc, prompt=prompt)
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, scene_id, "image.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"scene_id": scene_id, "scene_desc": scene_desc}, ensure_ascii=False))
        result.artifacts["image_path"] = path
        return result

    def generate_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        result = self._run_template(
            "generate_scene_video",
            scene_id=scene_id,
            scene_desc=scene_desc,
            prompt=prompt,
            image_path=image_path,
        )
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, scene_id, "video.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"scene_id": scene_id, "image_path": image_path}, ensure_ascii=False))
        result.artifacts["video_path"] = path
        return result

    def zoom_in_clip(self, scene_id: str, video_path: str, zoom_factor: float = 1.15) -> ToolResult:
        template = self.command_templates.get("zoom_in_clip")
        if template:
            result = self._run_template(
                "zoom_in_clip",
                scene_id=scene_id,
                video_path=video_path,
                zoom_factor=str(zoom_factor),
            )
            if not result.success:
                return result
            path = os.path.join(self.artifacts_dir, scene_id, "video_zoom_in.mp4")
            result.artifacts["video_path"] = path
            result.meta.setdefault("qualitative_feedback", f"Applied scripted zoom-in to {scene_id}.")
            return result

        path = os.path.join(self.artifacts_dir, scene_id, "video_zoom_in.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _ffmpeg_zoom_in(video_path, path, zoom_factor=max(1.01, float(zoom_factor)))
        return ToolResult(
            success=True,
            artifacts={"video_path": path},
            meta={"qualitative_feedback": f"Applied built-in ffmpeg zoom-in to {scene_id}."},
        )

    def zoom_out_clip(self, scene_id: str, video_path: str, zoom_factor: float = 0.85) -> ToolResult:
        template = self.command_templates.get("zoom_out_clip")
        if template:
            result = self._run_template(
                "zoom_out_clip",
                scene_id=scene_id,
                video_path=video_path,
                zoom_factor=str(zoom_factor),
            )
            if not result.success:
                return result
            path = os.path.join(self.artifacts_dir, scene_id, "video_zoom_out.mp4")
            result.artifacts["video_path"] = path
            result.meta.setdefault("qualitative_feedback", f"Applied scripted zoom-out to {scene_id}.")
            return result

        path = os.path.join(self.artifacts_dir, scene_id, "video_zoom_out.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _ffmpeg_zoom_out(video_path, path, zoom_factor=min(0.99, max(0.1, float(zoom_factor))))
        return ToolResult(
            success=True,
            artifacts={"video_path": path},
            meta={"qualitative_feedback": f"Applied built-in ffmpeg zoom-out to {scene_id}."},
        )

    def cut_for_transition(self, scene_id: str, video_path: str, cut_preference: Optional[str] = None) -> ToolResult:
        template = self.command_templates.get("cut_for_transition")
        if template:
            result = self._run_template(
                "cut_for_transition",
                scene_id=scene_id,
                video_path=video_path,
                cut_preference=cut_preference,
            )
            if not result.success:
                return result
            base_dir = os.path.join(self.artifacts_dir, scene_id)
            result.artifacts.setdefault("transition_lead_clip_path", os.path.join(base_dir, "transition_lead.mp4"))
            result.artifacts.setdefault("transition_tail_clip_path", os.path.join(base_dir, "transition_tail.mp4"))
            result.meta.setdefault("qualitative_feedback", f"Prepared transition cut for {scene_id}.")
            return result

        base_dir = os.path.join(self.artifacts_dir, scene_id)
        os.makedirs(base_dir, exist_ok=True)
        duration_sec = _probe_video_duration(video_path)
        candidate_points = _detect_scene_change_candidates(video_path)
        cut_sec, decision_meta = _choose_content_aware_cut_point(duration_sec, candidate_points, cut_preference)
        lead_path = os.path.join(base_dir, "transition_lead.mp4")
        tail_path = os.path.join(base_dir, "transition_tail.mp4")
        _ffmpeg_cut_for_transition(video_path, lead_path, tail_path, cut_sec)
        return ToolResult(
            success=True,
            artifacts={
                "transition_lead_clip_path": lead_path,
                "transition_tail_clip_path": tail_path,
            },
            metrics={
                "cut_point_sec": round(cut_sec, 3),
                "source_duration_sec": round(duration_sec, 3),
            },
            meta={
                "qualitative_feedback": f"Prepared transition cut for {scene_id} using built-in ffmpeg logic.",
                **decision_meta,
            },
        )

    def cut_for_transition_pair(
        self,
        from_scene_id: str,
        from_video_path: str,
        to_scene_id: str,
        to_video_path: str,
        cut_preference: Optional[str] = None,
    ) -> ToolResult:
        template = self.command_templates.get("cut_for_transition_pair")
        if template:
            result = self._run_template(
                "cut_for_transition_pair",
                from_scene_id=from_scene_id,
                from_video_path=from_video_path,
                to_scene_id=to_scene_id,
                to_video_path=to_video_path,
                cut_preference=cut_preference,
            )
            if not result.success:
                return result
            base_dir = os.path.join(self.artifacts_dir, from_scene_id)
            result.artifacts.setdefault("transition_lead_clip_path", os.path.join(base_dir, f"transition_to_{to_scene_id}_lead.mp4"))
            result.artifacts.setdefault("transition_tail_clip_path", os.path.join(base_dir, f"transition_to_{to_scene_id}_tail.mp4"))
            result.meta.setdefault("qualitative_feedback", f"Prepared pairwise transition cut for {from_scene_id} -> {to_scene_id}.")
            return result

        base_dir = os.path.join(self.artifacts_dir, from_scene_id)
        os.makedirs(base_dir, exist_ok=True)
        duration_sec = _probe_video_duration(from_video_path)
        candidate_points = _detect_scene_change_candidates(from_video_path)
        cut_sec, decision_meta = _choose_cross_scene_cut_point(
            from_video_path=from_video_path,
            to_video_path=to_video_path,
            duration_sec=duration_sec,
            candidate_points=candidate_points,
            cut_preference=cut_preference,
        )
        lead_path = os.path.join(base_dir, f"transition_to_{to_scene_id}_lead.mp4")
        tail_path = os.path.join(base_dir, f"transition_to_{to_scene_id}_tail.mp4")
        _ffmpeg_cut_for_transition(from_video_path, lead_path, tail_path, cut_sec)
        chosen_score = None
        for item in decision_meta.get("candidate_scores", []):
            if abs(float(item["cut_point_sec"]) - cut_sec) < 1e-3:
                chosen_score = item["total_score"]
                break
        return ToolResult(
            success=True,
            artifacts={
                "transition_lead_clip_path": lead_path,
                "transition_tail_clip_path": tail_path,
            },
            metrics={
                "cut_point_sec": round(cut_sec, 3),
                "source_duration_sec": round(duration_sec, 3),
                "transition_pair_score": float(chosen_score) if chosen_score is not None else 0.0,
            },
            meta={
                "qualitative_feedback": f"Prepared pairwise transition cut for {from_scene_id} -> {to_scene_id}.",
                **decision_meta,
            },
        )

    def transition_edit(self, from_scene_id: str, to_scene_id: str, style: str) -> ToolResult:
        result = self._run_template_optional(
            "transition_edit",
            from_scene_id=from_scene_id,
            to_scene_id=to_scene_id,
            style=style,
        )
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, "transitions", f"{from_scene_id}_to_{to_scene_id}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"from_scene_id": from_scene_id, "to_scene_id": to_scene_id, "style": style}, ensure_ascii=False))
        result.artifacts["transition_path"] = path
        result.metrics.setdefault("transition_score", 0.7)
        result.meta.setdefault("qualitative_feedback", f"Transition {style} applied between {from_scene_id} and {to_scene_id}.")
        return result

    def trim_clip(self, scene_id: str, video_path: str, trim_start_sec: float, trim_end_sec: float) -> ToolResult:
        result = self._run_template_optional(
            "trim_clip",
            scene_id=scene_id,
            video_path=video_path,
            trim_start_sec=str(trim_start_sec),
            trim_end_sec=str(trim_end_sec),
        )
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, scene_id, "video_trimmed.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"scene_id": scene_id, "video_path": video_path, "trim_start_sec": trim_start_sec, "trim_end_sec": trim_end_sec}, ensure_ascii=False))
        result.artifacts["video_path"] = path
        result.meta.setdefault("qualitative_feedback", f"Trimmed clip for {scene_id}.")
        return result

    def stitch_preview(self, scene_videos: List[str], transition_hints: Optional[List[str]] = None) -> ToolResult:
        result = self._run_template_optional(
            "stitch_preview",
            scene_videos=" ".join(scene_videos),
            transition_hints=" | ".join(transition_hints or []),
        )
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, "preview", "stitched_preview.mp4")
        _ffmpeg_stitch_preview(scene_videos, path)
        result.artifacts["preview_path"] = path
        result.metrics.setdefault("continuity_score", 0.7)
        result.meta.setdefault("qualitative_feedback", "Generated stitched preview.")
        return result

    def replace_scene_video(self, scene_id: str, scene_desc: str, prompt: str, image_path: str) -> ToolResult:
        result = self._run_template_optional(
            "replace_scene_video",
            scene_id=scene_id,
            scene_desc=scene_desc,
            prompt=prompt,
            image_path=image_path,
        )
        if not result.success:
            return result
        path = os.path.join(self.artifacts_dir, scene_id, "video_replaced.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"scene_id": scene_id, "image_path": image_path, "scene_desc": scene_desc, "prompt": prompt}, ensure_ascii=False))
        result.artifacts["video_path"] = path
        result.meta.setdefault("qualitative_feedback", f"Replaced scene video for {scene_id}.")
        return result

    def judge_scene(self, scene_id: str, scene_desc: str, image_path: Optional[str], video_path: Optional[str]) -> ToolResult:
        result = self._run_template(
            "judge_scene",
            scene_id=scene_id,
            scene_desc=scene_desc,
            image_path=image_path or "",
            video_path=video_path or "",
        )
        if not result.success:
            return result
        result.metrics["auto_score"] = result.metrics.get("auto_score", 0.7)
        return result

    def judge_final(
        self,
        prompt: str,
        scene_videos: List[str],
        judge_context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        result = self._run_template("judge_final", prompt=prompt, scene_videos=" ".join(scene_videos))
        if not result.success:
            return result
        result.metrics.setdefault("auto_score", 0.75)
        result.metrics.setdefault("continuity_score", 0.7)
        result.metrics.setdefault("final_score", 0.725)
        result.meta.setdefault(
            "qualitative_feedback",
            "Script judge completed. Review final alignment, continuity, product prominence, and persona preference.",
        )
        self._attach_ad_ranker_metrics(result, prompt, scene_videos, judge_context)
        return result
