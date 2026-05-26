from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from demo_pipe1 import extract_frames


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _normalize_text(text).split(" ") if tok]


def _unique_tokens(texts: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for text in texts:
        for tok in _tokenize(text):
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
    return ordered


def build_spec_text(specification: Dict[str, Any]) -> str:
    parts = [
        f"Brand or product: {specification.get('brand_or_product', '')}",
        f"Target audience: {specification.get('target_audience', '')}",
        f"Theme message: {specification.get('theme_message', '')}",
        f"Persuasion strategy: {specification.get('persuasion_strategy', '')}",
        f"Emotional tone: {specification.get('emotional_tone', '')}",
        f"Visual concepts: {', '.join(specification.get('visual_concepts') or [])}",
        f"Camera style: {specification.get('camera_style', '')}",
        f"Pacing: {specification.get('pacing', '')}",
        f"Scene summary: {specification.get('scene_summary', '')}",
        f"Scene sequence: {' | '.join(str(x) for x in (specification.get('scene_sequence') or []))}",
        f"Opening hook: {specification.get('opening_hook', '')}",
        f"Closing payoff: {specification.get('closing_payoff', '')}",
        f"Reconstruction goal: {specification.get('reconstruction_goal', '')}",
        f"Style keywords: {', '.join(specification.get('style_keywords') or [])}",
    ]
    return "\n".join(parts)


def _run_cmd(cmd: List[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def evaluate_xclip_alignment(
    *,
    output_video: str,
    specification: Dict[str, Any],
    command_template: str,
) -> Dict[str, Any]:
    if not command_template:
        raise ValueError("xclip command_template is required for xclip alignment evaluation")

    spec_text = build_spec_text(specification)
    with tempfile.TemporaryDirectory(prefix="xclip_eval_") as tmpdir:
        text_file = os.path.join(tmpdir, "spec.txt")
        output_json = os.path.join(tmpdir, "result.json")
        Path(text_file).write_text(spec_text, encoding="utf-8")
        formatted = command_template.format(
            video=shlex.quote(output_video),
            text_file=shlex.quote(text_file),
            output_json=shlex.quote(output_json),
        )
        cmd = shlex.split(formatted)
        code, stdout, stderr = _run_cmd(cmd)
        if code != 0:
            raise RuntimeError(f"xclip command failed: {stderr or stdout}")

        payload: Dict[str, Any]
        if os.path.exists(output_json):
            payload = json.loads(Path(output_json).read_text(encoding="utf-8"))
        else:
            text = (stdout or "").strip()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"score": float(text)}

        score = float(payload.get("score", 0.0))
        passed = bool(payload.get("pass", score >= 0.6))
        return {
            "backend": "xclip",
            "command": cmd,
            "score": score,
            "pass": passed,
            "raw": payload,
        }


def _extract_eval_frames(video_path: str, out_dir: str, fps: float = 1.0, max_frames: int = 8) -> List[str]:
    extract_frames(video_path, out_dir, fps=fps)
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))[:max_frames]


def _ocr_frame_with_tesseract(frame_path: str) -> str:
    code, stdout, stderr = _run_cmd(["tesseract", frame_path, "stdout", "--psm", "6"])
    if code != 0:
        raise RuntimeError(stderr or stdout or f"tesseract failed for {frame_path}")
    return stdout


def evaluate_ocr_text_fidelity(
    *,
    output_video: str,
    specification: Dict[str, Any],
    fps: float = 1.0,
    max_frames: int = 8,
) -> Dict[str, Any]:
    frame_dir = os.path.join(os.path.dirname(output_video), "ocr_eval_frames")
    frame_paths = _extract_eval_frames(output_video, frame_dir, fps=fps, max_frames=max_frames)
    if shutil.which("tesseract") is None:
        return {
            "backend": "ocr",
            "available": False,
            "frame_dir": frame_dir,
            "frame_paths": frame_paths,
            "score": None,
            "pass": None,
            "reasoning": "tesseract not available in environment",
        }

    ocr_texts: List[str] = []
    for frame in frame_paths:
        try:
            ocr_texts.append(_ocr_frame_with_tesseract(frame))
        except Exception:
            continue

    combined = "\n".join(ocr_texts)
    normalized = _normalize_text(combined)

    brand = str(specification.get("brand_or_product") or "")
    closing_payoff = str(specification.get("closing_payoff") or "")
    theme_message = str(specification.get("theme_message") or "")

    brand_tokens = _unique_tokens([brand])[:4]
    payoff_tokens = _unique_tokens([closing_payoff])[:6]
    theme_tokens = _unique_tokens([theme_message])[:6]

    def frac(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for tok in tokens if tok in normalized)
        return hits / len(tokens)

    brand_score = frac(brand_tokens)
    payoff_score = frac(payoff_tokens)
    theme_score = frac(theme_tokens)
    score = round(0.5 * brand_score + 0.3 * payoff_score + 0.2 * theme_score, 6)
    passed = brand_score >= 0.5 and score >= 0.5

    return {
        "backend": "ocr",
        "available": True,
        "frame_dir": frame_dir,
        "frame_paths": frame_paths,
        "ocr_text": combined,
        "brand_tokens": brand_tokens,
        "closing_payoff_tokens": payoff_tokens,
        "theme_tokens": theme_tokens,
        "brand_score": round(brand_score, 6),
        "closing_payoff_score": round(payoff_score, 6),
        "theme_score": round(theme_score, 6),
        "score": score,
        "pass": passed,
    }


def combine_skillbench_scores(
    *,
    semantic_score: Optional[float],
    fidelity_score: Optional[float],
    text_score: Optional[float],
) -> Dict[str, Any]:
    semantic = float(semantic_score or 0.0)
    fidelity = float(fidelity_score or 0.0)
    text = float(text_score or 0.0)
    final_score = round(0.5 * semantic + 0.4 * fidelity + 0.1 * text, 6)
    # OCR is informative but too brittle to act as a hard gate on pass/fail.
    passed = semantic >= 0.6 and fidelity >= 0.5 and final_score >= 0.56
    return {
        "score": final_score,
        "pass": passed,
        "weights": {"semantic": 0.5, "fidelity": 0.4, "text": 0.1},
        "components": {
            "semantic_score": semantic,
            "fidelity_score": fidelity,
            "text_score": text,
        },
        "gates": {
            "semantic_min": 0.6,
            "fidelity_min": 0.5,
            "final_score_min": 0.56,
            "text_score_hard_gate": None,
        },
    }
