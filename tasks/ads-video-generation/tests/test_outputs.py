from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/root/output")
VIDEO_PATH = OUTPUT_DIR / "generated_video.mp4"
EVAL_PATH = OUTPUT_DIR / "eval.json"


REQUIRED_EVAL_FIELDS = {
    "semantic_alignment_score": float,
    "semantic_alignment_pass": bool,
    "reference_fidelity_composite_score": float,
    "ocr_text_score": (int, float),
    "ocr_text_pass": bool,
    "skillbench_final_score": float,
    "skillbench_final_pass": bool,
}


def _load_eval() -> dict:
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def test_generated_video_exists() -> None:
    assert VIDEO_PATH.exists(), f"Missing generated video: {VIDEO_PATH}"
    assert VIDEO_PATH.stat().st_size > 0, "Generated video is empty"


def test_generated_video_is_valid_mp4() -> None:
    duration = _probe_duration(VIDEO_PATH)
    assert duration > 0.0, "Generated video has non-positive duration"


def test_eval_json_exists() -> None:
    assert EVAL_PATH.exists(), f"Missing eval summary: {EVAL_PATH}"
    assert EVAL_PATH.stat().st_size > 0, "Eval summary JSON is empty"


def test_eval_json_schema() -> None:
    payload = _load_eval()
    for key, expected_type in REQUIRED_EVAL_FIELDS.items():
        assert key in payload, f"Missing eval field: {key}"
        assert isinstance(payload[key], expected_type), (
            f"Field {key} has wrong type: expected {expected_type}, got {type(payload[key])}"
        )


def test_eval_scores_are_bounded() -> None:
    payload = _load_eval()
    bounded_fields = [
        "semantic_alignment_score",
        "reference_fidelity_composite_score",
        "ocr_text_score",
        "skillbench_final_score",
    ]
    for field in bounded_fields:
        value = float(payload[field])
        assert 0.0 <= value <= 1.0, f"{field} is out of range: {value}"


def test_eval_pass_logic() -> None:
    payload = _load_eval()
    semantic = float(payload["semantic_alignment_score"])
    fidelity = float(payload["reference_fidelity_composite_score"])
    final_score = float(payload["skillbench_final_score"])
    expected_pass = semantic >= 0.6 and fidelity >= 0.5 and final_score >= 0.56
    assert payload["skillbench_final_pass"] is expected_pass, (
        "Final pass flag does not match deterministic rubric"
    )


def test_semantic_and_fidelity_are_present_for_valid_solution() -> None:
    payload = _load_eval()
    assert float(payload["semantic_alignment_score"]) > 0.0, "Semantic alignment score should be non-zero"
    assert float(payload["reference_fidelity_composite_score"]) > 0.0, "Reference fidelity score should be non-zero"
