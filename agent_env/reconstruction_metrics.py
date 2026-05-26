from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Sequence

def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


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
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(stderr or stdout or f"ffprobe failed for {video_path}")
    try:
        return float((proc.stdout or "0").strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid ffprobe duration output for {video_path}: {proc.stdout!r}") from exc


def _extract_rgb_frame(video_path: str, timestamp_sec: float, width: int = 32, height: int = 32) -> bytes:
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
        f"scale={width}:{height},format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = _run(cmd)
    expected = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < expected:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or f"Failed to extract frame from {video_path}")
    return bytes(proc.stdout[:expected])


def _sample_timestamps(video_a: str, video_b: str, samples: int) -> List[float]:
    duration = max(0.1, min(_probe_video_duration(video_a), _probe_video_duration(video_b)))
    if samples <= 1:
        return [min(duration * 0.5, max(0.0, duration - 0.05))]
    start = min(0.05, duration * 0.2)
    end = max(start, duration - 0.05)
    step = (end - start) / float(samples - 1)
    return [start + i * step for i in range(samples)]


def _rgb_signature(frame: bytes) -> List[float]:
    total_r = 0
    total_g = 0
    total_b = 0
    count = len(frame) // 3
    for i in range(0, len(frame), 3):
        total_r += frame[i]
        total_g += frame[i + 1]
        total_b += frame[i + 2]
    return [total_r / count, total_g / count, total_b / count]


def _signature_distance(sig_a: Sequence[float], sig_b: Sequence[float]) -> float:
    return float(sum(abs(a - b) for a, b in zip(sig_a, sig_b)))


def _frame_histogram(frame: bytes, bins_per_channel: int = 8) -> List[float]:
    bins = [0] * (bins_per_channel ** 3)
    step = 256 // bins_per_channel
    count = len(frame) // 3
    for i in range(0, len(frame), 3):
        r = min(bins_per_channel - 1, frame[i] // step)
        g = min(bins_per_channel - 1, frame[i + 1] // step)
        b = min(bins_per_channel - 1, frame[i + 2] // step)
        idx = (r * bins_per_channel + g) * bins_per_channel + b
        bins[idx] += 1
    return [value / float(count) for value in bins]


def _l1_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return float(sum(abs(a - b) for a, b in zip(vec_a, vec_b)))


def _frame_delta(frame_a: bytes, frame_b: bytes) -> float:
    if len(frame_a) != len(frame_b):
        raise ValueError("Frame sizes must match for temporal delta comparison.")
    total = 0.0
    count = len(frame_a)
    for a, b in zip(frame_a, frame_b):
        total += abs(a - b)
    return total / float(count)


def _legacy_rgb_similarity(video_a: str, video_b: str, samples: int = 8) -> Dict[str, Any]:
    distances: List[float] = []
    timestamps = _sample_timestamps(video_a, video_b, samples)
    for ts in timestamps:
        frame_a = _extract_rgb_frame(video_a, ts, width=1, height=1)
        frame_b = _extract_rgb_frame(video_b, ts, width=1, height=1)
        distances.append(_signature_distance(_rgb_signature(frame_a), _rgb_signature(frame_b)))
    avg_distance = sum(distances) / len(distances)
    similarity = max(0.0, 1.0 - (avg_distance / (255.0 * 3.0)))
    return {
        "metric": "legacy_rgb_signature",
        "samples": samples,
        "avg_distance": round(avg_distance, 6),
        "score": round(similarity, 6),
    }


def _frame_hist_similarity(video_a: str, video_b: str, samples: int = 8) -> Dict[str, Any]:
    distances: List[float] = []
    timestamps = _sample_timestamps(video_a, video_b, samples)
    for ts in timestamps:
        frame_a = _extract_rgb_frame(video_a, ts)
        frame_b = _extract_rgb_frame(video_b, ts)
        hist_a = _frame_histogram(frame_a)
        hist_b = _frame_histogram(frame_b)
        distances.append(_l1_distance(hist_a, hist_b) / 2.0)
    avg_distance = sum(distances) / len(distances)
    return {
        "metric": "frame_histogram_similarity",
        "samples": samples,
        "avg_hist_l1_distance": round(avg_distance, 6),
        "score": round(max(0.0, 1.0 - avg_distance), 6),
    }


def _temporal_delta_similarity(video_a: str, video_b: str, samples: int = 8) -> Dict[str, Any]:
    timestamps = _sample_timestamps(video_a, video_b, max(3, samples))
    frames_a = [_extract_rgb_frame(video_a, ts) for ts in timestamps]
    frames_b = [_extract_rgb_frame(video_b, ts) for ts in timestamps]
    deltas_a = [_frame_delta(frames_a[i], frames_a[i + 1]) for i in range(len(frames_a) - 1)]
    deltas_b = [_frame_delta(frames_b[i], frames_b[i + 1]) for i in range(len(frames_b) - 1)]
    raw_distance = sum(abs(a - b) for a, b in zip(deltas_a, deltas_b)) / len(deltas_a)
    normalized = min(1.0, raw_distance / 255.0)
    return {
        "metric": "temporal_delta_similarity",
        "samples": len(timestamps),
        "avg_temporal_delta_distance": round(raw_distance, 6),
        "score": round(max(0.0, 1.0 - normalized), 6),
    }


def _parse_last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        value = matches[-1].lower()
        if value in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        return float(value)
    except ValueError:
        return None


def _ffmpeg_ssim(video_a: str, video_b: str) -> Dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_a,
        "-i",
        video_b,
        "-lavfi",
        "[0:v][1:v]ssim",
        "-f",
        "null",
        "-",
    ]
    proc = _run(cmd)
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    all_ssim = _parse_last_float(r"All:([0-9.]+|inf|INF|infinity|INFINITY)", stderr)
    if proc.returncode != 0 or all_ssim is None:
        raise RuntimeError(stderr.strip() or "ffmpeg ssim failed")
    return {
        "metric": "ffmpeg_ssim",
        "score": round(all_ssim, 6),
    }


def _ffmpeg_psnr(video_a: str, video_b: str) -> Dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_a,
        "-i",
        video_b,
        "-lavfi",
        "[0:v][1:v]psnr",
        "-f",
        "null",
        "-",
    ]
    proc = _run(cmd)
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    avg_psnr = _parse_last_float(r"average:([0-9.]+|inf|INF|infinity|INFINITY)", stderr)
    if proc.returncode != 0 or avg_psnr is None:
        raise RuntimeError(stderr.strip() or "ffmpeg psnr failed")
    if math.isinf(avg_psnr):
        normalized = 1.0
    else:
        normalized = 1.0 - math.exp(-avg_psnr / 20.0)
    return {
        "metric": "ffmpeg_psnr",
        "psnr_db": "inf" if math.isinf(avg_psnr) else round(avg_psnr, 6),
        "score": round(max(0.0, min(1.0, normalized)), 6),
    }


def compute_reconstruction_metrics(video_a: str, video_b: str, samples: int = 8) -> Dict[str, Any]:
    legacy = _legacy_rgb_similarity(video_a, video_b, samples=samples)
    hist = _frame_hist_similarity(video_a, video_b, samples=samples)
    temporal = _temporal_delta_similarity(video_a, video_b, samples=samples)
    ssim = _ffmpeg_ssim(video_a, video_b)
    psnr = _ffmpeg_psnr(video_a, video_b)
    composite = (
        0.15 * legacy["score"]
        + 0.25 * hist["score"]
        + 0.2 * temporal["score"]
        + 0.25 * ssim["score"]
        + 0.15 * psnr["score"]
    )
    return {
        "video_a": os.path.abspath(video_a),
        "video_b": os.path.abspath(video_b),
        "metrics": {
            "legacy_rgb_signature": legacy,
            "frame_histogram_similarity": hist,
            "temporal_delta_similarity": temporal,
            "ffmpeg_ssim": ssim,
            "ffmpeg_psnr": psnr,
        },
        "composite_score": round(composite, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reference-based reconstruction metrics for two videos.")
    parser.add_argument("--video-a", required=True, help="Candidate or generated video")
    parser.add_argument("--video-b", required=True, help="Reference video")
    parser.add_argument("--samples", type=int, default=8, help="Number of sampled timestamps for framewise metrics")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    payload = compute_reconstruction_metrics(args.video_a, args.video_b, samples=args.samples)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(os.path.abspath(args.output), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
