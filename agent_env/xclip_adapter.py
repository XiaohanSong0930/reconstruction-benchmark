from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

def _add_xclip_repo_to_path(xclip_repo: str) -> None:
    repo = os.path.abspath(xclip_repo)
    xclip_dir = os.path.join(repo, "X-CLIP")
    if xclip_dir not in sys.path:
        sys.path.insert(0, xclip_dir)


def _load_xclip_modules(xclip_repo: str):
    _add_xclip_repo_to_path(xclip_repo)
    for key in list(sys.modules.keys()):
        if key == "clip" or key.startswith("clip."):
            sys.modules.pop(key, None)
    xclip_clip = importlib.import_module("clip")
    from models.xclip import build_model  # type: ignore

    return xclip_clip, build_model


def _load_checkpoint_state(checkpoint_path: str) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _infer_num_frames_from_state(state_dict: Dict[str, Any], fallback: int) -> int:
    positional = state_dict.get("mit.positional_embedding")
    if hasattr(positional, "shape") and len(positional.shape) >= 2:
        try:
            return int(positional.shape[1])
        except Exception:
            pass
    return fallback


def _load_model(
    *,
    xclip_repo: str,
    checkpoint_path: str,
    arch: str,
    num_frames: int,
    device: str,
):
    xclip_clip, build_model = _load_xclip_modules(xclip_repo)
    state_dict = _load_checkpoint_state(checkpoint_path)
    effective_num_frames = _infer_num_frames_from_state(state_dict, num_frames)
    model = build_model(
        state_dict,
        T=effective_num_frames,
        droppath=0.0,
        use_checkpoint=False,
        logger=_NullLogger(),
        prompts_alpha=1e-1,
        prompts_layers=2,
        use_cache=False,
        mit_layers=1,
    )
    model = model.to(device)
    model.eval()
    input_resolution = getattr(model, "input_resolution", None)
    if input_resolution is None:
        input_resolution = getattr(getattr(model, "visual", None), "input_resolution", None)
    if input_resolution is None:
        raise AttributeError("Unable to determine X-CLIP input resolution from model or model.visual")
    preprocess = _build_preprocess(input_resolution)
    tokenize = xclip_clip.tokenize
    return model, preprocess, tokenize, effective_num_frames


class _NullLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None


def _build_preprocess(input_resolution: Any):
    n_px = int(input_resolution.item()) if hasattr(input_resolution, "item") else int(input_resolution)
    return Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def _extract_frames(video_path: str, output_folder: str, fps: float = 1.0) -> None:
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        f"{output_folder}/frame_%04d.jpg",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"ffmpeg failed for {video_path}")


def _sample_video_frames(video_path: str, out_dir: str, num_frames: int) -> List[str]:
    _extract_frames(video_path, out_dir, fps=1.0)
    frames = sorted(str(p) for p in Path(out_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    if len(frames) >= num_frames:
        if len(frames) == num_frames:
            return frames
        idxs = []
        for i in range(num_frames):
            pos = round(i * (len(frames) - 1) / max(num_frames - 1, 1))
            idxs.append(pos)
        return [frames[i] for i in idxs]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def _load_video_tensor(frame_paths: List[str], preprocess, device: str) -> torch.Tensor:
    tensors = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        tensors.append(preprocess(image))
    video = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    return video


def _load_spec_text(text_file: str) -> str:
    path = Path(text_file)
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        return raw
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw
    visual_concepts = payload.get("visual_concepts") or []
    scene_sequence = payload.get("scene_sequence") or []
    parts = []
    if payload.get("brand_or_product"):
        parts.append(f"Brand: {payload['brand_or_product']}.")
    if payload.get("theme_message"):
        parts.append(f"Message: {payload['theme_message']}.")
    if payload.get("emotional_tone"):
        parts.append(f"Tone: {payload['emotional_tone']}.")
    if payload.get("opening_hook"):
        parts.append(f"Opening: {payload['opening_hook']}.")
    if visual_concepts:
        parts.append(f"Visuals: {', '.join(map(str, visual_concepts[:3]))}.")
    if scene_sequence:
        parts.append(f"Scenes: {'; '.join(map(str, scene_sequence[:3]))}.")
    if payload.get("closing_payoff"):
        parts.append(f"Closing: {payload['closing_payoff']}.")
    if not parts and payload.get("scene_summary"):
        parts.append(str(payload["scene_summary"]))
    return " ".join(parts) if parts else raw


def _fit_spec_text_for_tokenizer(spec_text: str, tokenize) -> str:
    compact = " ".join(spec_text.split())
    candidates = [compact]
    clauses = [part.strip() for part in compact.split(".") if part.strip()]
    if len(clauses) > 1:
        for keep in range(len(clauses) - 1, 0, -1):
            candidates.append(". ".join(clauses[:keep]) + ".")
    words = compact.split()
    for limit in [64, 48, 40, 32, 24, 20, 16, 12]:
        if len(words) > limit:
            candidates.append(" ".join(words[:limit]))
    last_error = None
    for candidate in candidates:
        try:
            tokenize([candidate])
            return candidate
        except RuntimeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return compact


def _compute_alignment_score(
    *,
    model,
    tokenize,
    spec_text: str,
    video_tensor: torch.Tensor,
    device: str,
) -> float:
    with torch.no_grad():
        safe_spec_text = _fit_spec_text_for_tokenizer(spec_text, tokenize)
        text_tokens = tokenize([safe_spec_text]).to(device)
        video_features, _ = model.encode_video(video_tensor)
        text_features = model.encode_text(text_tokens)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = torch.matmul(video_features, text_features.t()).squeeze().item()
    normalized = (score + 1.0) / 2.0
    return max(0.0, min(1.0, float(normalized)))


def evaluate_once(
    *,
    xclip_repo: str,
    checkpoint_path: str,
    arch: str,
    video_path: str,
    text_file: str,
    output_json: str,
    num_frames: int,
    device: str,
    pass_threshold: float,
) -> Dict[str, Any]:
    model, preprocess, tokenize, effective_num_frames = _load_model(
        xclip_repo=xclip_repo,
        checkpoint_path=checkpoint_path,
        arch=arch,
        num_frames=num_frames,
        device=device,
    )
    spec_text = _load_spec_text(text_file)
    with tempfile.TemporaryDirectory(prefix="xclip_frames_") as tmpdir:
        frame_paths = _sample_video_frames(video_path, tmpdir, num_frames=effective_num_frames)
        video_tensor = _load_video_tensor(frame_paths, preprocess, device)
        score = _compute_alignment_score(
            model=model,
            tokenize=tokenize,
            spec_text=spec_text,
            video_tensor=video_tensor,
            device=device,
        )
    payload = {
        "backend": "xclip",
        "score": round(score, 6),
        "pass": bool(score >= pass_threshold),
        "arch": arch,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "video_path": os.path.abspath(video_path),
        "text_file": os.path.abspath(text_file),
        "num_frames": effective_num_frames,
        "device": device,
    }
    Path(output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local X-CLIP text-video alignment adapter.")
    parser.add_argument("--xclip-repo", default="/home/sxh/VideoX", help="Path to cloned VideoX repo")
    parser.add_argument("--checkpoint", required=True, help="Path to X-CLIP checkpoint")
    parser.add_argument("--arch", default="ViT-B/16", help="X-CLIP architecture name")
    parser.add_argument("--video", required=True, help="Video path to score")
    parser.add_argument("--text-file", required=True, help="Text file containing specification text")
    parser.add_argument("--output", required=True, help="Where to write JSON result")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_once(
        xclip_repo=args.xclip_repo,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        video_path=args.video,
        text_file=args.text_file,
        output_json=args.output,
        num_frames=args.num_frames,
        device=args.device,
        pass_threshold=args.pass_threshold,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
