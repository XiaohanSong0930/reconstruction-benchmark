#!/usr/bin/env bash
set -euo pipefail

INPUT_SPEC="/root/input/specification.json"
OUTPUT_VIDEO="/root/output/generated_video.mp4"
OUTPUT_EVAL="/root/output/eval.json"
WORKDIR="/workspace"

mkdir -p /root/output

# Expected environment variables:
#   GOOGLE_API_KEY
#   OPENAI_API_KEY
# Optional overrides:
#   GEMINI_MODEL_NAME
#   XCLIP_REPO
#   XCLIP_CHECKPOINT
#   XCLIP_ARCH

: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

GEMINI_MODEL_NAME="${GEMINI_MODEL_NAME:-gemini-2.5-flash}"
XCLIP_REPO="${XCLIP_REPO:-/opt/VideoX}"
XCLIP_CHECKPOINT="${XCLIP_CHECKPOINT:-/opt/VideoX/checkpoints/zero.pth}"
XCLIP_ARCH="${XCLIP_ARCH:-ViT-B/16}"

REFERENCE_VIDEO="/root/input/reference.mp4"
if [[ ! -f "$REFERENCE_VIDEO" ]]; then
  echo "Missing reference video at $REFERENCE_VIDEO" >&2
  exit 1
fi

python -m agent_env.reconstruction_spec_runner \
  --specification-json "$INPUT_SPEC" \
  --reference-video "$REFERENCE_VIDEO" \
  --runner gemini \
  --workspace "$WORKDIR" \
  --gemini-model-name "$GEMINI_MODEL_NAME" \
  --google-api-key "$GOOGLE_API_KEY" \
  --openai-api-key "$OPENAI_API_KEY" \
  --max-videos 1 \
  --eval-profile skillbench_local \
  --xclip-command "python -m agent_env.xclip_adapter --xclip-repo $XCLIP_REPO --checkpoint $XCLIP_CHECKPOINT --arch $XCLIP_ARCH --video {video} --text-file {text_file} --output {output_json}" \
  --output /root/output/benchmark_result.json

python - <<'PY'
import json
import shutil
from pathlib import Path

benchmark_path = Path('/root/output/benchmark_result.json')
payload = json.loads(benchmark_path.read_text(encoding='utf-8'))
results = payload.get('results') or []
if not results:
    raise SystemExit('No successful result found in benchmark_result.json')
result = results[0]

video_src = Path(result['video_output'])
video_dst = Path('/root/output/generated_video.mp4')
shutil.copyfile(video_src, video_dst)

metadata_path = Path(result['metadata'])
metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
summary = {
    'video_id': result.get('video_id'),
    'semantic_alignment_score': result.get('semantic_alignment_score'),
    'semantic_alignment_pass': result.get('semantic_alignment_pass'),
    'reference_fidelity_composite_score': result.get('reference_fidelity_composite_score'),
    'ocr_text_score': result.get('ocr_text_score'),
    'ocr_text_pass': result.get('ocr_text_pass'),
    'skillbench_final_score': result.get('skillbench_final_score'),
    'skillbench_final_pass': result.get('skillbench_final_pass'),
    'eval_profile': result.get('eval_profile'),
    'run_dir': result.get('run_dir'),
    'metadata_path': str(metadata_path),
    'skillbench_eval': metadata.get('skillbench_eval'),
    'semantic_alignment_eval': metadata.get('semantic_alignment_eval'),
    'ocr_text_eval': metadata.get('ocr_text_eval'),
}
Path('/root/output/eval.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
PY
