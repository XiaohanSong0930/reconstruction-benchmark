---
name: video-fidelity-eval
description: Use this skill when a task needs deterministic local comparison between a generated video and a reference video, including visual similarity, temporal similarity, OCR text fidelity, and final rubric-based scoring.
---

# Video Fidelity Eval

Use this skill when a generated video must be compared against a reference clip using local tools only.

## Workflow

1. Compare the generated video and reference video with fixed visual and temporal metrics.
2. Compute a composite fidelity score from deterministic local metrics.
3. Run OCR as a soft auxiliary signal for visible text, brand, or slogan evidence.
4. Combine semantic, fidelity, and OCR scores with a fixed rubric.

## Required metrics

- Frame histogram similarity
- Temporal delta similarity
- SSIM
- PSNR

## Requirements

- Keep every score bounded in `[0, 1]`.
- Use deterministic thresholds.
- Do not rely on external judge APIs.

## Outputs

- Reference fidelity score.
- OCR text fidelity score.
- Final deterministic pass/fail result.
