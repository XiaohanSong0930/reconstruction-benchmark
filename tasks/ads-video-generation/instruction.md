Create an ads video that matches the provided structured specification.

Inputs are available in `/root/input/`:
- `specification.json`: the structured advertisement specification
- `reference.mp4`: a reference clip for visual grounding and fidelity-based evaluation

Generate a single MP4 advertisement at `/root/output/generated_video.mp4`.

Also write an evaluation summary JSON to `/root/output/eval.json`. The summary must report semantic alignment, reference fidelity, OCR text fidelity, and the final deterministic pass/fail result.

The generated advertisement should faithfully realize the product, message, tone, and scene progression described in the specification, while staying visually consistent with the reference clip.
