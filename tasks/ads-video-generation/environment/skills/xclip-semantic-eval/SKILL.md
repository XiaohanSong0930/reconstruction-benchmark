---
name: xclip-semantic-eval
description: Use this skill when a task needs deterministic local semantic evaluation between advertisement text or specification content and a generated video, using X-CLIP or a similar text-video alignment model.
---

# X-CLIP Semantic Eval

Use this skill when semantic alignment must be measured locally without an LLM judge.

## Workflow

1. Convert the specification into a short text description that fits the text-video model context window.
2. Run X-CLIP on the generated video and the compressed specification text.
3. Record a bounded semantic alignment score and a deterministic pass/fail flag.

## Requirements

- Keep the text prompt short enough for the tokenizer limit.
- Use a fixed checkpoint and architecture for reproducibility.
- Output a machine-readable JSON result.

## Outputs

- Semantic alignment score in `[0, 1]`.
- Deterministic semantic pass/fail decision.
