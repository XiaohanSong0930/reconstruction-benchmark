---
name: ad-spec-to-video
description: Use this skill when a task requires turning a structured advertisement specification into a short generated ad video, especially when the workflow includes script generation, visual anchoring from a reference clip, and a fixed output video path.
---

# Ad Spec To Video

Use this skill when the input is a structured ad specification and the required outcome is a short advertisement video.

## Workflow

1. Read the structured specification and identify the product, message, tone, opening beat, and scene progression.
2. Use the reference clip only for visual grounding and anchor selection when the task provides one.
3. Generate a short reconstruction script that preserves the scene order and advertising intent.
4. Produce a single output MP4 at the required path.

## Requirements

- Preserve the product or brand focus.
- Preserve the intended tone and scene order.
- Do not turn the ad into a sequel or continuation.
- Keep the video short and coherent.

## Outputs

- A generated MP4 advertisement.
- Any intermediate script or metadata files required by the task runtime.
