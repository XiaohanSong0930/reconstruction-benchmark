from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


TaskFamily = Literal["task1_open_generation_plan", "task2_reference_reconstruction_plan"]


@dataclass
class TaskExample:
    example_id: str
    task_family: TaskFamily
    title: str
    prompt: str
    inputs: Dict[str, Any]
    scoring_targets: Dict[str, Any]
    notes: Optional[str] = None


TASK1_SYSTEM_PROMPT = """You are a video advertising planner.
Return only valid JSON.
Do not use markdown fences.
Do not add explanation before or after the JSON object.
Your job is to create a concise multi-step plan for an agentic video-generation system.
Keep the response short.
Use exactly 2 scenes.
Use at most 6 planned_actions.
Do not repeat the same action more than once unless absolutely necessary.
Each style_keywords list should contain at most 4 short strings.
Do not write long sentences inside style_keywords.

The JSON schema must be:
{
  "summary": "short string",
  "scenes": [
    {
      "scene_id": "scene_001",
      "goal": "string",
      "visual_focus": "string",
      "camera_motion": "string",
      "style_keywords": ["string", "..."]
    }
  ],
  "planned_actions": ["plan_scenes", "gen_scene_image", "..."],
  "transition_strategy": "string",
  "verification_plan": ["judge_scene", "judge_final"]
}
"""


TASK2_SYSTEM_PROMPT = """You are a reference-reconstruction planner for a video agent.
Return only valid JSON.
Do not use markdown fences.
Do not add explanation before or after the JSON object.
Your job is to create a concise action plan for matching a target reference style or edit property.

The JSON schema must be:
{
  "summary": "short string",
  "reference_understanding": "string",
  "planned_actions": ["plan_scenes", "gen_scene_video", "..."],
  "edit_decisions": {
    "camera_motion": "string",
    "transition_strategy": "string",
    "cut_policy": "string"
  },
  "verification_plan": ["judge_scene", "judge_final"]
}
"""


def build_task1_examples() -> List[TaskExample]:
    return [
        TaskExample(
            example_id="task1_ring_ad_001",
            task_family="task1_open_generation_plan",
            title="Luxury Ring Ad Planning",
            prompt=(
                "Create a short two-scene luxury ring ad. "
                "Scene 1 should be a polished hero product shot. "
                "Scene 2 should transition into a refined lifestyle payoff shot. "
                "The result should feel premium, elegant, and persuasive."
            ),
            inputs={
                "product_image": "product/ring.jpg",
                "target_audience": "Affluent young professionals who value subtle luxury and craftsmanship.",
                "num_scenes": 2,
                "target_edit_properties": {
                    "needs_smooth_transition": True,
                    "camera_motion": "slow_zoom_in",
                    "visual_style": "clean_luxury",
                },
            },
            scoring_targets={
                "required_num_scenes": 2,
                "required_actions": [
                    "plan_scenes",
                    "gen_scene_image",
                    "gen_scene_video",
                    "judge_final",
                    "finish",
                ],
                "preferred_actions": ["transition_edit", "judge_scene"],
                "preferred_scene_keywords": {
                    "scene_001": ["hero", "product", "premium", "close-up"],
                    "scene_002": ["lifestyle", "payoff", "brand", "elegant"],
                },
                "preferred_transition_keywords": ["smooth", "refined", "fade", "continuity"],
            },
            notes="This is the canonical open-generation advertising example.",
        ),
        TaskExample(
            example_id="task1_watch_ad_001",
            task_family="task1_open_generation_plan",
            title="Performance Watch Ad Planning",
            prompt=(
                "Create a short two-scene performance watch ad. "
                "Scene 1 should establish the watch as precise and durable. "
                "Scene 2 should connect the product to an active lifestyle while keeping the brand presentation sharp."
            ),
            inputs={
                "product_image": "product/watch.jpg",
                "target_audience": "Busy professionals and athletes who care about durability, status, and practical utility.",
                "num_scenes": 2,
                "target_edit_properties": {
                    "needs_smooth_transition": True,
                    "camera_motion": "dynamic_cut_then_zoom",
                    "visual_style": "performance_minimal",
                },
            },
            scoring_targets={
                "required_num_scenes": 2,
                "required_actions": [
                    "plan_scenes",
                    "gen_scene_image",
                    "gen_scene_video",
                    "judge_final",
                    "finish",
                ],
                "preferred_actions": ["transition_edit", "judge_scene", "trim_clip"],
                "preferred_scene_keywords": {
                    "scene_001": ["precision", "watch", "detail", "durable"],
                    "scene_002": ["active", "lifestyle", "motion", "performance"],
                },
                "preferred_transition_keywords": ["dynamic", "smooth", "pace", "continuity"],
            },
            notes="This example stresses different style and pacing goals from the ring-ad task.",
        ),
    ]


def build_task2_examples() -> List[TaskExample]:
    return [
        TaskExample(
            example_id="task2_zoom_reconstruct_001",
            task_family="task2_reference_reconstruction_plan",
            title="Reconstruct Zoom-In Product Shot",
            prompt=(
                "Reconstruct a short premium product clip with a smooth push-in camera feeling. "
                "Preserve the centered subject and avoid abrupt motion."
            ),
            inputs={
                "reference_video": "agent_env_runs/scene_001/video_zoom_in.mp4",
                "target_edit_properties": {
                    "match_reference_motion": True,
                    "camera_motion": "zoom_in",
                    "preferred_transition": "none",
                },
                "num_scenes": 1,
            },
            scoring_targets={
                "required_actions": [
                    "plan_scenes",
                    "gen_scene_video",
                    "judge_final",
                    "finish",
                ],
                "preferred_actions": ["zoom_in_clip", "judge_scene"],
                "required_edit_decisions": {
                    "camera_motion": "zoom_in",
                    "transition_strategy": "none",
                },
                "reference_keywords": ["push-in", "zoom", "centered", "smooth"],
            },
            notes="This is the simplest reconstruction task and aligns with the existing zoom demo.",
        ),
        TaskExample(
            example_id="task2_transition_reconstruct_001",
            task_family="task2_reference_reconstruction_plan",
            title="Reconstruct Smooth Two-Scene Transition",
            prompt=(
                "Match a two-scene transition style where scene 1 should hand off smoothly into scene 2. "
                "The edit should preserve product focus while choosing a cut point that makes the transition feel intentional."
            ),
            inputs={
                "reference_video_a": "agent_env_runs/scene_001/video.mp4",
                "reference_video_b": "agent_env_runs/scene_002/video.mp4",
                "target_edit_properties": {
                    "preferred_transition": "smooth",
                    "cut_policy": "cross_scene_content_aware",
                    "camera_motion": "stable_to_payoff",
                },
                "num_scenes": 2,
            },
            scoring_targets={
                "required_actions": [
                    "plan_scenes",
                    "gen_scene_video",
                    "judge_final",
                    "finish",
                ],
                "preferred_actions": ["cut_for_transition_pair", "transition_edit", "stitch_preview"],
                "required_edit_decisions": {
                    "transition_strategy": "smooth",
                    "cut_policy": "content_aware",
                },
                "reference_keywords": ["transition", "handoff", "cut point", "continuity"],
            },
            notes="This example is designed for the new transition-cut tooling rather than single-shot zoom editing.",
        ),
    ]


def build_all_examples() -> List[TaskExample]:
    return build_task1_examples() + build_task2_examples()


def get_examples(task_family: Optional[TaskFamily] = None) -> List[TaskExample]:
    examples = build_all_examples()
    if task_family is None:
        return examples
    return [item for item in examples if item.task_family == task_family]
