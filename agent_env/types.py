from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentAction:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneState:
    scene_id: str
    description: str
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    transition_cut_sec: Optional[float] = None
    transition_lead_clip_path: Optional[str] = None
    transition_tail_clip_path: Optional[str] = None
    transition_pair_score: Optional[float] = None
    transition_to_next: Optional[str] = None
    trim_start_sec: float = 0.0
    trim_end_sec: float = 0.0
    last_preview_path: Optional[str] = None
    scene_score: Optional[float] = None
    qualitative_feedback: Optional[str] = None
    retries_image: int = 0
    retries_video: int = 0
    edit_history: List[str] = field(default_factory=list)


@dataclass
class EpisodeState:
    task_id: str
    prompt: str
    transcript_file: Optional[str] = None
    long_video_path: Optional[str] = None
    object_image: Optional[str] = None
    target_audience: Optional[str] = None
    done: bool = False
    step_count: int = 0
    scenes: Dict[str, SceneState] = field(default_factory=dict)
    scene_order: List[str] = field(default_factory=list)
    final_score: Optional[float] = None
    preference_score: Optional[float] = None
    final_feedback: Optional[str] = None
    stitched_preview_path: Optional[str] = None
    last_error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
