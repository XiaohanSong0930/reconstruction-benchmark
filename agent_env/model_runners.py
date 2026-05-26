from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


class RunnerError(RuntimeError):
    pass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0


class BaseModelRunner:
    def __init__(self) -> None:
        self._last_generation_meta: Dict[str, Any] = {}

    def generate(self, system_prompt: str, user_prompt: str, config: Optional[GenerationConfig] = None) -> str:
        raise NotImplementedError

    def get_last_generation_meta(self) -> Dict[str, Any]:
        return dict(self._last_generation_meta)


class MockPlannerRunner(BaseModelRunner):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, system_prompt: str, user_prompt: str, config: Optional[GenerationConfig] = None) -> str:
        self._last_generation_meta = {
            "provider": "mock",
            "model": "mock-planner",
            "usage": None,
        }
        if "cut point" in user_prompt or "handoff" in user_prompt or "transition" in user_prompt:
            return json.dumps(
                {
                    "summary": "Plan a content-aware handoff between two scenes.",
                    "reference_understanding": "The target is a smooth scene-to-scene transition with intentional continuity.",
                    "planned_actions": ["plan_scenes", "gen_scene_video", "cut_for_transition_pair", "transition_edit", "stitch_preview", "judge_final", "finish"],
                    "edit_decisions": {
                        "camera_motion": "stable_to_payoff",
                        "transition_strategy": "smooth",
                        "cut_policy": "content_aware",
                    },
                    "verification_plan": ["judge_final"],
                },
                ensure_ascii=False,
            )
        if "push-in" in user_prompt or "zoom" in user_prompt:
            return json.dumps(
                {
                    "summary": "Reconstruct the reference with a smooth push-in motion.",
                    "reference_understanding": "The target is a centered premium product shot with smooth zoom-in motion.",
                    "planned_actions": ["plan_scenes", "gen_scene_video", "zoom_in_clip", "judge_scene", "judge_final", "finish"],
                    "edit_decisions": {
                        "camera_motion": "zoom_in",
                        "transition_strategy": "none",
                        "cut_policy": "not_needed",
                    },
                    "verification_plan": ["judge_scene", "judge_final"],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "summary": "Create a two-scene premium ad plan.",
                "scenes": [
                    {
                        "scene_id": "scene_001",
                        "goal": "Hero product reveal with premium close-up focus.",
                        "visual_focus": "product centerpiece",
                        "camera_motion": "slow_zoom_in",
                        "style_keywords": ["premium", "hero", "clean", "elegant"],
                    },
                    {
                        "scene_id": "scene_002",
                        "goal": "Lifestyle payoff shot with refined brand finish.",
                        "visual_focus": "product in polished context",
                        "camera_motion": "smooth_follow",
                        "style_keywords": ["lifestyle", "payoff", "brand", "continuity"],
                    },
                ],
                "planned_actions": ["plan_scenes", "gen_scene_image", "gen_scene_video", "transition_edit", "judge_scene", "judge_final", "finish"],
                "transition_strategy": "smooth luxury fade",
                "verification_plan": ["judge_scene", "judge_final"],
            },
            ensure_ascii=False,
        )


class QwenLocalRunner(BaseModelRunner):
    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RunnerError("transformers is not installed; cannot run local Qwen.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map=self.device,
        )

    def generate(self, system_prompt: str, user_prompt: str, config: Optional[GenerationConfig] = None) -> str:
        self._lazy_load()
        cfg = config or GenerationConfig()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        model_inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, "device"):
            model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
        generate_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.temperature > 0,
        }
        if cfg.temperature > 0:
            generate_kwargs["temperature"] = cfg.temperature
        outputs = model.generate(**generate_kwargs)
        prompt_len = model_inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        output_tokens = int(generated.shape[-1]) if hasattr(generated, "shape") else None
        self._last_generation_meta = {
            "provider": "local",
            "model": self.model_path,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + (output_tokens or 0),
            },
        }
        return output_text


class GeminiRunner(BaseModelRunner):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key_env: str = "GEMINI_API_KEY"):
        super().__init__()
        self.model_name = model_name
        self.api_key_env = api_key_env
        self._client = None

    def _lazy_load(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RunnerError(f"{self.api_key_env} is not set; cannot call Gemini API.")
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RunnerError("google-genai is not installed; cannot call Gemini API.") from exc

        self._client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, config: Optional[GenerationConfig] = None) -> str:
        self._lazy_load()
        cfg = config or GenerationConfig()
        client = self._client
        assert client is not None
        combined_prompt = f"{system_prompt}\n\nUser task:\n{user_prompt}"
        response = client.models.generate_content(
            model=self.model_name,
            contents=combined_prompt,
            config={
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_new_tokens,
                "thinking_config": {
                    "thinking_budget": 0,
                },
            },
        )
        text = getattr(response, "text", None)
        if not text:
            raise RunnerError("Gemini returned an empty response.")
        usage_obj = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_meta: Dict[str, Any] | None = None
        if usage_obj is not None:
            usage_meta = {}
            for key in [
                "prompt_token_count",
                "candidates_token_count",
                "total_token_count",
                "thoughts_token_count",
                "cached_content_token_count",
            ]:
                if hasattr(usage_obj, key):
                    usage_meta[key] = getattr(usage_obj, key)
        self._last_generation_meta = {
            "provider": "google",
            "model": self.model_name,
            "usage": usage_meta,
        }
        return text.strip()


def build_runner(
    runner_name: str,
    *,
    qwen_model_path: Optional[str] = None,
    gemini_model_name: str = "gemini-2.5-flash",
) -> BaseModelRunner:
    normalized = runner_name.strip().lower()
    if normalized == "mock":
        return MockPlannerRunner()
    if normalized == "qwen":
        if not qwen_model_path:
            raise RunnerError("--qwen-model-path is required for runner=qwen")
        return QwenLocalRunner(model_path=qwen_model_path)
    if normalized == "gemini":
        return GeminiRunner(model_name=gemini_model_name)
    raise RunnerError(f"Unknown runner: {runner_name}")
