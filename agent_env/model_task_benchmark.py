from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .model_runners import GenerationConfig, RunnerError, build_runner
from .model_task_examples import (
    TASK1_SYSTEM_PROMPT,
    TASK2_SYSTEM_PROMPT,
    TaskExample,
    get_examples,
)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def _repair_json_text(text: str) -> str:
    repaired = text.strip()
    # Remove trailing commas before closing braces/brackets.
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    # Insert a missing comma between a closing brace/bracket and the next quoted key.
    repaired = re.sub(r'([}\]])(\s*)"([A-Za-z0-9_]+)"\s*:', r'\1,\2"\3":', repaired)
    # Insert a missing comma between a JSON string value and the next quoted key.
    repaired = re.sub(r'(")\s*\n(\s*)"([A-Za-z0-9_]+)"\s*:', r'\1,\n\2"\3":', repaired)
    return repaired


def _extract_json_block(text: str) -> Dict[str, Any]:
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_json_text(candidate)
        return json.loads(repaired)


def _lower_join(values: List[str]) -> str:
    return " ".join(values).lower()


def _score_task1(example: TaskExample, parsed: Dict[str, Any]) -> Dict[str, Any]:
    score = 0.0
    breakdown: Dict[str, float] = {}

    scenes = parsed.get("scenes") or []
    planned_actions = [str(item) for item in parsed.get("planned_actions") or []]
    verification_plan = [str(item) for item in parsed.get("verification_plan") or []]
    transition_strategy = str(parsed.get("transition_strategy") or "")

    breakdown["format_valid"] = 1.0
    score += 1.0

    required_num_scenes = int(example.scoring_targets["required_num_scenes"])
    scene_count_score = 1.0 if len(scenes) == required_num_scenes else 0.0
    breakdown["scene_count_match"] = scene_count_score
    score += scene_count_score

    required_actions = set(example.scoring_targets["required_actions"])
    action_coverage = len(required_actions.intersection(planned_actions)) / max(1, len(required_actions))
    breakdown["required_action_coverage"] = round(action_coverage, 3)
    score += 2.0 * action_coverage

    preferred_actions = set(example.scoring_targets["preferred_actions"])
    preferred_action_coverage = len(preferred_actions.intersection(planned_actions + verification_plan)) / max(1, len(preferred_actions))
    breakdown["preferred_action_coverage"] = round(preferred_action_coverage, 3)
    score += 1.0 * preferred_action_coverage

    preferred_scene_keywords = example.scoring_targets["preferred_scene_keywords"]
    scene_keyword_hits = 0
    scene_keyword_total = 0
    for scene in scenes:
        scene_id = str(scene.get("scene_id") or "")
        goal_blob = _lower_join(
            [
                str(scene.get("goal") or ""),
                str(scene.get("visual_focus") or ""),
                str(scene.get("camera_motion") or ""),
                _lower_join([str(item) for item in scene.get("style_keywords") or []]),
            ]
        )
        expected_keywords = preferred_scene_keywords.get(scene_id, [])
        scene_keyword_total += len(expected_keywords)
        for keyword in expected_keywords:
            if keyword.lower() in goal_blob:
                scene_keyword_hits += 1
    scene_keyword_score = scene_keyword_hits / max(1, scene_keyword_total)
    breakdown["scene_keyword_alignment"] = round(scene_keyword_score, 3)
    score += 2.0 * scene_keyword_score

    transition_blob = f"{transition_strategy} {' '.join(planned_actions)}".lower()
    transition_keywords = example.scoring_targets["preferred_transition_keywords"]
    transition_hits = sum(1 for keyword in transition_keywords if keyword.lower() in transition_blob)
    transition_score = transition_hits / max(1, len(transition_keywords))
    breakdown["transition_awareness"] = round(transition_score, 3)
    score += 1.0 * transition_score

    return {
        "score": round(score, 3),
        "score_max": 8.0,
        "normalized_score": round(score / 8.0, 3),
        "breakdown": breakdown,
    }


def _score_task2(example: TaskExample, parsed: Dict[str, Any]) -> Dict[str, Any]:
    score = 0.0
    breakdown: Dict[str, float] = {}

    planned_actions = [str(item) for item in parsed.get("planned_actions") or []]
    verification_plan = [str(item) for item in parsed.get("verification_plan") or []]
    edit_decisions = parsed.get("edit_decisions") or {}
    reference_understanding = str(parsed.get("reference_understanding") or "")

    breakdown["format_valid"] = 1.0
    score += 1.0

    required_actions = set(example.scoring_targets["required_actions"])
    action_coverage = len(required_actions.intersection(planned_actions + verification_plan)) / max(1, len(required_actions))
    breakdown["required_action_coverage"] = round(action_coverage, 3)
    score += 2.0 * action_coverage

    preferred_actions = set(example.scoring_targets["preferred_actions"])
    preferred_action_coverage = len(preferred_actions.intersection(planned_actions + verification_plan)) / max(1, len(preferred_actions))
    breakdown["preferred_action_coverage"] = round(preferred_action_coverage, 3)
    score += 1.5 * preferred_action_coverage

    required_edit_decisions = example.scoring_targets["required_edit_decisions"]
    edit_matches = 0
    for key, target_value in required_edit_decisions.items():
        predicted = str(edit_decisions.get(key) or "").lower()
        if str(target_value).lower() in predicted:
            edit_matches += 1
    edit_score = edit_matches / max(1, len(required_edit_decisions))
    breakdown["edit_decision_match"] = round(edit_score, 3)
    score += 2.0 * edit_score

    ref_keywords = example.scoring_targets["reference_keywords"]
    ref_blob = f"{reference_understanding} {json.dumps(edit_decisions, ensure_ascii=False)}".lower()
    ref_hits = sum(1 for keyword in ref_keywords if keyword.lower() in ref_blob)
    ref_score = ref_hits / max(1, len(ref_keywords))
    breakdown["reference_understanding"] = round(ref_score, 3)
    score += 1.5 * ref_score

    return {
        "score": round(score, 3),
        "score_max": 8.0,
        "normalized_score": round(score / 8.0, 3),
        "breakdown": breakdown,
    }


def _build_user_prompt(example: TaskExample) -> str:
    return json.dumps(
        {
            "example_id": example.example_id,
            "title": example.title,
            "prompt": example.prompt,
            "inputs": example.inputs,
            "notes": example.notes,
        },
        ensure_ascii=False,
        indent=2,
    )


def _evaluate_example(example: TaskExample, parsed: Dict[str, Any]) -> Dict[str, Any]:
    if example.task_family == "task1_open_generation_plan":
        return _score_task1(example, parsed)
    if example.task_family == "task2_reference_reconstruction_plan":
        return _score_task2(example, parsed)
    raise ValueError(f"Unsupported task_family: {example.task_family}")


def _select_system_prompt(example: TaskExample) -> str:
    if example.task_family == "task1_open_generation_plan":
        return TASK1_SYSTEM_PROMPT
    if example.task_family == "task2_reference_reconstruction_plan":
        return TASK2_SYSTEM_PROMPT
    raise ValueError(f"Unsupported task_family: {example.task_family}")


def run_benchmark(
    *,
    runner_name: str,
    task_family: Optional[str],
    qwen_model_path: Optional[str],
    gemini_model_name: str,
    output_path: str,
    example_id: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    examples = get_examples(task_family=task_family) if task_family else get_examples()
    if example_id:
        examples = [item for item in examples if item.example_id == example_id]
    if not examples:
        raise ValueError("No matching examples found.")

    runner = build_runner(
        runner_name,
        qwen_model_path=qwen_model_path,
        gemini_model_name=gemini_model_name,
    )

    records: List[Dict[str, Any]] = []
    totals: List[float] = []
    parse_failures: List[Dict[str, Any]] = []
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    for example in examples:
        system_prompt = _select_system_prompt(example)
        user_prompt = _build_user_prompt(example)
        raw_response = runner.generate(system_prompt, user_prompt, generation_config)
        try:
            parsed = _extract_json_block(raw_response)
        except Exception as exc:
            parse_failures.append(
                {
                    "example": asdict(example),
                    "error": str(exc),
                    "raw_response": raw_response,
                }
            )
            continue
        scoring = _evaluate_example(example, parsed)
        totals.append(float(scoring["normalized_score"]))
        records.append(
            {
                "example": asdict(example),
                "raw_response": raw_response,
                "parsed_response": parsed,
                "scoring": scoring,
            }
        )

    payload = {
        "runner": runner_name,
        "qwen_model_path": qwen_model_path,
        "gemini_model_name": gemini_model_name,
        "task_family": task_family or "all",
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "num_examples": len(records),
        "num_parse_failures": len(parse_failures),
        "average_normalized_score": round(sum(totals) / max(1, len(totals)), 3) if totals else None,
        "records": records,
        "parse_failures": parse_failures,
    }

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)
    with open(abs_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen/Gemini planning benchmarks for task1 and task2.")
    parser.add_argument("--runner", default="mock", choices=["mock", "qwen", "gemini"], help="Model runner backend")
    parser.add_argument(
        "--task-family",
        default="all",
        choices=["all", "task1_open_generation_plan", "task2_reference_reconstruction_plan"],
        help="Which task family to run",
    )
    parser.add_argument("--example-id", default=None, help="Optional single example id to run")
    parser.add_argument("--qwen-model-path", default=None, help="Local model path for runner=qwen")
    parser.add_argument("--gemini-model-name", default="gemini-2.5-flash", help="Gemini model name for runner=gemini")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation token budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument(
        "--output",
        default="agent_env_runs/benchmarks/model_task_benchmark.json",
        help="Output JSON file for benchmark results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_benchmark(
            runner_name=args.runner,
            task_family=None if args.task_family == "all" else args.task_family,
            qwen_model_path=args.qwen_model_path,
            gemini_model_name=args.gemini_model_name,
            output_path=args.output,
            example_id=args.example_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except RunnerError as exc:
        raise SystemExit(f"Runner error: {exc}") from exc

    print(json.dumps(
        {
            "runner": payload["runner"],
            "task_family": payload["task_family"],
            "num_examples": payload["num_examples"],
            "average_normalized_score": payload["average_normalized_score"],
            "output": os.path.abspath(args.output),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
