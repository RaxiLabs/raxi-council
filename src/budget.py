import json
import math
from functools import lru_cache

from src.config import (
    EVALUATION_MAX_TOKENS,
    GENERATION_MAX_TOKENS,
    GENERATION_SYSTEM_PROMPT,
    PROMPT_PATHS,
    SEMANTIC_ENTROPY_MAX_TOKENS,
    TOKEN_ESTIMATION_CHARS_PER_TOKEN,
    TOKEN_ESTIMATION_MESSAGE_OVERHEAD,
    TOKEN_ESTIMATION_SAFETY_MARGIN,
)


GENERATION_MIN_COMPLETION_ESTIMATE = 180
EVALUATION_BASE_COMPLETION_ESTIMATE = 120
EVALUATION_COMPLETION_ESTIMATE_PER_RESPONSE = 190
SEMANTIC_BASE_COMPLETION_ESTIMATE = 80
SEMANTIC_COMPLETION_ESTIMATE_PER_RESPONSE = 45


def estimate_text_tokens(text):
    text = str(text or "")
    if not text:
        return 0
    return max(1, math.ceil(len(text) / TOKEN_ESTIMATION_CHARS_PER_TOKEN))


def _with_margin(token_count):
    return math.ceil(token_count * TOKEN_ESTIMATION_SAFETY_MARGIN)


@lru_cache(maxsize=None)
def _load_prompt(persona):
    with open(PROMPT_PATHS[persona], "r", encoding="utf-8") as file:
        return file.read()


def _estimate_chat_prompt(system_prompt, user_prompt):
    return (
        estimate_text_tokens(system_prompt)
        + estimate_text_tokens(user_prompt)
        + TOKEN_ESTIMATION_MESSAGE_OVERHEAD
    )


def _build_stage_estimate(stage, calls):
    estimated_prompt_tokens = sum(call["estimated_prompt_tokens"] for call in calls)
    estimated_completion_tokens = sum(call["estimated_completion_tokens"] for call in calls)
    raw_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

    return {
        "stage": stage,
        "method": "chars_per_token_with_safety_margin",
        "chars_per_token": TOKEN_ESTIMATION_CHARS_PER_TOKEN,
        "safety_margin": TOKEN_ESTIMATION_SAFETY_MARGIN,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "estimated_total_tokens": _with_margin(raw_total_tokens),
        "calls": calls,
    }


def _estimate_call(stage, model, system_prompt, user_prompt, completion_tokens):
    prompt_tokens = _estimate_chat_prompt(system_prompt, user_prompt)
    completion_tokens = max(0, int(completion_tokens))

    return {
        "stage": stage,
        "model": model,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": _with_margin(prompt_tokens + completion_tokens),
    }


def estimate_generation_completion_tokens(user_prompt):
    prompt_tokens = estimate_text_tokens(user_prompt)
    scaled_completion = math.ceil(prompt_tokens * 1.25)
    return min(GENERATION_MAX_TOKENS, max(GENERATION_MIN_COMPLETION_ESTIMATE, scaled_completion))


def estimate_generation_stage(user_prompt, generation_models):
    completion_tokens = estimate_generation_completion_tokens(user_prompt)
    calls = [
        _estimate_call(
            stage="generation",
            model=model,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            completion_tokens=completion_tokens,
        )
        for model in generation_models
    ]
    return _build_stage_estimate("generation", calls)


def _format_anonymised_responses(anonymised_responses):
    return "\n\n".join(
        f"{item['id']}:\n{item['response']}"
        for item in anonymised_responses
    )


def _build_response_schema_hint(response_ids):
    evaluations = {
        response_id: {
            "evaluator_confidence": "HIGH/MEDIUM/LOW/UNKNOWN",
            "dimensions": {
                "factual_accuracy": {"score": 0, "reason": "brief reason"},
                "completeness": {"score": 0, "reason": "brief reason"},
                "reasoning_quality": {"score": 0, "reason": "brief reason"},
            },
            "total_score": 0,
            "hallucination_flag": False,
            "hallucination_detail": None,
            "hallucination_assessment": {
                "flag": False,
                "severity": None,
                "issues": [],
            },
            "evaluation_notes": None,
        }
        for response_id in response_ids
    }
    return json.dumps({"arbiter_id": "ARBITER-ID", "evaluations": evaluations}, indent=2)


def _build_evaluation_user_prompt(user_prompt, anonymised_responses):
    formatted_responses = _format_anonymised_responses(anonymised_responses)
    response_ids = [item["id"] for item in anonymised_responses]

    evaluation_prompt = (
        f"Original question: {user_prompt}\n\n"
        f"Evaluate each of the following anonymous responses:\n\n"
        f"{formatted_responses}"
    )

    return (
        evaluation_prompt
        + "\n\nReturn JSON in this exact top-level shape:\n"
        + _build_response_schema_hint(response_ids)
    )


def estimate_evaluation_stage(user_prompt, anonymised_responses, evaluation_personas, evaluation_models):
    response_count = len(anonymised_responses)
    completion_tokens = min(
        EVALUATION_MAX_TOKENS,
        EVALUATION_BASE_COMPLETION_ESTIMATE
        + response_count * EVALUATION_COMPLETION_ESTIMATE_PER_RESPONSE,
    )
    evaluation_user_prompt = _build_evaluation_user_prompt(user_prompt, anonymised_responses)

    calls = [
        _estimate_call(
            stage="evaluation",
            model=model,
            system_prompt=_load_prompt(persona),
            user_prompt=evaluation_user_prompt,
            completion_tokens=completion_tokens,
        )
        for persona, model in zip(evaluation_personas, evaluation_models)
    ]
    return _build_stage_estimate("evaluation", calls)


def _build_semantic_user_prompt(user_prompt, anonymised_responses):
    formatted_responses = _format_anonymised_responses(anonymised_responses)
    return (
        f"Original question: {user_prompt}\n\n"
        "Cluster the following responses by semantic equivalence.\n\n"
        + formatted_responses
    )


def estimate_semantic_entropy_stage(user_prompt, anonymised_responses, semantic_model):
    response_count = len(anonymised_responses)
    completion_tokens = min(
        SEMANTIC_ENTROPY_MAX_TOKENS,
        SEMANTIC_BASE_COMPLETION_ESTIMATE
        + response_count * SEMANTIC_COMPLETION_ESTIMATE_PER_RESPONSE,
    )
    calls = [
        _estimate_call(
            stage="semantic_entropy",
            model=semantic_model,
            system_prompt=_load_prompt("semantic_entropy"),
            user_prompt=_build_semantic_user_prompt(user_prompt, anonymised_responses),
            completion_tokens=completion_tokens,
        )
    ]
    return _build_stage_estimate("semantic_entropy", calls)


def estimate_attempt_budget(user_prompt, generation_models, evaluation_personas, evaluation_models, semantic_model):
    estimated_response_tokens = estimate_generation_completion_tokens(user_prompt)
    placeholder_response = "x" * (estimated_response_tokens * TOKEN_ESTIMATION_CHARS_PER_TOKEN)
    anonymised_responses = [
        {
            "id": f"Response_{chr(65 + index)}",
            "response": placeholder_response,
        }
        for index in range(len(generation_models))
    ]

    stages = [
        estimate_generation_stage(user_prompt, generation_models),
        estimate_evaluation_stage(user_prompt, anonymised_responses, evaluation_personas, evaluation_models),
        estimate_semantic_entropy_stage(user_prompt, anonymised_responses, semantic_model),
    ]

    return {
        "stage": "attempt",
        "method": "preflight_estimate",
        "estimated_total_tokens": sum(stage["estimated_total_tokens"] for stage in stages),
        "stages": stages,
    }