import math

from src.agents import (
    EVALUATION_CACHE_NAMESPACE,
    GENERATION_CACHE_NAMESPACE,
    anonymise_and_shuffle,
    build_evaluation_cache_request,
    build_generation_cache_request,
)
from src.cache import get_cached_result, has_cached_result
from src.config import (
    EVALUATION_MAX_TOKENS,
    GENERATION_MAX_TOKENS,
    GENERATION_SYSTEM_PROMPT,
    MODEL_PRICING,
    SEMANTIC_ENTROPY_MAX_TOKENS,
    TOKEN_ESTIMATION_CHARS_PER_TOKEN,
    TOKEN_ESTIMATION_MESSAGE_OVERHEAD,
    TOKEN_ESTIMATION_SAFETY_MARGIN,
)
from src.semantic_entropy import (
    SEMANTIC_ENTROPY_CACHE_NAMESPACE,
    build_semantic_entropy_cache_request,
)


GENERATION_MIN_COMPLETION_TOKENS = 180
EVALUATION_BASE_COMPLETION_TOKENS = 140
EVALUATION_COMPLETION_TOKENS_PER_RESPONSE = 220
SEMANTIC_BASE_COMPLETION_TOKENS = 80
SEMANTIC_COMPLETION_TOKENS_PER_RESPONSE = 45


def estimate_text_tokens(text):
    text = str(text or "")
    if not text:
        return 0
    return max(1, math.ceil(len(text) / TOKEN_ESTIMATION_CHARS_PER_TOKEN))


def _estimate_call_cost(model, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000


def _estimate_chat_prompt_tokens(system_prompt, user_prompt):
    return (
        estimate_text_tokens(system_prompt)
        + estimate_text_tokens(user_prompt)
        + TOKEN_ESTIMATION_MESSAGE_OVERHEAD
    )


def _with_margin(value):
    return math.ceil(value * TOKEN_ESTIMATION_SAFETY_MARGIN)


def _estimate_call(
    stage,
    model,
    system_prompt,
    user_prompt,
    expected_completion_tokens,
    max_completion_tokens,
    cache_namespace=None,
    cache_request=None,
):
    cache_hit = bool(
        cache_namespace
        and cache_request
        and has_cached_result(cache_namespace, cache_request)
    )

    if cache_hit:
        return {
            "stage": stage,
            "model": model,
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
            "estimated_total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "cache_hit": True,
        }

    prompt_tokens = _estimate_chat_prompt_tokens(system_prompt, user_prompt)
    completion_tokens = min(max_completion_tokens, max(0, int(expected_completion_tokens)))
    total_tokens = prompt_tokens + completion_tokens

    return {
        "stage": stage,
        "model": model,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": _with_margin(total_tokens),
        "estimated_cost_usd": round(_estimate_call_cost(model, prompt_tokens, completion_tokens), 6),
        "cache_hit": False,
    }


def _build_stage_estimate(stage, calls):
    prompt_tokens = sum(call["estimated_prompt_tokens"] for call in calls)
    completion_tokens = sum(call["estimated_completion_tokens"] for call in calls)
    total_tokens = sum(call["estimated_total_tokens"] for call in calls)
    estimated_cost = sum(call["estimated_cost_usd"] for call in calls)

    return {
        "stage": stage,
        "method": "chars_per_token_with_stage_completion_heuristics",
        "chars_per_token": TOKEN_ESTIMATION_CHARS_PER_TOKEN,
        "safety_margin": TOKEN_ESTIMATION_SAFETY_MARGIN,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "cache_hits": sum(1 for call in calls if call.get("cache_hit")),
        "calls": calls,
    }


def _estimate_generation_completion(user_prompt):
    prompt_tokens = estimate_text_tokens(user_prompt)
    return min(
        GENERATION_MAX_TOKENS,
        max(GENERATION_MIN_COMPLETION_TOKENS, math.ceil(prompt_tokens * 1.25)),
    )


def estimate_generation_stage(user_prompt, generation_models):
    completion_tokens = _estimate_generation_completion(user_prompt)
    calls = [
        _estimate_call(
            stage="generation",
            model=model,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            expected_completion_tokens=completion_tokens,
            max_completion_tokens=GENERATION_MAX_TOKENS,
            cache_namespace=GENERATION_CACHE_NAMESPACE,
            cache_request=build_generation_cache_request(model, user_prompt, GENERATION_MAX_TOKENS),
        )
        for model in generation_models
    ]
    return _build_stage_estimate("generation", calls)


def estimate_evaluation_stage(user_prompt, anonymised_responses, evaluation_personas, evaluation_models):
    response_count = len(anonymised_responses)
    response_ids = [item["id"] for item in anonymised_responses]
    evaluation_prompt = build_evaluation_prompt(user_prompt, anonymised_responses)
    completion_tokens = min(
        EVALUATION_MAX_TOKENS,
        EVALUATION_BASE_COMPLETION_TOKENS
        + response_count * EVALUATION_COMPLETION_TOKENS_PER_RESPONSE,
    )
    calls = []
    for persona, model in zip(evaluation_personas, evaluation_models):
        cache_request = build_evaluation_cache_request(
            user_prompt,
            anonymised_responses,
            persona,
            model,
            EVALUATION_MAX_TOKENS,
            evaluation_prompt=evaluation_prompt,
            response_ids=response_ids,
        )
        calls.append(
            _estimate_call(
                stage="evaluation",
                model=model,
                system_prompt=cache_request["system_prompt"],
                user_prompt=cache_request["user_prompt"],
                expected_completion_tokens=completion_tokens,
                max_completion_tokens=EVALUATION_MAX_TOKENS,
                cache_namespace=EVALUATION_CACHE_NAMESPACE,
                cache_request=cache_request,
            )
        )
    return _build_stage_estimate("evaluation", calls)


def estimate_semantic_entropy_stage(user_prompt, anonymised_responses, semantic_model):
    response_count = len(anonymised_responses)
    completion_tokens = min(
        SEMANTIC_ENTROPY_MAX_TOKENS,
        SEMANTIC_BASE_COMPLETION_TOKENS
        + response_count * SEMANTIC_COMPLETION_TOKENS_PER_RESPONSE,
    )
    cache_request = build_semantic_entropy_cache_request(user_prompt, anonymised_responses)
    calls = [
        _estimate_call(
            stage="semantic_entropy",
            model=semantic_model,
            system_prompt=cache_request["system_prompt"],
            user_prompt=cache_request["user_prompt"],
            expected_completion_tokens=completion_tokens,
            max_completion_tokens=SEMANTIC_ENTROPY_MAX_TOKENS,
            cache_namespace=SEMANTIC_ENTROPY_CACHE_NAMESPACE,
            cache_request=cache_request,
        )
    ]
    return _build_stage_estimate("semantic_entropy", calls)


def estimate_attempt_budget(
    user_prompt,
    generation_models,
    evaluation_personas,
    evaluation_models,
    semantic_model,
):
    cached_generation_responses = []
    for model in generation_models:
        cached_response, _ = get_cached_result(
            GENERATION_CACHE_NAMESPACE,
            build_generation_cache_request(model, user_prompt, GENERATION_MAX_TOKENS),
        )
        if cached_response is None:
            cached_generation_responses = []
            break
        cached_generation_responses.append({
            "model": model,
            "response": cached_response,
        })

    if cached_generation_responses:
        downstream_responses = anonymise_and_shuffle(cached_generation_responses, user_prompt)
    else:
        estimated_response_tokens = _estimate_generation_completion(user_prompt)
        placeholder_response = "x" * (estimated_response_tokens * TOKEN_ESTIMATION_CHARS_PER_TOKEN)
        downstream_responses = [
            {
                "id": f"Response_{chr(65 + index)}",
                "response": placeholder_response,
            }
            for index in range(len(generation_models))
        ]

    stages = [
        estimate_generation_stage(user_prompt, generation_models),
        estimate_evaluation_stage(user_prompt, downstream_responses, evaluation_personas, evaluation_models),
        estimate_semantic_entropy_stage(user_prompt, downstream_responses, semantic_model),
    ]

    return {
        "stage": "attempt",
        "method": "preflight_stage_sum",
        "chars_per_token": TOKEN_ESTIMATION_CHARS_PER_TOKEN,
        "safety_margin": TOKEN_ESTIMATION_SAFETY_MARGIN,
        "estimated_prompt_tokens": sum(stage["estimated_prompt_tokens"] for stage in stages),
        "estimated_completion_tokens": sum(stage["estimated_completion_tokens"] for stage in stages),
        "estimated_total_tokens": sum(stage["estimated_total_tokens"] for stage in stages),
        "estimated_cost_usd": round(sum(stage["estimated_cost_usd"] for stage in stages), 6),
        "stages": stages,
    }
