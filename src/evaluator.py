import time
from colorama import Fore, init
from src.agents import generate_responses, anonymise_and_shuffle, evaluate_responses
from src.aggregator import calculate_final_score, select_best_response
from src.budget import (
    estimate_attempt_budget,
    estimate_evaluation_stage,
    estimate_generation_stage,
    estimate_semantic_entropy_stage,
)
from src.config import (
    MIN_VALID_ARBITERS,
    GENERATION_MODELS,
    EVALUATION_MODELS,
    EVALUATION_PERSONAS,
    HALLUCINATION_POLICY,
    HALLUCINATION_WEIGHT_THRESHOLD,
    RESPONSE_CACHE_ENABLED,
    SEMANTIC_ENTROPY_MODEL,
    SEMANTIC_ENTROPY_WARNING_THRESHOLD,
)
from src.output import save_results
from src.safety import classify_prompt_safety
from src.semantic_entropy import analyse_semantic_entropy

init(autoreset=True)


def _token_budget_exceeded(usage_summary, max_total_tokens, stage):
    if max_total_tokens is None or usage_summary["total_tokens"] <= max_total_tokens:
        return False

    print(
        Fore.RED
        + f"  ✗ Token budget exceeded during {stage}. "
        + f"Spent {usage_summary['total_tokens']} of {max_total_tokens} tokens."
    )
    return True


def _format_tokens(value):
    return f"{int(value):,}"


def _format_estimated_cost(value):
    if value is None:
        return "N/A"
    return f"${value:.4f}"


def _record_budget_estimate(usage_summary, estimate, attempt, status="estimated"):
    estimate["attempt"] = attempt
    estimate["status"] = status
    estimate["actual_prompt_tokens"] = None
    estimate["actual_completion_tokens"] = None
    estimate["actual_total_tokens"] = None
    estimate["actual_cost_usd"] = None
    usage_summary["estimates"].append(estimate)
    return estimate


def _complete_budget_estimate(estimate, usage_calls):
    usage_calls = usage_calls or []
    estimate["actual_prompt_tokens"] = sum(call["prompt_tokens"] for call in usage_calls)
    estimate["actual_completion_tokens"] = sum(call["completion_tokens"] for call in usage_calls)
    estimate["actual_total_tokens"] = sum(call["total_tokens"] for call in usage_calls)
    estimate["actual_cost_usd"] = round(sum(call["cost_usd"] for call in usage_calls), 6)
    estimate["status"] = "completed"


def _print_budget_estimate(estimate, usage_summary, max_total_tokens):
    stage = estimate["stage"].replace("_", " ")
    estimated_tokens = estimate["estimated_total_tokens"]
    estimated_cost = _format_estimated_cost(estimate.get("estimated_cost_usd"))
    cache_hits = estimate.get("cache_hits", 0)
    cache_suffix = f"; cache hits {cache_hits}" if cache_hits else ""

    if max_total_tokens is None:
        print(
            Fore.WHITE
            + f"  [budget] {stage}: estimated {_format_tokens(estimated_tokens)} tokens ({estimated_cost}){cache_suffix}."
        )
        return

    remaining = max_total_tokens - usage_summary["total_tokens"]
    projected = usage_summary["total_tokens"] + estimated_tokens
    print(
        Fore.WHITE
        + f"  [budget] {stage}: estimated {_format_tokens(estimated_tokens)} tokens ({estimated_cost}); "
        + f"remaining {_format_tokens(remaining)}; projected {_format_tokens(projected)}/{_format_tokens(max_total_tokens)}"
        + f"{cache_suffix}."
    )


def _estimated_budget_exceeded(usage_summary, estimate, max_total_tokens):
    if max_total_tokens is None:
        return False

    projected = usage_summary["total_tokens"] + estimate["estimated_total_tokens"]
    if projected <= max_total_tokens:
        return False

    estimate["status"] = "blocked"
    print(
        Fore.RED
        + f"  ✗ Estimated {estimate['stage'].replace('_', ' ')} usage would exceed the token budget. "
        + f"Projected {_format_tokens(projected)} of {_format_tokens(max_total_tokens)} tokens."
    )
    return True


def _empty_usage_summary(max_total_tokens=None):
    return {
        "api_calls": 0,
        "cache_enabled": RESPONSE_CACHE_ENABLED,
        "cache_hits": 0,
        "cached_completion_tokens": 0,
        "cached_cost_usd": 0.0,
        "cached_prompt_tokens": 0,
        "cached_total_tokens": 0,
        "calls": [],
        "estimates": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "max_total_tokens": max_total_tokens,
        "remaining_tokens": None if max_total_tokens is None else max_total_tokens,
    }


def _extend_usage_summary(usage_summary, new_calls):
    if not new_calls:
        return usage_summary

    usage_summary["calls"].extend(new_calls)
    usage_summary["api_calls"] += sum(1 for call in new_calls if not call.get("cached"))
    usage_summary["cache_hits"] += sum(1 for call in new_calls if call.get("cached"))
    usage_summary["cached_prompt_tokens"] += sum(call.get("cached_prompt_tokens", 0) for call in new_calls)
    usage_summary["cached_completion_tokens"] += sum(call.get("cached_completion_tokens", 0) for call in new_calls)
    usage_summary["cached_total_tokens"] += sum(call.get("cached_total_tokens", 0) for call in new_calls)
    usage_summary["cached_cost_usd"] = round(
        usage_summary["cached_cost_usd"] + sum(call.get("cached_cost_usd", 0.0) for call in new_calls),
        4,
    )
    usage_summary["total_prompt_tokens"] += sum(call.get("prompt_tokens", 0) for call in new_calls)
    usage_summary["total_completion_tokens"] += sum(call.get("completion_tokens", 0) for call in new_calls)
    usage_summary["total_tokens"] += sum(call.get("total_tokens", 0) for call in new_calls)
    usage_summary["total_cost_usd"] = round(
        usage_summary["total_cost_usd"] + sum(call.get("cost_usd", 0.0) for call in new_calls),
        4,
    )

    max_total_tokens = usage_summary["max_total_tokens"]
    if max_total_tokens is not None:
        usage_summary["remaining_tokens"] = max(0, max_total_tokens - usage_summary["total_tokens"])

    return usage_summary


def _select_generation_models(generation_count):
    max_generation = len(GENERATION_MODELS)
    min_generation = 2 if max_generation >= 2 else max_generation

    if generation_count is None:
        generation_count = max_generation

    generation_count = max(min_generation, min(generation_count, max_generation))
    return GENERATION_MODELS[:generation_count]


def _select_evaluation_setup(evaluation_count):
    max_evaluators = len(EVALUATION_MODELS)

    if evaluation_count is None:
        evaluation_count = max_evaluators

    evaluation_count = max(1, min(evaluation_count, max_evaluators))

    return (
        EVALUATION_PERSONAS[:evaluation_count],
        EVALUATION_MODELS[:evaluation_count],
        min(MIN_VALID_ARBITERS, evaluation_count),
    )


def _valid_arbiter_count(evaluations):
    return sum(1 for value in evaluations.values() if value is not None)


def _build_run_config(generation_models, evaluation_personas, evaluation_models, min_valid_arbiters):
    return {
        "generation_models": generation_models,
        "generation_count": len(generation_models),
        "evaluation_personas": evaluation_personas,
        "evaluation_models": evaluation_models,
        "evaluation_count": len(evaluation_models),
        "min_valid_arbiters": min_valid_arbiters,
        "hallucination_policy": HALLUCINATION_POLICY,
        "hallucination_weight_threshold": HALLUCINATION_WEIGHT_THRESHOLD,
    }


def _build_results(
    user_prompt,
    report_name,
    attempt,
    elapsed_time,
    final_score,
    best_response,
    anonymised,
    evaluations,
    aggregation,
    usage_summary,
    run_config,
    safety_response=None,
):
    results = {
        "prompt": user_prompt,
        "report_name": report_name,
        "attempt": attempt,
        "elapsed_time": elapsed_time,
        "final_score": final_score,
        "best_response": best_response,
        "anonymised_responses": anonymised,
        "evaluations": evaluations,
        "aggregation": aggregation,
        "usage": usage_summary,
        "run_config": run_config,
    }

    if safety_response is not None:
        results["safety_response"] = safety_response

    return results


def run_council(
    user_prompt,
    score_threshold,
    max_retries,
    report_name=None,
    max_total_tokens=None,
    generation_count=None,
    evaluation_count=None,
):
    attempt = 0
    results = None
    usage_summary = _empty_usage_summary(max_total_tokens)
    generation_models = _select_generation_models(generation_count)
    evaluation_personas, evaluation_models, min_valid_arbiters = _select_evaluation_setup(evaluation_count)
    run_config = _build_run_config(
        generation_models,
        evaluation_personas,
        evaluation_models,
        min_valid_arbiters,
    )
    safety_response = classify_prompt_safety(user_prompt)

    if safety_response["blocked"]:
        print(Fore.RED + f"\n  ✗ {safety_response['title']}. Council run blocked.")
        results = _build_results(
            user_prompt=user_prompt,
            report_name=report_name,
            attempt=0,
            elapsed_time=0,
            final_score=None,
            best_response=None,
            anonymised=[],
            evaluations={},
            aggregation=None,
            usage_summary=usage_summary,
            run_config=run_config,
            safety_response=safety_response,
        )
        save_results(results)
        return results

    while attempt < max_retries:
        if max_total_tokens is not None and usage_summary["total_tokens"] >= max_total_tokens:
            print(
                Fore.RED
                + f"\n  ✗ Token budget exhausted before attempt {attempt + 1}. "
                + f"Spent {usage_summary['total_tokens']} of {max_total_tokens} tokens."
            )
            return None

        attempt += 1
        print(Fore.YELLOW + f"\n  ── Attempt {attempt} of {max_retries} " + "─" * 40)
        start_time = time.time()

        preflight_estimate = _record_budget_estimate(
            usage_summary,
            estimate_attempt_budget(
                user_prompt,
                generation_models,
                evaluation_personas,
                evaluation_models,
                SEMANTIC_ENTROPY_MODEL,
            ),
            attempt,
            status="preflight",
        )
        _print_budget_estimate(preflight_estimate, usage_summary, max_total_tokens)
        if _estimated_budget_exceeded(usage_summary, preflight_estimate, max_total_tokens):
            return None

        generation_estimate = _record_budget_estimate(
            usage_summary,
            estimate_generation_stage(user_prompt, generation_models),
            attempt,
        )
        _print_budget_estimate(generation_estimate, usage_summary, max_total_tokens)
        if _estimated_budget_exceeded(usage_summary, generation_estimate, max_total_tokens):
            return None

        responses, gen_usage = generate_responses(user_prompt, generation_models=generation_models)
        _complete_budget_estimate(generation_estimate, gen_usage)
        _extend_usage_summary(usage_summary, gen_usage)

        if _token_budget_exceeded(usage_summary, max_total_tokens, "generation"):
            return None

        if not responses:
            print(Fore.RED + "  ✗ No responses generated. Retrying...")
            continue

        if len(responses) < 2:
            print(Fore.RED + "  ✗ Insufficient responses. Retrying...")
            continue

        anonymised = anonymise_and_shuffle(responses, user_prompt)

        evaluation_estimate = _record_budget_estimate(
            usage_summary,
            estimate_evaluation_stage(user_prompt, anonymised, evaluation_personas, evaluation_models),
            attempt,
        )
        _print_budget_estimate(evaluation_estimate, usage_summary, max_total_tokens)
        if _estimated_budget_exceeded(usage_summary, evaluation_estimate, max_total_tokens):
            return None

        evaluations, eval_usage = evaluate_responses(
            user_prompt,
            anonymised,
            evaluation_personas=evaluation_personas,
            evaluation_models=evaluation_models,
            min_valid_arbiters=min_valid_arbiters,
        )
        _complete_budget_estimate(evaluation_estimate, eval_usage)
        _extend_usage_summary(usage_summary, eval_usage)

        if _token_budget_exceeded(usage_summary, max_total_tokens, "evaluation"):
            return None

        valid_arbiter_count = _valid_arbiter_count(evaluations)
        if valid_arbiter_count < min_valid_arbiters:
            print(Fore.RED + f"  ✗ Only {valid_arbiter_count} valid arbiter(s); need at least {min_valid_arbiters}. Retrying...")
            continue

        best_response, response_scores, persona_breakdown = select_best_response(anonymised, evaluations)

        if best_response is None:
            print(Fore.RED + "  ✗ Best response selection failed. Retrying...")
            continue

        semantic_entropy_estimate = _record_budget_estimate(
            usage_summary,
            estimate_semantic_entropy_stage(user_prompt, anonymised, SEMANTIC_ENTROPY_MODEL),
            attempt,
        )
        _print_budget_estimate(semantic_entropy_estimate, usage_summary, max_total_tokens)
        if _estimated_budget_exceeded(usage_summary, semantic_entropy_estimate, max_total_tokens):
            return None

        semantic_entropy, semantic_usage = analyse_semantic_entropy(user_prompt, anonymised, response_scores)
        _complete_budget_estimate(semantic_entropy_estimate, semantic_usage)
        _extend_usage_summary(usage_summary, semantic_usage)

        if _token_budget_exceeded(usage_summary, max_total_tokens, "semantic entropy analysis"):
            return None

        aggregation = calculate_final_score(evaluations, best_response["id"])

        if aggregation is None:
            print(Fore.RED + "  ✗ Aggregation failed. Retrying...")
            continue

        aggregation["response_scores"] = response_scores
        aggregation["response_persona_scores"] = persona_breakdown
        aggregation["semantic_entropy"] = semantic_entropy

        final_score = aggregation["final_score"]

        if semantic_entropy is not None:
            print(
                Fore.WHITE
                + f"  ✦ Semantic entropy: {semantic_entropy['normalized_entropy']} "
                + f"(agreement: {semantic_entropy['agreement_score']})"
            )
            if semantic_entropy["normalized_entropy"] >= SEMANTIC_ENTROPY_WARNING_THRESHOLD:
                print(Fore.YELLOW + "  [WARN] High semantic disagreement detected across candidate responses.")

        if final_score < score_threshold:
            print(Fore.RED + f"  ✗ Score {final_score}/100 below threshold {score_threshold}/100. Retrying...")
            continue

        elapsed = round(time.time() - start_time, 2)

        print(Fore.GREEN + f"\n  ✓ Completed in {elapsed}s  ·  Final score: {final_score}/100")
        print(Fore.WHITE  + f"  ✦ API usage: {usage_summary['total_tokens']} tokens  ·  ${usage_summary['total_cost_usd']:.4f}")
        if usage_summary["cache_hits"]:
            print(
                Fore.WHITE
                + f"  ✦ Cache: {usage_summary['cache_hits']} hit(s)  ·  saved "
                + f"{usage_summary['cached_total_tokens']} tokens  ·  ${usage_summary['cached_cost_usd']:.4f}"
            )

        results = _build_results(
            user_prompt=user_prompt,
            report_name=report_name,
            attempt=attempt,
            elapsed_time=elapsed,
            final_score=final_score,
            best_response=best_response,
            anonymised=anonymised,
            evaluations=evaluations,
            aggregation=aggregation,
            usage_summary=usage_summary,
            run_config=run_config,
        )

        break

    if results is None:
        print(Fore.RED + f"\n  ✗ Failed after {max_retries} attempt(s). Best score below threshold ({score_threshold}/100).")
        return None

    save_results(results)
    return results
