from colorama import Fore, init
from src.config import (
    ARBITER_WEIGHTS,
    DIMENSION_WEIGHTS,
    HALLUCINATION_POLICY,
    HALLUCINATION_WEIGHT_THRESHOLD,
)

init(autoreset=True)


def _empty_arbiter_result(raw_scores, hallucination_flags, hallucination_details, hallucination_assessments, persona):
    raw_scores[persona] = None
    hallucination_flags[persona] = None
    hallucination_details[persona] = None
    hallucination_assessments[persona] = None


def _calculate_hallucination_result(hallucination_flags):
    valid_flags = {persona: flag for persona, flag in hallucination_flags.items() if isinstance(flag, bool)}
    flagged_count = sum(valid_flags.values())
    flagged_weight = sum(ARBITER_WEIGHTS.get(persona, 0.0) for persona, flag in valid_flags.items() if flag)

    if not valid_flags:
        return False, {
            "policy": HALLUCINATION_POLICY,
            "threshold": HALLUCINATION_WEIGHT_THRESHOLD,
            "flagged_count": 0,
            "flagged_weight": 0.0,
            "valid_count": 0,
        }

    base_result = {
        "policy": HALLUCINATION_POLICY,
        "threshold": None,
        "flagged_count": flagged_count,
        "flagged_weight": round(flagged_weight, 4),
        "valid_count": len(valid_flags),
    }

    if HALLUCINATION_POLICY == "majority":
        return flagged_count > (len(valid_flags) / 2), base_result

    if HALLUCINATION_POLICY == "weighted":
        base_result["threshold"] = HALLUCINATION_WEIGHT_THRESHOLD
        return flagged_weight >= HALLUCINATION_WEIGHT_THRESHOLD, {
            **base_result,
        }

    return any(valid_flags.values()), {
        **base_result,
        "policy": "any",
    }


def calculate_weighted_dimension_score(evaluation):
    try:
        score = 0.0
        for dimension, weight in DIMENSION_WEIGHTS.items():
            dimension_score = evaluation["dimensions"][dimension]["score"]
            if not isinstance(dimension_score, (int, float)):
                raise TypeError(f"{dimension} score must be numeric, got {type(dimension_score).__name__}")
            if dimension_score < 0 or dimension_score > 10:
                raise ValueError(f"{dimension} score out of range: {dimension_score}")
            score += dimension_score * weight
        return round(score, 2)
    except (KeyError, TypeError, ValueError) as e:
        print(Fore.RED + f"[ERROR] Failed to calculate dimension score: {e}")
        return None


def _extract_response_scores(evaluations):
    response_scores = {}
    persona_breakdown = {}

    for persona, evaluation in evaluations.items():
        if evaluation is None:
            continue

        weight = ARBITER_WEIGHTS.get(persona, 0)
        response_evaluations = evaluation.get("evaluations", {})

        for response_id, response_evaluation in response_evaluations.items():
            score = calculate_weighted_dimension_score(response_evaluation)
            if score is None:
                continue

            response_scores[response_id] = response_scores.get(response_id, 0.0) + score * weight
            persona_breakdown.setdefault(response_id, {})[persona] = score

    return response_scores, persona_breakdown


def calculate_final_score(evaluations, best_response_id):
    print(Fore.YELLOW + "\n  ── Aggregation " + "─" * 45)

    raw_scores = {}
    hallucination_flags = {}
    hallucination_details = {}
    hallucination_assessments = {}

    for persona, evaluation in evaluations.items():
        if evaluation is None:
            print(Fore.RED + f"  [{persona.upper()}] No evaluation. Skipping.")
            _empty_arbiter_result(
                raw_scores,
                hallucination_flags,
                hallucination_details,
                hallucination_assessments,
                persona,
            )
            continue

        response_evaluation = evaluation.get("evaluations", {}).get(best_response_id)
        if response_evaluation is None:
            print(Fore.RED + f"  [{persona.upper()}] Missing evaluation for {best_response_id}. Skipping.")
            _empty_arbiter_result(
                raw_scores,
                hallucination_flags,
                hallucination_details,
                hallucination_assessments,
                persona,
            )
            continue

        score = calculate_weighted_dimension_score(response_evaluation)
        if score is None:
            print(Fore.RED + f"  [{persona.upper()}] Invalid score payload for {best_response_id}. Skipping.")
            _empty_arbiter_result(
                raw_scores,
                hallucination_flags,
                hallucination_details,
                hallucination_assessments,
                persona,
            )
            continue

        raw_scores[persona] = score
        hallucination_flags[persona] = response_evaluation.get("hallucination_flag", False)
        hallucination_details[persona] = response_evaluation.get("hallucination_detail")
        hallucination_assessments[persona] = response_evaluation.get("hallucination_assessment")
        print(Fore.CYAN + f"    {persona.upper():<12} {score}/10")

    final_score = 0.0
    total_weight = 0.0
    for persona, weight in ARBITER_WEIGHTS.items():
        score = raw_scores.get(persona)
        if score is not None:
            final_score += score * weight
            total_weight += weight

    if total_weight == 0:
        print(Fore.RED + "[ERROR] No valid scores to aggregate.")
        return None

    if total_weight < 1.0:
        final_score = final_score / total_weight

    final_score_normalised = round(max(0, min(100, final_score / 10 * 100)), 2)
    any_hallucination, hallucination_policy_result = _calculate_hallucination_result(hallucination_flags)

    print(Fore.GREEN + f"\n    Score  →  {final_score_normalised}/100")
    if any_hallucination:
        print(Fore.RED + "    ⚠  Hallucination flagged by one or more arbiters.")

    return {
        "best_response_id": best_response_id,
        "final_score": final_score_normalised,
        "raw_scores": raw_scores,
        "any_hallucination": any_hallucination,
        "hallucination_flags": hallucination_flags,
        "hallucination_details": hallucination_details,
        "hallucination_assessments": hallucination_assessments,
        "hallucination_policy_result": hallucination_policy_result,
    }


def select_best_response(anonymised_responses, evaluations):
    response_scores, persona_breakdown = _extract_response_scores(evaluations)

    if not response_scores:
        best_response = anonymised_responses[0] if anonymised_responses else None
        return best_response, response_scores, persona_breakdown

    best_id = max(response_scores, key=response_scores.get)
    best_response = next(
        (response for response in anonymised_responses if response["id"] == best_id),
        anonymised_responses[0]
    )

    print(Fore.GREEN + f"    Best response  →  {best_id}  (weighted score: {round(response_scores[best_id], 2)})")
    return best_response, response_scores, persona_breakdown
