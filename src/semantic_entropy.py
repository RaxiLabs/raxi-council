import json
import math
from colorama import Fore, init
from src.agents import call_llm, _extract_json_object, load_prompt
from src.cache import get_cached_result, store_cached_result
from src.config import SEMANTIC_ENTROPY_MAX_TOKENS, SEMANTIC_ENTROPY_MODEL

init(autoreset=True)

SEMANTIC_ENTROPY_CACHE_NAMESPACE = "semantic_entropy_clusters"

def _normalize_zero(value, precision=4):
    rounded = round(value, precision)
    return 0.0 if abs(rounded) < 10 ** (-precision) else rounded

def _singleton_clusters(anonymised_responses):
    return [[item["id"]] for item in anonymised_responses]

def _validate_clusters(payload, expected_response_ids):
    if not isinstance(payload, dict):
        raise ValueError("Semantic entropy payload was not a JSON object.")

    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Semantic entropy payload is missing a valid 'clusters' list.")

    expected_set = set(expected_response_ids)
    seen = set()
    normalized_clusters = []

    for cluster in clusters:
        if not isinstance(cluster, list) or not cluster:
            raise ValueError("Each semantic cluster must be a non-empty list.")

        normalized_cluster = []
        for response_id in cluster:
            if not isinstance(response_id, str):
                raise ValueError("Semantic cluster response ids must be strings.")
            if response_id not in expected_set:
                raise ValueError(f"Unexpected response id in semantic cluster: {response_id}.")
            if response_id in seen:
                raise ValueError(f"Duplicate response id in semantic clusters: {response_id}.")
            seen.add(response_id)
            normalized_cluster.append(response_id)

        normalized_clusters.append(normalized_cluster)

    missing = [response_id for response_id in expected_response_ids if response_id not in seen]
    if missing:
        raise ValueError(f"Missing response ids from semantic clusters: {', '.join(missing)}.")

    order = {response_id: index for index, response_id in enumerate(expected_response_ids)}
    normalized_clusters = [
        sorted(cluster, key=lambda response_id: order[response_id])
        for cluster in normalized_clusters
    ]
    normalized_clusters.sort(key=lambda cluster: order[cluster[0]])

    return normalized_clusters

def _build_semantic_entropy_metrics(clusters, response_scores):
    cluster_entries = []
    score_masses = []

    for cluster in clusters:
        score_mass = sum(max(response_scores.get(response_id, 0.0), 0.0) for response_id in cluster)
        score_masses.append(score_mass)

    use_score_mass = any(score_mass > 0 for score_mass in score_masses)
    masses = score_masses if use_score_mass else [len(cluster) for cluster in clusters]
    total_mass = sum(masses)

    probabilities = [mass / total_mass for mass in masses] if total_mass else []
    raw_entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0) # Actual entropy formula translated in python
    max_entropy = math.log(len(clusters)) if len(clusters) > 1 else 0.0
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    agreement_score = 1.0 - normalized_entropy if len(clusters) > 1 else 1.0
    dominant_cluster_probability = max(probabilities) if probabilities else 1.0

    for index, cluster in enumerate(clusters, start=1):
        cluster_entries.append({
            "cluster_id": f"Cluster_{index}",
            "response_ids": cluster,
            "size": len(cluster),
            "probability": _normalize_zero(probabilities[index - 1]) if probabilities else 1.0,
            "mass": _normalize_zero(masses[index - 1]) if masses else 0.0,
        })

    return {
        "cluster_count": len(clusters),
        "clusters": cluster_entries,
        "raw_entropy": _normalize_zero(raw_entropy),
        "normalized_entropy": _normalize_zero(normalized_entropy),
        "agreement_score": _normalize_zero(agreement_score),
        "dominant_cluster_probability": _normalize_zero(dominant_cluster_probability),
        "mass_basis": "weighted_response_scores" if use_score_mass else "cluster_size",
    }


def build_semantic_entropy_prompt(user_prompt, anonymised_responses):
    formatted_responses = []
    for item in anonymised_responses:
        formatted_responses.append(f"{item['id']}:\n{item['response']}")

    return (
        f"Original question: {user_prompt}\n\n"
        "Cluster the following responses by semantic equivalence.\n\n"
        + "\n\n".join(formatted_responses)
    )


def build_semantic_entropy_cache_request(user_prompt, anonymised_responses):
    return {
        "max_tokens": SEMANTIC_ENTROPY_MAX_TOKENS,
        "model": SEMANTIC_ENTROPY_MODEL,
        "system_prompt": load_prompt("semantic_entropy"),
        "user_prompt": build_semantic_entropy_prompt(user_prompt, anonymised_responses),
    }


def analyse_semantic_entropy(user_prompt, anonymised_responses, response_scores):
    if not anonymised_responses:
        return None, []

    if len(anonymised_responses) == 1:
        clusters = _singleton_clusters(anonymised_responses)
        metrics = _build_semantic_entropy_metrics(clusters, response_scores)
        metrics["method"] = "singleton_fallback"
        metrics["model"] = None
        return metrics, []

    response_ids = [item["id"] for item in anonymised_responses]
    cache_request = build_semantic_entropy_cache_request(user_prompt, anonymised_responses)
    cached_clusters, cached_usage = get_cached_result(SEMANTIC_ENTROPY_CACHE_NAMESPACE, cache_request)

    if cached_clusters is not None:
        metrics = _build_semantic_entropy_metrics(cached_clusters, response_scores)
        metrics["method"] = "llm_semantic_clustering_cached"
        metrics["model"] = SEMANTIC_ENTROPY_MODEL
        return metrics, cached_usage

    raw, usage = call_llm(
        model=SEMANTIC_ENTROPY_MODEL,
        system_prompt=cache_request["system_prompt"],
        user_prompt=cache_request["user_prompt"],
        max_tokens=SEMANTIC_ENTROPY_MAX_TOKENS,
    )

    usage_entries = [usage] if usage else []

    if not raw:
        print(Fore.YELLOW + "  [SEMANTIC] Unable to compute semantic clusters. Falling back to singleton clusters.")
        clusters = _singleton_clusters(anonymised_responses)
        metrics = _build_semantic_entropy_metrics(clusters, response_scores)
        metrics["method"] = "singleton_fallback"
        metrics["model"] = SEMANTIC_ENTROPY_MODEL
        return metrics, usage_entries

    try:
        parsed = _extract_json_object(raw)
        clusters = _validate_clusters(parsed, response_ids)
        store_cached_result(
            SEMANTIC_ENTROPY_CACHE_NAMESPACE,
            cache_request,
            clusters,
            [usage] if usage else [],
        )
        metrics = _build_semantic_entropy_metrics(clusters, response_scores)
        metrics["method"] = "llm_semantic_clustering"
        metrics["model"] = SEMANTIC_ENTROPY_MODEL
        return metrics, usage_entries
    except (json.JSONDecodeError, ValueError) as error:
        print(Fore.YELLOW + f"  [SEMANTIC] Invalid semantic clustering payload ({error}). Falling back to singleton clusters.")
        clusters = _singleton_clusters(anonymised_responses)
        metrics = _build_semantic_entropy_metrics(clusters, response_scores)
        metrics["method"] = "singleton_fallback"
        metrics["model"] = SEMANTIC_ENTROPY_MODEL
        return metrics, usage_entries
