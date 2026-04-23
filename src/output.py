import os
import json
from datetime import datetime
from colorama import Fore, init
from src.config import OUTPUT_DIR

init(autoreset=True)


def _markdown_quote(text):
    if not text:
        return "> _No content available._"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _format_currency(amount):
    return f"${amount:.4f}" if amount is not None else "N/A"


def _format_token_count(amount):
    return str(int(amount)) if amount is not None else "N/A"


def _markdown_inline(text):
    if text is None:
        return "N/A"
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _truncate_text(text, max_length=180):
    cleaned = _markdown_inline(text)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3].rstrip() + "..."


def _build_report_filename(report_name=None, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    if not report_name:
        return f"raxi_court_output_{timestamp}.md"

    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in str(report_name).strip())
    sanitized = "_".join(part for part in sanitized.split("_") if part)

    if not sanitized:
        return f"raxi_court_output_{timestamp}.md"

    return f"{sanitized}.md"


def _format_single_evaluation(evaluation):
    if evaluation is None:
        return "_No evaluation available._"

    lines = []
    lines.append(f"- Confidence: `{evaluation.get('evaluator_confidence', 'N/A')}`")
    lines.append(f"- Hallucination: `{evaluation.get('hallucination_flag', False)}`")

    hallucination_assessment = evaluation.get("hallucination_assessment")
    if hallucination_assessment:
        severity = hallucination_assessment.get("severity")
        if severity:
            lines.append(f"- Hallucination severity: `{severity}`")
        issues = hallucination_assessment.get("issues") or []
        if issues:
            lines.append("- Hallucination issues:")
            for issue in issues:
                claim = issue.get("claim") or issue.get("span_text") or "Unspecified claim"
                reason = issue.get("reason") or "No reason provided."
                category = issue.get("category")
                if category:
                    lines.append(
                        f"  - {_truncate_text(claim)} "
                        f"(`{_markdown_inline(category)}`): {_truncate_text(reason)}"
                    )
                else:
                    lines.append(f"  - {_truncate_text(claim)}: {_truncate_text(reason)}")
    elif evaluation.get("hallucination_detail"):
        lines.append(f"- Hallucination detail: {_truncate_text(evaluation['hallucination_detail'])}")

    lines.append(f"- Overall: {_truncate_text(evaluation.get('evaluation_notes', 'No additional notes provided.'))}")
    lines.append("")
    lines.append("| Dimension | Score | Reason |")
    lines.append("|---|---:|---|")

    for dimension, data in evaluation.get("dimensions", {}).items():
        label = dimension.replace("_", " ").title()
        score = data.get("score", "N/A")
        reason = _markdown_inline(data.get("reason", "N/A"))
        lines.append(f"| {label} | {score}/10 | {reason} |")

    lines.append("")
    lines.append(f"- Total score: `{evaluation.get('total_score', 'N/A')}/30`")
    return "\n".join(lines)


def _format_safety_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safety_response = results.get("safety_response", {})
    usage = results.get("usage", {})
    lines = []

    lines.append("# Raxi Court Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Prompt: `{_markdown_inline(results['prompt'])}`")
    if results.get("report_name"):
        lines.append(f"- Report name: `{_markdown_inline(results['report_name'])}`")
    lines.append("- Council run started: `False`")
    lines.append("- Safety blocked: `True`")
    lines.append(f"- Safety tag: `{_markdown_inline(safety_response.get('tag'))}`")
    lines.append(f"- Safety title: `{_markdown_inline(safety_response.get('title'))}`")
    lines.append(f"- Safety category: `{_markdown_inline(safety_response.get('category'))}`")
    lines.append("")
    lines.append("## Safety Response")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(safety_response, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Attempt: `{results.get('attempt', 0)}`")
    lines.append(f"- Elapsed time: `{results.get('elapsed_time', 0)}s`")
    lines.append(f"- Total tokens: `{usage.get('total_tokens', 0)}`")
    lines.append(f"- Estimated API cost: `{_format_currency(usage.get('total_cost_usd', 0))}`")
    lines.append("")

    return "\n".join(lines)


def format_results(results):
    if results.get("safety_response", {}).get("blocked"):
        return _format_safety_results(results)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    usage = results.get("usage", {})
    aggregation = results["aggregation"]
    semantic_entropy = aggregation.get("semantic_entropy")
    best_response = results["best_response"]
    run_config = results.get("run_config", {})
    lines = []

    lines.append("# Raxi Court Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Prompt: `{_markdown_inline(results['prompt'])}`")
    if results.get("report_name"):
        lines.append(f"- Report name: `{_markdown_inline(results['report_name'])}`")
    lines.append(f"- Attempt: `{results['attempt']}`")
    lines.append(f"- Elapsed time: `{results['elapsed_time']}s`")
    lines.append(f"- Final score: `{results['final_score']}/100`")
    lines.append(f"- Hallucination flagged: `{aggregation['any_hallucination']}`")
    lines.append(f"- Hallucination policy: `{_markdown_inline(run_config.get('hallucination_policy', 'N/A'))}`")
    if run_config.get("hallucination_policy") == "weighted":
        lines.append(f"- Hallucination weight threshold: `{run_config.get('hallucination_weight_threshold', 'N/A')}`")
    lines.append(f"- Best response: `{best_response['id']}`")
    lines.append(f"- Best response model: `{best_response['_original_model']}`")
    lines.append(f"- Generation models used (k): `{run_config.get('generation_count', 'N/A')}`")
    lines.append(f"- Evaluation arbiters used (m): `{run_config.get('evaluation_count', 'N/A')}`")
    generation_models = run_config.get("generation_models")
    if generation_models:
        lines.append(f"- Generation model pool: `{_markdown_inline(', '.join(generation_models))}`")
    evaluation_personas = run_config.get("evaluation_personas")
    evaluation_models = run_config.get("evaluation_models")
    if evaluation_personas and evaluation_models:
        lines.append("- Evaluation arbiters:")
        for persona, model in zip(evaluation_personas, evaluation_models):
            lines.append(f"  - `{_markdown_inline(persona)}` -> `{_markdown_inline(model)}`")
    if semantic_entropy is not None:
        lines.append(f"- Semantic entropy: `{semantic_entropy.get('normalized_entropy', 'N/A')}`")
        lines.append(f"- Semantic agreement: `{semantic_entropy.get('agreement_score', 'N/A')}`")
    lines.append(f"- Total tokens: `{usage.get('total_tokens', 'N/A')}`")
    if usage.get("max_total_tokens") is not None:
        lines.append(f"- Token budget: `{usage['max_total_tokens']}`")
        lines.append(f"- Remaining tokens: `{usage.get('remaining_tokens', 'N/A')}`")
    attempt_estimates = [
        estimate
        for estimate in usage.get("estimates", [])
        if estimate.get("stage") == "attempt"
    ]
    if attempt_estimates:
        latest_attempt_estimate = attempt_estimates[-1]
        lines.append(f"- Estimated attempt tokens: `{latest_attempt_estimate.get('estimated_total_tokens', 'N/A')}`")
        lines.append(f"- Estimated attempt cost: `{_format_currency(latest_attempt_estimate.get('estimated_cost_usd'))}`")
    lines.append(f"- Estimated API cost: `{_format_currency(usage.get('total_cost_usd'))}`")
    lines.append("")

    lines.append("## Response Scoreboard")
    lines.append("")
    lines.append("| Response | Model | Weighted Score |")
    lines.append("|---|---|---:|")
    for item in results["anonymised_responses"]:
        response_id = item["id"]
        weighted_score = aggregation.get("response_scores", {}).get(response_id)
        weighted_label = f"{round(weighted_score, 2)}/10" if weighted_score is not None else "N/A"
        lines.append(f"| {response_id} | `{item['_original_model']}` | {weighted_label} |")
    lines.append("")

    lines.append("## Best Response")
    lines.append("")
    lines.append(f"### {best_response['id']}")
    lines.append("")
    lines.append(f"**Original model:** `{best_response['_original_model']}`")
    lines.append("")
    lines.append(_markdown_quote(best_response["response"]))
    lines.append("")

    lines.append("## All Responses")
    lines.append("")
    for item in results["anonymised_responses"]:
        lines.append(f"### {item['id']}")
        lines.append("")
        lines.append(f"**Original model:** `{item['_original_model']}`")
        persona_scores = aggregation.get("response_persona_scores", {}).get(item["id"], {})
        if persona_scores:
            lines.append("")
            lines.append("| Arbiter | Score |")
            lines.append("|---|---:|")
            for persona, score in persona_scores.items():
                lines.append(f"| {persona.upper()} | {score}/10 |")
        lines.append("")
        lines.append(_markdown_quote(item["response"]))
        lines.append("")

    lines.append("## Winning Response Evaluations")
    lines.append("")
    for persona, evaluation in results["evaluations"].items():
        lines.append(f"### {persona.upper()}")
        lines.append("")
        winning_evaluation = None if evaluation is None else evaluation.get("evaluations", {}).get(best_response["id"])
        lines.append(_format_single_evaluation(winning_evaluation))
        lines.append("")

    lines.append("## Aggregation")
    lines.append("")
    lines.append(f"- Winning response id: `{aggregation['best_response_id']}`")
    lines.append(f"- Final score: `{aggregation['final_score']}/100`")
    lines.append(f"- Hallucination flagged: `{aggregation['any_hallucination']}`")
    policy_result = aggregation.get("hallucination_policy_result") or {}
    lines.append(f"- Hallucination decision policy: `{_markdown_inline(policy_result.get('policy', run_config.get('hallucination_policy', 'N/A')))}`")
    if policy_result.get("threshold") is not None:
        lines.append(f"- Hallucination threshold: `{policy_result['threshold']}`")
    if policy_result.get("flagged_count") is not None:
        lines.append(
            f"- Hallucination votes: `{policy_result['flagged_count']}/{policy_result.get('valid_count', 'N/A')}`"
        )
    if policy_result.get("flagged_weight") is not None:
        lines.append(f"- Hallucination weight: `{policy_result['flagged_weight']}`")
    lines.append("")
    lines.append("| Arbiter | Winning Response Score | Hallucination |")
    lines.append("|---|---:|---|")
    for persona, score in aggregation["raw_scores"].items():
        hall = aggregation["hallucination_flags"].get(persona)
        score_label = f"{score}/10" if score is not None else "N/A"
        hall_label = hall if hall is not None else "N/A"
        lines.append(f"| {persona.upper()} | {score_label} | {hall_label} |")

    if aggregation["any_hallucination"]:
        lines.append("")
        lines.append("- Hallucination details:")
        for persona, assessment in aggregation.get("hallucination_assessments", {}).items():
            if assessment and assessment.get("issues"):
                severity = assessment.get("severity")
                issue_lines = []
                for issue in assessment.get("issues", []):
                    claim = issue.get("claim") or issue.get("span_text") or "Unspecified claim"
                    reason = issue.get("reason") or "No reason provided."
                    category = issue.get("category")
                    if category:
                        issue_lines.append(
                            f"{_truncate_text(claim)} (`{_markdown_inline(category)}`): {_truncate_text(reason)}"
                        )
                    else:
                        issue_lines.append(f"{_truncate_text(claim)}: {_truncate_text(reason)}")
                prefix = f"  - {persona.upper()}"
                if severity:
                    prefix += f" (`{severity}`)"
                lines.append(prefix + ":")
                for issue_line in issue_lines:
                    lines.append(f"    - {issue_line}")
                continue

            detail = aggregation["hallucination_details"].get(persona)
            if detail:
                lines.append(f"  - {persona.upper()}: {_truncate_text(detail)}")
    lines.append("")

    if semantic_entropy is not None:
        lines.append("## Semantic Entropy")
        lines.append("")
        lines.append(f"- Method: `{_markdown_inline(semantic_entropy.get('method', 'N/A'))}`")
        lines.append(f"- Model: `{_markdown_inline(semantic_entropy.get('model', 'N/A'))}`")
        lines.append(f"- Cluster count: `{semantic_entropy.get('cluster_count', 'N/A')}`")
        lines.append(f"- Raw entropy: `{semantic_entropy.get('raw_entropy', 'N/A')}`")
        lines.append(f"- Normalized entropy: `{semantic_entropy.get('normalized_entropy', 'N/A')}`")
        lines.append(f"- Agreement score: `{semantic_entropy.get('agreement_score', 'N/A')}`")
        lines.append(f"- Dominant cluster probability: `{semantic_entropy.get('dominant_cluster_probability', 'N/A')}`")
        lines.append(f"- Mass basis: `{_markdown_inline(semantic_entropy.get('mass_basis', 'N/A'))}`")
        lines.append("")
        lines.append("| Cluster | Responses | Size | Probability | Mass |")
        lines.append("|---|---|---:|---:|---:|")
        for cluster in semantic_entropy.get("clusters", []):
            response_ids = ", ".join(cluster.get("response_ids", []))
            lines.append(
                f"| {cluster.get('cluster_id', 'N/A')} | `{_markdown_inline(response_ids)}` | "
                f"{cluster.get('size', 'N/A')} | {cluster.get('probability', 'N/A')} | {cluster.get('mass', 'N/A')} |"
            )
        lines.append("")

    lines.append("## API Usage")
    lines.append("")
    lines.append(f"- Total prompt tokens: `{usage.get('total_prompt_tokens', 'N/A')}`")
    lines.append(f"- Total completion tokens: `{usage.get('total_completion_tokens', 'N/A')}`")
    lines.append(f"- Total tokens: `{usage.get('total_tokens', 'N/A')}`")
    if usage.get("max_total_tokens") is not None:
        lines.append(f"- Token budget: `{usage['max_total_tokens']}`")
        lines.append(f"- Remaining tokens: `{usage.get('remaining_tokens', 'N/A')}`")
    lines.append(f"- Estimated cost: `{_format_currency(usage.get('total_cost_usd'))}`")
    lines.append("")

    estimates = usage.get("estimates") or []
    if estimates:
        lines.append("## Token Budget Estimates")
        lines.append("")
        lines.append("| Attempt | Stage | Status | Estimated Tokens | Actual Tokens | Estimated Cost | Actual Cost |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for estimate in estimates:
            lines.append(
                f"| {estimate.get('attempt', 'N/A')} | {_markdown_inline(estimate.get('stage', 'N/A'))} | "
                f"{_markdown_inline(estimate.get('status', 'N/A'))} | "
                f"{_format_token_count(estimate.get('estimated_total_tokens'))} | "
                f"{_format_token_count(estimate.get('actual_total_tokens'))} | "
                f"{_format_currency(estimate.get('estimated_cost_usd'))} | "
                f"{_format_currency(estimate.get('actual_cost_usd'))} |"
            )
        lines.append("")
        latest_estimate = estimates[-1]
        lines.append(
            f"_Estimator: `{_markdown_inline(latest_estimate.get('method', 'N/A'))}`, "
            f"{latest_estimate.get('chars_per_token', 'N/A')} chars/token, "
            f"{latest_estimate.get('safety_margin', 'N/A')} safety margin._"
        )
        lines.append("")

    lines.append("| Model | Prompt Tokens | Completion Tokens | Total Tokens | Cost |")
    lines.append("|---|---:|---:|---:|---:|")
    for call in usage.get("calls", []):
        lines.append(
            f"| {_markdown_inline(call['model'])} | {call['prompt_tokens']} | {call['completion_tokens']} | "
            f"{call['total_tokens']} | {_format_currency(call['cost_usd'])} |"
        )
    lines.append("")

    return "\n".join(lines)

def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = _build_report_filename(results.get("report_name"), timestamp=timestamp)
    filepath = os.path.join(OUTPUT_DIR, filename)

    formatted = format_results(results)

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(formatted)
        print(Fore.GREEN + f"\n[Raxi Court] Results saved to {filepath}")
    except IOError as e:
        print(Fore.RED + f"[ERROR] Failed to save results: {e}")
