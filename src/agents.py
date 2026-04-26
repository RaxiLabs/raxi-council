import json
import hashlib
import time
import requests
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, init
from src.cache import get_cached_result, store_cached_result
from src.config import (
    PROMPT_PATHS,
    OPENROUTER_ENDPOINT,
    HEADERS,
    GENERATION_MAX_TOKENS,
    EVALUATION_MAX_TOKENS,
    REQUEST_TIMEOUT,
    MAX_API_RETRIES,
    EVALUATION_PARSE_RETRIES,
    RETRY_BACKOFF_SECONDS,
    MAX_RETRY_BACKOFF_SECONDS,
    GENERATION_MODELS,
    GENERATION_SYSTEM_PROMPT,
    EVALUATION_PERSONAS,
    EVALUATION_MODELS,
    MODEL_PRICING,
)

init(autoreset=True)

_thread_local = threading.local()
GENERATION_CACHE_NAMESPACE = "generation_response"
EVALUATION_CACHE_NAMESPACE = "evaluation_payload"


def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _print_retry_warning(message):
    print(Fore.YELLOW + f"[WARN] {message}")


def _print_retryable_error(kind, model, error_message):
    print(Fore.RED + f"[ERROR] {kind} for {model}: {error_message}")


def _backoff_wait_time(attempt, retry_after=None):
    if retry_after is not None:
        return retry_after
    return min(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_BACKOFF_SECONDS)


def _normalise_issue_field(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError
    return value.strip() or None


@lru_cache(maxsize=None)
def load_prompt(persona):
    try:
        with open(PROMPT_PATHS[persona], "r") as f:
            return f.read()
    except FileNotFoundError:
        print(Fore.RED + f"[ERROR] Prompt file not found: {PROMPT_PATHS[persona]}")
        raise
    except KeyError:
        print(Fore.RED + f"[ERROR] Unknown persona: {persona}")
        raise


def _calculate_call_cost(model, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000


def _parse_retry_after(header_value):
    if not header_value:
        return None

    try:
        retry_after = float(header_value)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(retry_after, MAX_RETRY_BACKOFF_SECONDS))


def _extract_error_message(response):
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else "No error body returned."

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return message
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message

    return "No structured error message returned."


def _extract_content(data):
    if not isinstance(data, dict):
        raise ValueError("Response payload was not a JSON object.")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response did not include any choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Response choice did not include a message object.")

    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content:
            return content
        raise ValueError("Response content was empty.")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    text_parts.append(text)
        joined = "\n".join(text_parts).strip()
        if joined:
            return joined

    raise ValueError("Response content was missing or not text.")


def _extract_json_object(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "arbiter_id" in parsed and "evaluations" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    fallback_candidate = None
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(parsed, dict) and "arbiter_id" in parsed and "evaluations" in parsed:
                return parsed
            if fallback_candidate is None and isinstance(parsed, dict):
                fallback_candidate = parsed
        except json.JSONDecodeError:
            continue

    if fallback_candidate is not None:
        return fallback_candidate

    raise json.JSONDecodeError("No valid JSON object found.", cleaned, 0)


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
                "issues": []
            }, 
            "evaluation_notes": None,
        }
        for response_id in response_ids
    }
    return json.dumps({"arbiter_id": "ARBITER-ID", "evaluations": evaluations}, indent=2)


def format_anonymised_responses(anonymised_responses):
    return "\n\n".join(
        f"{item['id']}:\n{item['response']}"
        for item in anonymised_responses
    )


def build_evaluation_prompt(user_prompt, anonymised_responses):
    formatted_responses = format_anonymised_responses(anonymised_responses)
    return (
        f"Original question: {user_prompt}\n\n"
        f"Evaluate each of the following anonymous responses:\n\n"
        f"{formatted_responses}"
    )


def build_initial_evaluation_prompt(user_prompt, anonymised_responses):
    response_ids = [item["id"] for item in anonymised_responses]
    return (
        build_evaluation_prompt(user_prompt, anonymised_responses)
        + "\n\nReturn JSON in this exact top-level shape:\n"
        + _build_response_schema_hint(response_ids)
    )


def build_generation_cache_request(model, user_prompt, max_tokens=GENERATION_MAX_TOKENS):
    return {
        "max_tokens": max_tokens,
        "model": model,
        "system_prompt": GENERATION_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }


def build_evaluation_cache_request(
    user_prompt,
    anonymised_responses,
    persona,
    model,
    max_tokens=EVALUATION_MAX_TOKENS,
    evaluation_prompt=None,
    response_ids=None,
    system_prompt=None,
):
    response_ids = response_ids or [item["id"] for item in anonymised_responses]
    evaluation_prompt = evaluation_prompt or build_evaluation_prompt(user_prompt, anonymised_responses)
    system_prompt = system_prompt or load_prompt(persona)

    return {
        "max_tokens": max_tokens,
        "model": model,
        "persona": persona,
        "response_ids": response_ids,
        "system_prompt": system_prompt,
        "user_prompt": evaluation_prompt,
    }


def _validate_dimension_score(persona, dimension, value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{dimension} score must be numeric, got {type(value).__name__}.")
    if value < 0 or value > 10:
        raise ValueError(f"{dimension} score must be between 0 and 10, got {value}.")


def _build_hallucination_detail_from_assessment(assessment):
    if not isinstance(assessment, dict):
        return None

    issues = assessment.get("issues") or []
    if not issues:
        return None

    first_issue = issues[0]
    claim = first_issue.get("claim") or first_issue.get("span_text") or "Unspecified claim"
    reason = first_issue.get("reason")
    if reason:
        return f"Claim: '{claim}'. {reason}"
    return f"Claim: '{claim}'."


def _validate_hallucination_assessment(response_id, hallucination_flag, assessment):
    if assessment is None:
        return None

    if not isinstance(assessment, dict):
        raise ValueError(f"{response_id} hallucination_assessment must be an object.")

    flag = assessment.get("flag")
    if not isinstance(flag, bool):
        raise ValueError(f"{response_id} hallucination_assessment.flag must be a boolean.")

    if flag != hallucination_flag:
        raise ValueError(f"{response_id} hallucination_assessment.flag must match hallucination_flag.")

    severity = assessment.get("severity")
    allowed_severities = {None, "minor", "moderate", "major", "critical"}
    if severity not in allowed_severities:
        raise ValueError(
            f"{response_id} hallucination_assessment.severity must be one of "
            f"{', '.join(str(item) for item in sorted(value for value in allowed_severities if value is not None))}, or null."
        )

    issues = assessment.get("issues")
    if not isinstance(issues, list):
        raise ValueError(f"{response_id} hallucination_assessment.issues must be a list.")

    if hallucination_flag and not issues:
        raise ValueError(f"{response_id} hallucination_assessment.issues must contain at least one issue when hallucination_flag is true.")

    validated_issues = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError(f"{response_id} hallucination issue entries must be objects.")

        try:
            normalized_issue = {
                field: _normalise_issue_field(issue.get(field))
                for field in ("claim", "reason", "span_text", "category")
            }
        except TypeError:
            raise ValueError(f"{response_id} hallucination issue fields must be strings or null.") from None

        if hallucination_flag and not any(
            isinstance(normalized_issue.get(field), str) and normalized_issue[field].strip()
            for field in ("claim", "span_text")
        ):
            raise ValueError(f"{response_id} hallucination issue must include a non-empty claim or span_text.")

        validated_issues.append(normalized_issue)

    return {
        "flag": flag,
        "severity": severity,
        "issues": validated_issues,
    }


def _validate_single_response_evaluation(response_id, evaluation):
    if not isinstance(evaluation, dict):
        raise ValueError(f"{response_id} evaluation was not a JSON object.")

    dimensions = evaluation.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError(f"{response_id} is missing a valid 'dimensions' object.")

    expected_dimensions = {"factual_accuracy", "completeness", "reasoning_quality"}
    missing_dimensions = [name for name in expected_dimensions if name not in dimensions]
    if missing_dimensions:
        raise ValueError(f"{response_id} is missing dimension scores for: {', '.join(missing_dimensions)}.")

    for dimension in expected_dimensions:
        entry = dimensions.get(dimension)
        if not isinstance(entry, dict):
            raise ValueError(f"{response_id} {dimension} entry must be an object.")
        _validate_dimension_score(response_id, dimension, entry.get("score"))

    total_score = evaluation.get("total_score")
    if isinstance(total_score, (int, float)) and (total_score < 0 or total_score > 30):
        raise ValueError(f"{response_id} total_score must be between 0 and 30, got {total_score}.")

    hallucination_flag = evaluation.get("hallucination_flag")
    if hallucination_flag is not None and not isinstance(hallucination_flag, bool):
        raise ValueError(f"{response_id} hallucination_flag must be a boolean.")

    hallucination_assessment = _validate_hallucination_assessment(
        response_id,
        hallucination_flag,
        evaluation.get("hallucination_assessment"),
    )

    hallucination_detail = evaluation.get("hallucination_detail")
    if hallucination_detail is not None and not isinstance(hallucination_detail, str):
        raise ValueError(f"{response_id} hallucination_detail must be a string or null.")

    if hallucination_assessment is not None:
        evaluation["hallucination_assessment"] = hallucination_assessment
        if not hallucination_detail:
            evaluation["hallucination_detail"] = _build_hallucination_detail_from_assessment(hallucination_assessment)
    else:
        evaluation["hallucination_assessment"] = None

    return evaluation


def _validate_evaluation_payload(persona, evaluation, expected_response_ids):
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation payload was not a JSON object.")

    arbiter_id = evaluation.get("arbiter_id")
    if not isinstance(arbiter_id, str) or not arbiter_id.strip():
        raise ValueError("Evaluation payload is missing a valid 'arbiter_id'.")

    response_evaluations = evaluation.get("evaluations")
    if not isinstance(response_evaluations, dict):
        raise ValueError("Evaluation payload is missing a valid 'evaluations' object.")

    missing_responses = [response_id for response_id in expected_response_ids if response_id not in response_evaluations]
    if missing_responses:
        raise ValueError(f"Missing evaluations for: {', '.join(missing_responses)}.")

    extra_responses = [response_id for response_id in response_evaluations if response_id not in expected_response_ids]
    if extra_responses:
        raise ValueError(f"Unexpected response ids in evaluation: {', '.join(extra_responses)}.")

    validated_evaluations = {}
    for response_id in expected_response_ids:
        validated_evaluations[response_id] = _validate_single_response_evaluation(
            response_id,
            response_evaluations[response_id]
        )

    evaluation["evaluations"] = validated_evaluations

    return evaluation


def call_llm(model, system_prompt, user_prompt, max_tokens):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    }

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            res = _get_session().post(
                OPENROUTER_ENDPOINT,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            if res.status_code == 429:
                error_message = _extract_error_message(res)
                if attempt == MAX_API_RETRIES:
                    _print_retryable_error("Rate limited by API", model, error_message)
                    return None, None

                retry_after = _parse_retry_after(res.headers.get("Retry-After"))
                wait_time = _backoff_wait_time(attempt, retry_after)
                _print_retry_warning(
                    f"Rate limited by API for {model}. Retrying in {wait_time:.1f}s "
                    f"(attempt {attempt}/{MAX_API_RETRIES})."
                )
                time.sleep(wait_time)
                continue

            if 500 <= res.status_code < 600:
                error_message = _extract_error_message(res)
                if attempt == MAX_API_RETRIES:
                    _print_retryable_error("Upstream server error", model, error_message)
                    return None, None

                wait_time = _backoff_wait_time(attempt)
                _print_retry_warning(
                    f"Server error {res.status_code} for {model}. Retrying in {wait_time:.1f}s "
                    f"(attempt {attempt}/{MAX_API_RETRIES})."
                )
                time.sleep(wait_time)
                continue

            if 400 <= res.status_code < 500:
                error_message = _extract_error_message(res)
                print(Fore.RED + f"[ERROR] Request rejected for {model} ({res.status_code}): {error_message}")
                return None, None

            res.raise_for_status()

            try:
                data = res.json()
            except ValueError:
                print(Fore.RED + f"[ERROR] Invalid JSON returned by API for {model}.")
                return None, None

            try:
                content = _extract_content(data)
            except ValueError as e:
                print(Fore.RED + f"[ERROR] Malformed API response for {model}: {e}")
                return None, None

            raw_usage = data.get("usage", {})
            prompt_tokens = raw_usage.get("prompt_tokens", 0)
            completion_tokens = raw_usage.get("completion_tokens", 0)

            usage = {
                "model":              model,
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "total_tokens":       prompt_tokens + completion_tokens,
                "cost_usd":           _calculate_call_cost(model, prompt_tokens, completion_tokens),
            }
            return content, usage

        except requests.exceptions.Timeout:
            if attempt == MAX_API_RETRIES:
                print(Fore.RED + f"[ERROR] Timeout after {MAX_API_RETRIES} attempts: {model}")
                return None, None

            wait_time = _backoff_wait_time(attempt)
            _print_retry_warning(
                f"Timeout calling {model}. Retrying in {wait_time:.1f}s "
                f"(attempt {attempt}/{MAX_API_RETRIES})."
            )
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:
            if attempt == MAX_API_RETRIES:
                print(Fore.RED + f"[ERROR] Connection error for {model}: {e}")
                return None, None

            wait_time = _backoff_wait_time(attempt)
            _print_retry_warning(
                f"Connection issue calling {model}. Retrying in {wait_time:.1f}s "
                f"(attempt {attempt}/{MAX_API_RETRIES})."
            )
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            print(Fore.RED + f"[ERROR] Request error for {model}: {e}")
            return None, None

    return None, None


def generate_responses(user_prompt, generation_models=None):
    print(Fore.YELLOW + "\n  ── Generation " + "─" * 46)

    responses = []
    usage_log = []
    selected_generation_models = generation_models or GENERATION_MODELS

    with ThreadPoolExecutor(max_workers=len(selected_generation_models)) as executor:
        futures = {}

        for model in selected_generation_models:
            cache_request = build_generation_cache_request(model, user_prompt, GENERATION_MAX_TOKENS)
            cached_response, cached_usage = get_cached_result(GENERATION_CACHE_NAMESPACE, cache_request)

            if cached_response is not None:
                responses.append({"model": model, "response": cached_response})
                usage_log.extend(cached_usage)
                print(Fore.CYAN + f"    ↺  {model} (cache)")
                continue

            futures[
                executor.submit(call_llm, model, GENERATION_SYSTEM_PROMPT, user_prompt, GENERATION_MAX_TOKENS)
            ] = (model, cache_request)

        for future in as_completed(futures):
            model, cache_request = futures[future]
            try:
                response, usage = future.result()
                if response:
                    responses.append({"model": model, "response": response})
                    if usage:
                        usage_log.append(usage)
                    store_cached_result(
                        GENERATION_CACHE_NAMESPACE,
                        cache_request,
                        response,
                        [usage] if usage else [],
                    )
                    print(Fore.GREEN + f"    ✓  {model}")
                else:
                    print(Fore.RED + f"    ✗  {model}")
            except Exception as e:
                print(Fore.RED + f"  [ERROR] {model}: {e}")

    return responses, usage_log


def _anonymised_sort_key(user_prompt, item):
    payload = f"{user_prompt}\n{item['model']}\n{item['response']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def anonymise_and_shuffle(responses, user_prompt=""):
    shuffled = sorted(responses, key=lambda item: _anonymised_sort_key(user_prompt, item))

    anonymised = []

    for i, item in enumerate(shuffled):
        anonymised.append({
            "id": f"Response_{chr(65 + i)}",
            "response": item["response"],
            "_original_model": item["model"]
        })

    print(Fore.YELLOW + "  Responses anonymised and ordered.")

    return anonymised


def evaluate_responses(
    user_prompt,
    anonymised_responses,
    evaluation_personas=None,
    evaluation_models=None,
    min_valid_arbiters=1,
):
    print(Fore.YELLOW + "\n  ── Evaluation " + "─" * 46)

    evaluation_prompt = build_evaluation_prompt(user_prompt, anonymised_responses)

    personas = evaluation_personas or EVALUATION_PERSONAS
    models = evaluation_models or EVALUATION_MODELS

    if len(personas) != len(models):
        raise ValueError("Evaluation personas and models must have the same length.")

    expected_response_ids = [item["id"] for item in anonymised_responses]
    evaluations = {}
    usage_log = []
    schema_hint = _build_response_schema_hint(expected_response_ids)
    persona_prompts = {persona: load_prompt(persona) for persona in personas}
    cache_requests = {
        persona: build_evaluation_cache_request(
            user_prompt,
            anonymised_responses,
            persona,
            models[index],
            EVALUATION_MAX_TOKENS,
            evaluation_prompt=evaluation_prompt,
            response_ids=expected_response_ids,
            system_prompt=persona_prompts[persona],
        )
        for index, persona in enumerate(personas)
    }

    def evaluate_persona(persona, model, cache_request):
        attempt_prompt = (
            evaluation_prompt
            + "\n\nReturn JSON in this exact top-level shape:\n"
            + schema_hint
        )
        usage_entries = []

        for attempt in range(1, EVALUATION_PARSE_RETRIES + 2):
            raw, usage = call_llm(
                model=model,
                system_prompt=persona_prompts[persona],
                user_prompt=attempt_prompt,
                max_tokens=EVALUATION_MAX_TOKENS
            )

            if usage:
                usage_entries.append(usage)

            if not raw:
                print(Fore.RED + f"  [{persona.upper()}] No response received.")
                return persona, None, usage_entries

            try:
                parsed = _extract_json_object(raw)
                validated = _validate_evaluation_payload(persona, parsed, expected_response_ids)
                store_cached_result(
                    EVALUATION_CACHE_NAMESPACE,
                    cache_request,
                    validated,
                    usage_entries,
                )
                return persona, validated, usage_entries
            except json.JSONDecodeError:
                preview = raw.strip().replace("\n", " ")[:200]
                if attempt > EVALUATION_PARSE_RETRIES:
                    print(Fore.RED + f"  [{persona.upper()}] Failed to parse JSON after {attempt} attempt(s).")
                    print(Fore.RED + f"  [{persona.upper()}] Raw preview: {preview}")
                    return persona, None, usage_entries
                print(Fore.YELLOW + f"  [{persona.upper()}] Invalid JSON format. Retrying ({attempt}/{EVALUATION_PARSE_RETRIES + 1})...")
                attempt_prompt = (
                    evaluation_prompt
                    + "\n\nReturn JSON in this exact top-level shape:\n"
                    + schema_hint
                    + "\n\nYour previous reply was not valid raw JSON. "
                    + "Return exactly one JSON object only, with no markdown fences and no explanatory text."
                )
            except ValueError as e:
                preview = raw.strip().replace("\n", " ")[:200]
                if attempt > EVALUATION_PARSE_RETRIES:
                    print(Fore.RED + f"  [{persona.upper()}] Invalid evaluation payload after {attempt} attempt(s): {e}")
                    print(Fore.RED + f"  [{persona.upper()}] Raw preview: {preview}")
                    return persona, None, usage_entries
                print(Fore.YELLOW + f"  [{persona.upper()}] Invalid payload shape. Retrying ({attempt}/{EVALUATION_PARSE_RETRIES + 1})...")
                attempt_prompt = (
                    evaluation_prompt
                    + "\n\nReturn JSON in this exact top-level shape:\n"
                    + schema_hint
                    + "\n\nYour previous reply used the wrong schema. "
                    + f"You must include arbiter_id and an evaluations object containing exactly these ids: {', '.join(expected_response_ids)}."
                )

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {}

        for i, persona in enumerate(personas):
            model = models[i]
            cache_request = cache_requests[persona]
            cached_result, cached_usage = get_cached_result(EVALUATION_CACHE_NAMESPACE, cache_request)

            if cached_result is not None:
                evaluations[persona] = cached_result
                usage_log.extend(cached_usage)
                print(Fore.CYAN + f"    ↺  {persona.upper()} (cache)")
                continue

            futures[executor.submit(evaluate_persona, persona, model, cache_request)] = persona

        for future in as_completed(futures):
            try:
                persona, result, usages = future.result()
                evaluations[persona] = result
                usage_log.extend(usages)
                if result:
                    print(Fore.GREEN + f"    ✓  {persona.upper()}")
            except Exception as e:
                persona = futures[future]
                print(Fore.RED + f"  [{persona.upper()}] Error: {e}")
                evaluations[persona] = None

    valid_arbiter_count = sum(1 for value in evaluations.values() if value is not None)
    if valid_arbiter_count < min_valid_arbiters:
        print(Fore.RED + f"  ✗ Only {valid_arbiter_count} valid arbiter(s); at least {min_valid_arbiters} required.")

    return evaluations, usage_log
