import copy
import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone

from src.config import RESPONSE_CACHE_DIR, RESPONSE_CACHE_ENABLED, RESPONSE_CACHE_VERSION


_WRITE_LOCK = threading.Lock()


def _cache_key(namespace, request):
    payload = {
        "namespace": namespace,
        "request": request,
        "version": RESPONSE_CACHE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(namespace, request):
    key = _cache_key(namespace, request)
    return os.path.join(RESPONSE_CACHE_DIR, namespace, key[:2], f"{key}.json")


def _load_cache_payload(namespace, request):
    path = _cache_path(namespace, request)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (IOError, json.JSONDecodeError):
        return None

    if payload.get("version") != RESPONSE_CACHE_VERSION or payload.get("namespace") != namespace:
        return None

    return payload


def has_cached_result(namespace, request):
    if not RESPONSE_CACHE_ENABLED:
        return False
    return _load_cache_payload(namespace, request) is not None


def _normalise_cached_usage_entry(usage_entry):
    prompt_tokens = int(usage_entry.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage_entry.get("completion_tokens", 0) or 0)
    total_tokens = int(usage_entry.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost_usd = float(usage_entry.get("cost_usd", 0.0) or 0.0)

    return {
        "model": usage_entry.get("model", "unknown"),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cached": True,
        "cached_prompt_tokens": prompt_tokens,
        "cached_completion_tokens": completion_tokens,
        "cached_total_tokens": total_tokens,
        "cached_cost_usd": cost_usd,
    }


def get_cached_result(namespace, request):
    if not RESPONSE_CACHE_ENABLED:
        return None, []

    payload = _load_cache_payload(namespace, request)
    if payload is None:
        return None, []

    result = copy.deepcopy(payload.get("result"))
    usage_entries = [
        _normalise_cached_usage_entry(usage_entry)
        for usage_entry in payload.get("usage_entries", [])
        if isinstance(usage_entry, dict)
    ]
    return result, usage_entries


def store_cached_result(namespace, request, result, usage_entries=None):
    if not RESPONSE_CACHE_ENABLED:
        return

    path = _cache_path(namespace, request)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "version": RESPONSE_CACHE_VERSION,
        "namespace": namespace,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "usage_entries": usage_entries or [],
    }

    with _WRITE_LOCK:
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix="cache_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
