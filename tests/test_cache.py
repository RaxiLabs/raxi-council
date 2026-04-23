import tempfile
import unittest
from unittest.mock import patch

from src.cache import get_cached_result, store_cached_result
import src.cache as cache_module


class CacheTests(unittest.TestCase):
    def test_cached_results_are_replayed_as_saved_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(cache_module, "RESPONSE_CACHE_DIR", temp_dir), patch.object(
                cache_module,
                "RESPONSE_CACHE_ENABLED",
                True,
            ):
                request = {"model": "openai/gpt-4o-mini", "user_prompt": "hello"}
                result = {"answer": "cached"}
                usage_entries = [
                    {
                        "model": "openai/gpt-4o-mini",
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                        "cost_usd": 0.0042,
                    }
                ]

                store_cached_result("test_namespace", request, result, usage_entries)
                cached_result, cached_usage = get_cached_result("test_namespace", request)

                self.assertEqual(cached_result, result)
                self.assertEqual(
                    cached_usage,
                    [
                        {
                            "model": "openai/gpt-4o-mini",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cost_usd": 0.0,
                            "cached": True,
                            "cached_prompt_tokens": 12,
                            "cached_completion_tokens": 8,
                            "cached_total_tokens": 20,
                            "cached_cost_usd": 0.0042,
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
