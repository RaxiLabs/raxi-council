import tempfile
import unittest
from unittest.mock import patch

from src.agents import GENERATION_CACHE_NAMESPACE, build_generation_cache_request
from src.budget import estimate_generation_stage
from src.cache import store_cached_result
import src.cache as cache_module


class BudgetCacheTests(unittest.TestCase):
    def test_generation_estimates_drop_to_zero_for_cached_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(cache_module, "RESPONSE_CACHE_DIR", temp_dir), patch.object(
                cache_module,
                "RESPONSE_CACHE_ENABLED",
                True,
            ):
                model = "openai/gpt-4o-mini"
                user_prompt = "What causes inflation?"
                request = build_generation_cache_request(model, user_prompt)

                store_cached_result(
                    GENERATION_CACHE_NAMESPACE,
                    request,
                    "cached response",
                    [
                        {
                            "model": model,
                            "prompt_tokens": 50,
                            "completion_tokens": 75,
                            "total_tokens": 125,
                            "cost_usd": 0.0123,
                        }
                    ],
                )

                estimate = estimate_generation_stage(user_prompt, [model])

                self.assertEqual(estimate["estimated_total_tokens"], 0)
                self.assertEqual(estimate["estimated_cost_usd"], 0.0)
                self.assertEqual(estimate["cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
