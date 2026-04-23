import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorama import Fore, init
from src.config import GENERATION_MODELS, EVALUATION_MODELS
from src.evaluator import run_council

init(autoreset=True)


def _prompt_int(label, default, minimum=None, maximum=None, error_label=None):
    raw_value = input(Fore.WHITE + label).strip()
    error_label = error_label or label.strip().rstrip(":")

    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        print(Fore.RED + f"  [!] Invalid {error_label}. Using default {default}.")
        return default

    below_minimum = minimum is not None and value < minimum
    above_maximum = maximum is not None and value > maximum
    if below_minimum or above_maximum:
        if minimum is not None and maximum is not None:
            print(Fore.RED + f"  [!] {error_label} must be between {minimum} and {maximum}. Using default {default}.")
        elif minimum is not None:
            print(Fore.RED + f"  [!] {error_label} must be at least {minimum}. Using default {default}.")
        else:
            print(Fore.RED + f"  [!] {error_label} must be at most {maximum}. Using default {default}.")
        return default

    return value


def _prompt_optional_positive_int(label, error_label):
    raw_value = input(Fore.WHITE + label).strip()
    if not raw_value:
        return None

    try:
        value = int(raw_value)
    except ValueError:
        print(Fore.RED + f"  [!] Invalid {error_label}. Using no budget cap.")
        return None

    if value < 1:
        print(Fore.RED + f"  [!] {error_label} must be at least 1. Using no budget cap.")
        return None

    return value


def get_user_config():
    print(Fore.YELLOW + r"""
 ____      _    __  __ ___    ____ ___  _   _ ____ _____
|  _ \    / \   \ \/ /|_ _|  / ___/ _ \| | | |  _ \_   _|
| |_) |  / _ \   \  /  | |  | |  | | | | | | | |_) || |
|  _ <  / ___ \  /  \  | |  | |__| |_| | |_| |  _ < | |
|_| \_\/_/   \_\/_/\_\|___|  \____\___/ \___/|_| \_\|_|

                 MULTI-LLM VERIFICATION
""")

    prompt = input(Fore.WHITE + "\n  Prompt : ").strip()

    if not prompt:
        print(Fore.RED + "[ERROR] Prompt cannot be empty.")
        return None

    report_name = input(Fore.WHITE + "  Report name (optional): ").strip() or None

    max_generation_models = len(GENERATION_MODELS)
    min_generation_models = 2 if max_generation_models >= 2 else max_generation_models
    max_evaluation_models = len(EVALUATION_MODELS)

    score_threshold = _prompt_int(
        "  Threshold (1-100, default 60): ",
        default=60,
        minimum=1,
        maximum=100,
        error_label="threshold",
    )
    max_retries = _prompt_int(
        "  Max retries (default 3): ",
        default=3,
        minimum=1,
        error_label="retries",
    )
    generation_count = _prompt_int(
        f"  Generation models k ({min_generation_models}-{max_generation_models}, default {max_generation_models}): ",
        default=max_generation_models,
        minimum=min_generation_models,
        maximum=max_generation_models,
        error_label="generation model count",
    )
    evaluation_count = _prompt_int(
        f"  Evaluation arbiters m (1-{max_evaluation_models}, default {max_evaluation_models}): ",
        default=max_evaluation_models,
        minimum=1,
        maximum=max_evaluation_models,
        error_label="evaluation arbiter count",
    )
    max_total_tokens = _prompt_optional_positive_int(
        "  Max token budget (blank = none) : ",
        error_label="token budget",
    )

    return {
        "prompt": prompt,
        "report_name": report_name,
        "score_threshold": score_threshold,
        "max_retries": max_retries,
        "generation_count": generation_count,
        "evaluation_count": evaluation_count,
        "max_total_tokens": max_total_tokens,
    }


def _score_bar(score, width=30):
    filled = round(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def display_results(results):
    safety_response = results.get("safety_response")
    if safety_response and safety_response.get("blocked"):
        print(Fore.RED + "\n  Safety block")
        print(Fore.WHITE + f"  Tag      : {safety_response.get('tag')}")
        print(Fore.WHITE + f"  Title    : {safety_response.get('title')}")
        print(Fore.WHITE + f"  Category : {safety_response.get('category')}")
        print(Fore.WHITE + f"  Reason   : {safety_response.get('reason')}")
        return

    score = results['final_score']
    hallucination = results['aggregation']['any_hallucination']
    score_color = Fore.GREEN if score >= 80 else Fore.YELLOW if score >= 60 else Fore.RED
    hall_label = Fore.RED + "YES  ⚠" if hallucination else Fore.GREEN + "NONE"

    print(Fore.YELLOW + r"""
 __     __  _____  ____   ____   ___   ____  _____
 \ \   / / | ____||  _ \ |  _ \ |_ _| / ___||_   _|
  \ \ / /  |  _|  | |_) || | | | | | | |      | |
   \ V /   | |___ |  _ < | |_| | | | | |___   | |
    \_/    |_____||_| \_\|____/ |___| \____|  |_|
""")

    print(score_color + f"\n  Score       {_score_bar(score)}  {score}/100")
    usage = results.get("usage", {})
    total_tokens = usage.get("total_tokens", "N/A")
    total_cost   = usage.get("total_cost_usd")
    token_budget = usage.get("max_total_tokens")
    cost_str     = f"${total_cost:.4f}" if total_cost is not None else "N/A"
    run_config = results.get("run_config", {})
    semantic_entropy = results.get("aggregation", {}).get("semantic_entropy")

    print(Fore.WHITE  + f"  Attempts    {results['attempt']}")
    print(Fore.WHITE  + f"  Elapsed     {results['elapsed_time']}s")
    print(Fore.WHITE  + f"  Hallucin.   {hall_label}")
    print(Fore.WHITE  + f"  Models      k={run_config.get('generation_count', 'N/A')}  m={run_config.get('evaluation_count', 'N/A')}")
    if semantic_entropy is not None:
        print(Fore.WHITE  + f"  Sem Ent.    {semantic_entropy.get('normalized_entropy', 'N/A')}")
        print(Fore.WHITE  + f"  Agreement   {semantic_entropy.get('agreement_score', 'N/A')}")
    print(Fore.WHITE  + f"  Tokens      {total_tokens}")
    attempt_estimates = [
        estimate
        for estimate in usage.get("estimates", [])
        if estimate.get("stage") == "attempt"
    ]
    if attempt_estimates:
        estimated_tokens = attempt_estimates[-1].get("estimated_total_tokens")
        print(Fore.WHITE  + f"  Est. run    {estimated_tokens}")
    if token_budget is not None:
        print(Fore.WHITE  + f"  Budget      {token_budget}")
    print(Fore.WHITE  + f"  Cost        {cost_str}")

    agg = results['aggregation']['raw_scores']
    print(Fore.YELLOW + "\n  ── Arbiter scores ──────────────────────────────────────")
    for persona, raw in agg.items():
        if raw is not None:
            bar = "█" * round(raw) + "░" * (10 - round(raw))
            print(Fore.CYAN + f"  {persona.upper():<12} {bar}  {raw}/10")

    print(Fore.YELLOW + r"""
 ____            _     ____                                     
| __ )  ___  ___| |_  |  _ \ ___  ___ _ __   ___  _ __  ___  ___
|  _ \ / _ \/ __| __| | |_) / _ \/ __| '_ \ / _ \| '_ \/ __|/ _ \
| |_) |  __/\__ \ |_  |  _ <  __/\__ \ |_) | (_) | | | \__ \  __/
|____/ \___||___/\__| |_| \_\___||___/ .__/ \___/|_| |_|___/\___|
                                     |_|
""")
    print(Fore.WHITE  + f"\n  ID     : {results['best_response']['id']}")
    print(Fore.WHITE  + f"  Model  : {results['best_response']['_original_model']}")
    print(Fore.WHITE  + f"\n{results['best_response']['response']}")


def main():
    config = get_user_config()

    if config is None:
        return

    results = run_council(
        user_prompt=config["prompt"],
        report_name=config["report_name"],
        score_threshold=config["score_threshold"],
        max_retries=config["max_retries"],
        generation_count=config["generation_count"],
        evaluation_count=config["evaluation_count"],
        max_total_tokens=config["max_total_tokens"],
    )

    if results is None:
        print(Fore.RED + "\n  No acceptable output produced.")
        print(Fore.RED + "  Try lowering the score threshold, increasing retries, or raising the token budget.")
        return

    display_results(results)


if __name__ == "__main__":
    main()
