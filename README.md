# Raxi Court

A multi-agent AI evaluation framework that generates responses from several LLMs, evaluates them through independent arbiters, and returns the highest-scoring response with full traceability, hallucination analysis, and semantic agreement metrics.

---

## How It Works

```text
                User Prompt
                    |
                    v
┌─────────────────────────────────────────┐
│  SAFETY CHECK                           │
│  Harmful/dangerous prompts are blocked  │
└───────────────────┬─────────────────────┘
                    |
                    v
┌─────────────────────────────────────────┐
│  GENERATION                             │
│  k LLMs answer the prompt in parallel   │
└───────────────────┬─────────────────────┘
                    |
                    v
┌─────────────────────────────────────────┐
│  ANONYMISATION                          │
│  Responses shuffled & labelled A/B/C    │
└───────────────────┬─────────────────────┘
                    |
                    v
┌─────────────────────────────────────────┐
│  EVALUATION                             │
│  m arbiter personas score each response │
│  - SCEPTIC · EXPERT · LOGICIAN          │
└───────────────────┬─────────────────────┘
                    |
                    v
┌─────────────────────────────────────────┐
│  AGGREGATION                            │
│  Weighted scores -> final score (0-100) │
│  Best response selected                 │
└───────────────────┬─────────────────────┘
                    |
                    v
┌─────────────────────────────────────────┐
│  SEMANTIC ENTROPY                       │
│  Measures agreement across candidates   │
└───────────────────┬─────────────────────┘
                    |
                    v
          Raxi Court Report
```

Responses are anonymised before evaluation so arbiters cannot identify the source model. If the final score falls below the configured threshold, the system retries with a fresh generation round. Successful runs are saved as Markdown reports for later review.

---

## The Three Arbiters

Each arbiter is a separate LLM instance operating under a distinct evaluation persona. They score independently; no arbiter sees another arbiter's scores.

| Arbiter | Role | Focus | Weight |
|---|---|---|---|
| **ARBITER-1 SCEPTIC** | Rigorous fact-checker | Verifiability, unsupported claims, epistemic overconfidence | 50% |
| **ARBITER-2 EXPERT** | Domain depth evaluator | Technical accuracy, correct use of terminology, precision | 25% |
| **ARBITER-3 LOGICIAN** | Reasoning auditor | Logical structure, internal consistency, argument validity | 25% |

Each arbiter scores on three dimensions:

| Dimension | Weight |
|---|---|
| Factual Accuracy | 50% |
| Completeness | 25% |
| Reasoning Quality | 25% |

---

## Models

**Generation** — answers the user prompt in parallel:

- `openai/gpt-4o-mini`
- `anthropic/claude-3-haiku`
- `google/gemini-2.5-flash-lite`

**Evaluation** — one model per arbiter:

- SCEPTIC -> `openai/gpt-5.4-nano`
- EXPERT -> `anthropic/claude-sonnet-4-5`
- LOGICIAN -> `mistralai/mistral-large-2411`

All models are called through OpenRouter. You can swap any model in `src/config.py`.

---

## Setup

**Requirements:** Python 3.12+, an OpenRouter API key.

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Usage

```bash
python3 main.py
```

You'll be prompted for the run configuration:

```text
  Prompt : What causes inflation?
  Report name (optional): inflation_test
  Threshold (1-100, default 60): 75
  Max retries (default 3): 3
  Generation models k (2-3, default 3): 3
  Evaluation arbiters m (1-3, default 3): 3
  Max token budget (blank = none) :
```

| Input | Description | Default |
|---|---|---|
| Prompt | The question or task | required |
| Report name | Optional filename for the Markdown report | auto timestamp |
| Threshold | Minimum score (0-100) to accept a result | 60 |
| Max retries | How many generation rounds before giving up | 3 |
| k | Number of generation models to use | 3 |
| m | Number of evaluator arbiters to use | 3 |
| Max token budget | Optional token cap for the whole run | none |

### Example Output

```text
  Score       ████████████████████████████░░  94/100
  Attempts    1
  Elapsed     10.95s
  Hallucin.   NONE
  Models      k=3  m=3
  Sem Ent.    0.0
  Agreement   1.0
  Tokens      14290
  Cost        $0.0283

  ── Arbiter scores ──────────────────────────────────────
  SCEPTIC      █████████░  9.5/10
  EXPERT       ████████░░  8.75/10
  LOGICIAN     ██████████  10.0/10

  ID     : Response_A
  Model  : google/gemini-2.5-flash-lite

  [response text]
```

---

## Configuration

All tunable settings live in `src/config.py`.

```python
GENERATION_MODELS = [...]       # LLMs that generate responses
EVALUATION_MODELS = [...]       # LLMs assigned to each arbiter

ARBITER_WEIGHTS = {
    "sceptic":  0.5,
    "expert":   0.25,
    "logician": 0.25,
}

DIMENSION_WEIGHTS = {
    "factual_accuracy":  0.5,
    "completeness":      0.25,
    "reasoning_quality": 0.25,
}

HALLUCINATION_POLICY = "any"
HALLUCINATION_WEIGHT_THRESHOLD = 0.5

GENERATION_MAX_TOKENS = 1000
EVALUATION_MAX_TOKENS = 1800
SEMANTIC_ENTROPY_MAX_TOKENS = 500
REQUEST_TIMEOUT = 30
MIN_VALID_ARBITERS = 2
OUTPUT_DIR = "./results"
```

Arbiter personas are plain-text system prompts in `prompts/`. Edit them to change how each arbiter evaluates.

---

## Safety Handling

Before generation starts, the prompt is checked for harmful or dangerous instructional requests. If the prompt is blocked, the council does not call generation or evaluation models. Instead, it returns and saves a structured safety response.

```json
{
  "tag": "harmful_or_dangerous_prompt",
  "title": "Weapons or explosives request",
  "category": "weapons_or_explosives",
  "blocked": true,
  "reason": "The prompt appears to request harmful or dangerous assistance, so the council run was not started."
}
```

Blocked prompts still create a Markdown report in `results/` so the run is auditable.

---

## Output Reports

Every successful run saves a full report to `results/`. If no report name is supplied, the filename uses `raxi_court_output_TIMESTAMP.md`. If a custom report name is supplied, that sanitized name is used directly.

Reports contain:

- Run metadata
- The original prompt
- The best response with source model revealed
- All generated responses
- Full evaluation from each arbiter
- Aggregation summary
- Hallucination flags and details
- Semantic entropy and agreement metrics
- API usage, token counts, and estimated cost

## Project Structure

```text
raxi-council/
├── main.py              # Entry point: input, orchestration, display
├── src/
│   ├── config.py        # Models, weights, paths, constants
│   ├── agents.py        # Generation and evaluation API calls
│   ├── evaluator.py     # Pipeline orchestration
│   ├── aggregator.py    # Scoring and best-response selection
│   ├── semantic_entropy.py
│   ├── safety.py        # Harmful prompt classification
│   └── output.py        # Markdown report generation
├── prompts/
│   ├── sceptic.txt
│   ├── expert.txt
│   ├── logician.txt
│   └── semantic_entropy.txt
├── results/             # Saved evaluation reports
├── .env.example
└── requirements.txt
```

---

## Semantic Entropy

The system performs semantic clustering across candidate responses after evaluation. This produces:

- `semantic_entropy`: normalized disagreement across candidate meanings
- `agreement_score`: inverse agreement-style score
- `semantic_cluster_count`: number of semantic clusters
- `dominant_cluster_probability`: how much response-score mass belongs to the largest cluster

These fields help measure whether increasing `k` produces genuinely different answers or repeated versions of the same answer.
