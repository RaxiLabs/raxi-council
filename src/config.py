import os
import re
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please add it to your .env file.")

if not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
    raise ValueError("OPENROUTER_API_KEY is invalid. Must start with 'sk-or-v1-'.")

if not re.match(r'^sk-or-v1-[a-f0-9]{64}$', OPENROUTER_API_KEY):
    raise ValueError("OPENROUTER_API_KEY is invalid. Incorrect format.")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Raxi Court"
}

GENERATION_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemini-2.5-flash-lite"
]

EVALUATION_MODELS = [
    "openai/gpt-5.4-nano",
    "anthropic/claude-sonnet-4-5",
    "mistralai/mistral-large-2411"
]

EVALUATION_PERSONAS = [
    "sceptic",
    "expert",
    "logician"
]

PROMPT_PATHS = {
    "sceptic": "prompts/sceptic.txt",
    "expert":  "prompts/expert.txt",
    "logician": "prompts/logician.txt",
    "semantic_entropy": "prompts/semantic_entropy.txt",
}

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful and accurate assistant. Answer the following question "
    "as accurately and completely as possible."
)

ARBITER_WEIGHTS = {
    "sceptic":  0.5,
    "expert":   0.25,
    "logician": 0.25
}

HALLUCINATION_POLICY = "any"
HALLUCINATION_WEIGHT_THRESHOLD = 0.5

DIMENSION_WEIGHTS = {
    "factual_accuracy":  0.5,
    "completeness":      0.25,
    "reasoning_quality": 0.25
}

GENERATION_MAX_TOKENS = 1000
EVALUATION_MAX_TOKENS = 1800
SEMANTIC_ENTROPY_MAX_TOKENS = 500
REQUEST_TIMEOUT = 30
MAX_API_RETRIES = 3
EVALUATION_PARSE_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
MAX_RETRY_BACKOFF_SECONDS = 8
MIN_VALID_ARBITERS = 2
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEMANTIC_ENTROPY_MODEL = "openai/gpt-5.4-nano"
SEMANTIC_ENTROPY_WARNING_THRESHOLD = 0.6
TOKEN_ESTIMATION_CHARS_PER_TOKEN = 4
TOKEN_ESTIMATION_MESSAGE_OVERHEAD = 12
TOKEN_ESTIMATION_SAFETY_MARGIN = 1.15

MODEL_PRICING = {
    "openai/gpt-4o-mini":           {"input": 0.15,  "output": 0.60},
    "anthropic/claude-3-haiku":     {"input": 0.25,  "output": 1.25},
    "google/gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "openai/gpt-5.4-nano":          {"input": 0.15,  "output": 0.60},
    "anthropic/claude-sonnet-4-5":  {"input": 3.00,  "output": 15.00},
    "mistralai/mistral-large-2411": {"input": 2.00,  "output": 6.00},
}
