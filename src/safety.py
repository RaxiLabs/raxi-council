import re


HARMFUL_PROMPT_TAG = "harmful_or_dangerous_prompt"


_INSTRUCTIONAL_INTENT = re.compile(
    r"\b("
    r"how\s+to|steps?\s+to|instructions?\s+for|guide\s+to|tutorial\s+for|"
    r"build|make|create|write|generate|bypass|evade|exploit|hack|weaponize"
    r")\b",
    re.IGNORECASE,
)

_HARM_CATEGORIES = [
    (
        "weapons_or_explosives",
        "Weapons or explosives request",
        re.compile(
            r"\b("
            r"bomb|explosive|grenade|pipe\s+bomb|molotov|detonator|firearm|"
            r"ghost\s+gun|silencer|poison\s+gas"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cyber_abuse",
        "Cyber abuse request",
        re.compile(
            r"\b("
            r"malware|ransomware|keylogger|phishing|credential\s+steal|"
            r"steal\s+password|ddos|botnet|reverse\s+shell|privilege\s+escalation"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm",
        "Self-harm request",
        re.compile(
            r"\b("
            r"kill\s+myself|suicide|self\s*harm|overdose|hang\s+myself|"
            r"cut\s+myself"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "violent_harm",
        "Violent harm request",
        re.compile(
            r"\b("
            r"kill\s+someone|hurt\s+someone|assassinate|torture|kidnap|"
            r"hide\s+a\s+body"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fraud_or_evasion",
        "Fraud or evasion request",
        re.compile(
            r"\b("
            r"fake\s+id|forge|counterfeit|launder\s+money|tax\s+evasion|"
            r"evade\s+police|avoid\s+detection"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "illegal_drug_production",
        "Illegal drug production request",
        re.compile(
            r"\b("
            r"cook\s+meth|make\s+meth|synthesize\s+lsd|produce\s+fentanyl|"
            r"make\s+illegal\s+drugs"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_prompt_safety(prompt):
    """Return a structured safety response when the prompt asks for harmful instructions."""
    prompt = str(prompt or "").strip()
    has_instructional_intent = bool(_INSTRUCTIONAL_INTENT.search(prompt))

    for category, title, pattern in _HARM_CATEGORIES:
        if not pattern.search(prompt):
            continue

        if category in {"self_harm", "violent_harm"} or has_instructional_intent:
            return {
                "tag": HARMFUL_PROMPT_TAG,
                "title": title,
                "category": category,
                "blocked": True,
                "reason": "The prompt appears to request harmful or dangerous assistance, so the council run was not started.",
            }

    return {
        "tag": "safe_prompt",
        "title": "Prompt allowed",
        "category": None,
        "blocked": False,
        "reason": None,
    }
