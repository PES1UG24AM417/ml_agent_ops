import re


INJECTION_PATTERNS = (
    re.compile(r"\b(ignore|disregard|override)\b.{0,50}\b(instruction|prompt|rule)s?\b", re.I),
    re.compile(r"\b(system prompt|developer message|reveal secrets|tool call)\b", re.I),
)
ABUSIVE_TERMS = {"kill", "rape", "hate"}


def inspect_untrusted_text(text: str) -> list[str]:
    errors = []
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        errors.append("possible prompt injection detected")
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & ABUSIVE_TERMS:
        errors.append("unsafe abusive content detected")
    return errors


def filter_recommendation(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("claim is approved", "claim is rejected", "claim denied")):
        raise ValueError("agent output contains an unauthorized claim determination")
    return text
