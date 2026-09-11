"""Provider-neutral failure classification and safe diagnostic redaction."""
from __future__ import annotations

import re

from config.fazle_ai.contracts import FailureClass

_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+"),
)


def safe_diagnostic(text: str, limit: int = 500) -> str:
    value = (text or "").replace("\x00", " ")
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value[-limit:]


def classify_provider_failure(text: str) -> FailureClass:
    value = (text or "").lower()
    if any(token in value for token in ("401", "invalid api key", "authentication failed")):
        return FailureClass.AUTHENTICATION_FAILURE
    if any(token in value for token in ("403", "unauthorized", "permission denied")):
        return FailureClass.AUTHORIZATION_FAILURE
    if any(token in value for token in ("insufficient_quota", "quota exhausted", "credit", "billing")):
        return FailureClass.QUOTA_CREDIT_EXHAUSTED
    if "429" in value or "rate limit" in value or "too many requests" in value:
        return FailureClass.RATE_LIMITED
    if any(token in value for token in ("model_not_found", "model not found", "provider unavailable", "404")):
        return FailureClass.PROVIDER_OR_MODEL_UNAVAILABLE
    if re.search(r"\b5\d\d\b", value) or any(token in value for token in ("upstream unavailable", "provider outage")):
        return FailureClass.PROVIDER_OUTAGE_OR_5XX
    if any(token in value for token in ("context length", "context_length", "maximum context", "too many tokens")):
        return FailureClass.CONTEXT_LENGTH_EXCEEDED
    if any(token in value for token in ("invalid json", "malformed response", "failed to parse")):
        return FailureClass.MALFORMED_PROVIDER_RESPONSE
    if any(token in value for token in ("safety refusal", "content safety", "content policy", "refused")):
        return FailureClass.SAFETY_OR_PROVIDER_REFUSAL
    if any(token in value for token in ("invalid request", "policy rejection")):
        return FailureClass.INVALID_REQUEST_OR_POLICY_REJECTION
    if any(token in value for token in ("timed out", "timeout")):
        return FailureClass.TIMEOUT
    return FailureClass.UNKNOWN_UNCLASSIFIED


def retry_after_seconds(text: str) -> float | None:
    match = re.search(r"(?i)retry-after\s*[:=]\s*(\d+(?:\.\d+)?)", text or "")
    return float(match.group(1)) if match else None
