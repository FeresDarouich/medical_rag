from __future__ import annotations

import re


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact_phi(text: str) -> str:
    """Best-effort redaction for obvious identifiers (placeholder).

    This is intentionally conservative and lightweight; replace with a proper
    de-identification pipeline for production use.
    """

    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text
