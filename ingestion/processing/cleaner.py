from __future__ import annotations


def clean_text(text: str) -> str:
    # Minimal cleaner: collapse whitespace.
    return " ".join(text.split())
