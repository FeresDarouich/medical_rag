from __future__ import annotations

from pathlib import Path


def load_pdf(path: str | Path) -> str:
    """Load a PDF into plain text.

    Requires `pypdf`.
    """

    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF loading requires 'pypdf' (pip install pypdf)") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)
