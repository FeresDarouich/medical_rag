from __future__ import annotations


def load_url(url: str) -> str:
    """Fetch a URL and return visible text.

    Requires `requests` and `beautifulsoup4`.
    """

    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Web loading requires 'requests' and 'beautifulsoup4'") from exc

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return " ".join(soup.get_text(" ").split())
