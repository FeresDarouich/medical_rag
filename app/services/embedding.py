from __future__ import annotations

import hashlib


class SimpleHashEmbedding:
    """Deterministic, dependency-free embedding baseline.

    Not semantically strong, but good enough for template wiring + tests.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = -1.0 if (digest[4] % 2) else 1.0
            vec[idx] += sign
        return vec
