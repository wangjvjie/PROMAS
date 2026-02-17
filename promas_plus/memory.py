from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from .models import MemoryEntry

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/\.]+")


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


@dataclass
class HashEmbeddingProvider:
    dim: int = 384

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class SharedMemoryPool:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._items: list[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, entry_id: str, text: str, metadata: dict | None = None) -> None:
        embedding = self.provider.embed(text)
        item = MemoryEntry(
            entry_id=entry_id,
            text=text,
            metadata=metadata or {},
            embedding=embedding,
        )
        with self._lock:
            self._items.append(item)

    def query(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[tuple[float, MemoryEntry]]:
        q = self.provider.embed(query)
        with self._lock:
            scored = [
                (cosine_similarity(q, item.embedding), item)
                for item in self._items
            ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x for x in scored[:top_k] if x[0] >= min_score]

    def snapshot(self) -> list[MemoryEntry]:
        with self._lock:
            return list(self._items)
