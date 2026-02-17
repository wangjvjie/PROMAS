from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class LLMClient:
    """Thin wrapper around chat and embedding clients.

    Chat can use any OpenAI-compatible endpoint (e.g. DeepSeek).
    Embedding can be pinned to OpenAI endpoint/key independently.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        chat_model: str = "deepseek-chat",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.chat_client = None
        self.embedding_client = None
        if OpenAI is not None:
            chat_base_url = base_url or os.getenv("PROMAS_BASE_URL", "https://api.deepseek.com")
            chat_api_key = api_key or os.getenv("PROMAS_API_KEY", "YOUR_API_KEY")

            # Embedding defaults to OpenAI key. Base URL is optional; omitted uses SDK default.
            embed_base_url = embedding_base_url or os.getenv("PROMAS_EMBED_BASE_URL")
            embed_api_key = (
                embedding_api_key
                or os.getenv("PROMAS_EMBED_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )

            self.chat_client = OpenAI(
                base_url=chat_base_url,
                api_key=chat_api_key,
            )
            if embed_api_key:
                if embed_base_url:
                    self.embedding_client = OpenAI(
                        base_url=embed_base_url,
                        api_key=embed_api_key,
                    )
                else:
                    self.embedding_client = OpenAI(
                        api_key=embed_api_key,
                    )

    def chat(
        self,
        message: str,
        *,
        temperature: float = 0.4,
        max_completion_tokens: int = 22000,
    ) -> str:
        if self.chat_client is None:
            raise RuntimeError("openai package is not installed; cannot call chat model")
        response = self.chat_client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": message}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        return response.choices[0].message.content or ""

    def embed(self, text: str) -> list[float]:
        if self.embedding_client is None:
            raise RuntimeError(
                "embedding client is not configured; set PROMAS_EMBED_API_KEY or OPENAI_API_KEY"
            )
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=text)
        return list(response.data[0].embedding)


def openai_send_messages(message: str, temperature: float = 0.4) -> str:
    """Backwards-compatible helper."""
    return LLMClient().chat(message=message, temperature=temperature)
