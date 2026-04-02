"""LLM client — wraps any OpenAI-compatible chat completions API.

Features:
- Real token tracking via API usage field
- Configurable timeout (default 120s, prevents infinite hangs)
- json_mode with graceful fallback for unsupported models
- Retry with backoff on transient failures
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("promas")

DEFAULT_TOOL_MAX_TOKENS = 65536
DEFAULT_CHAT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))  # seconds per call
MAX_RETRIES = 2


@lru_cache(maxsize=1)
def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=DEFAULT_TIMEOUT,
    )


def get_model(override: str = "") -> str:
    return override or os.getenv("OPENAI_MODEL")


def get_context_limit() -> int:
    return int(os.getenv("MODEL_CONTEXT_LIMIT", "128000"))


# ── Response models ───────────────────────────────────────────────────────────

@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def was_truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def get_first_tool(self) -> ToolCallResult | None:
        return self.tool_calls[0] if self.tool_calls else None


# ── Main client class ─────────────────────────────────────────────────────────

class LLMClient:
    """Stateless LLM client. One instance shared across the pipeline."""

    def __init__(self, model: str = "", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Plain chat completion, returns text.

        json_mode=True adds response_format={"type":"json_object"} which forces
        valid JSON output. Falls back to normal mode if the model doesn't support it.
        """
        client = _get_client()
        kwargs: dict[str, Any] = dict(
            model=get_model(self.model),
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )

        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                call_kwargs = dict(kwargs)
                # Only try json_mode on first attempt; fall back if it fails
                if json_mode and attempt == 1:
                    call_kwargs["response_format"] = {"type": "json_object"}

                resp = await asyncio.wait_for(
                    client.chat.completions.create(**call_kwargs),
                    timeout=DEFAULT_TIMEOUT,
                )
                return resp.choices[0].message.content or ""

            except asyncio.TimeoutError:
                logger.warning(f"LLM chat timeout ({DEFAULT_TIMEOUT}s) attempt {attempt}")
                last_err = TimeoutError(f"LLM call timed out after {DEFAULT_TIMEOUT}s")
            except Exception as e:
                err_str = str(e).lower()
                # json_mode not supported → retry without it
                if json_mode and attempt == 1 and (
                    "response_format" in err_str
                    or "json" in err_str
                    or "not supported" in err_str
                    or "400" in err_str
                ):
                    logger.info(f"json_mode not supported by model, retrying without it")
                    json_mode = False
                    continue
                logger.warning(f"LLM chat error attempt {attempt}: {e}")
                last_err = e

            if attempt <= MAX_RETRIES:
                await asyncio.sleep(min(2 ** attempt, 8))

        raise last_err or RuntimeError("LLM chat failed")

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = DEFAULT_TOOL_MAX_TOKENS,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Chat completion with function-calling tools."""
        client = _get_client()

        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=get_model(self.model),
                        messages=messages,
                        tools=tools,
                        temperature=temperature if temperature is not None else self.temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=DEFAULT_TIMEOUT,
                )
                return self._parse_tool_response(resp)

            except asyncio.TimeoutError:
                logger.warning(f"LLM tool call timeout ({DEFAULT_TIMEOUT}s) attempt {attempt}")
                last_err = TimeoutError(f"LLM call timed out after {DEFAULT_TIMEOUT}s")
            except Exception as e:
                logger.warning(f"LLM tool call error attempt {attempt}: {e}")
                last_err = e

            if attempt <= MAX_RETRIES:
                await asyncio.sleep(min(2 ** attempt, 8))

        raise last_err or RuntimeError("LLM tool call failed")

    @staticmethod
    def _parse_tool_response(resp) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        if finish_reason == "length":
            logger.warning(
                f"LLM output truncated (finish_reason=length, "
                f"tool_calls={bool(msg.tool_calls)}, content_len={len(msg.content or '')})"
            )

        raw: dict[str, Any] = {"role": "assistant", "content": msg.content or None}

        result = LLMResponse(
            content=msg.content or "",
            finish_reason=finish_reason,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

        if msg.tool_calls:
            raw["tool_calls"] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    raw_args = tc.function.arguments or ""
                    logger.warning(
                        f"Tool '{tc.function.name}' has unparseable args "
                        f"({len(raw_args)} chars) — likely truncation"
                    )
                    args = {"_raw_truncated": raw_args, "_parse_failed": True}

                result.tool_calls.append(ToolCallResult(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
                raw["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        result.raw_message = raw
        return result
