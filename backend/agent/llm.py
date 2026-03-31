"""LLM abstraction layer — wraps OpenAI-compatible chat completions.

Supports any OpenAI-compatible provider (OpenAI, DeepSeek, local vLLM, etc.)
by setting OPENAI_BASE_URL and OPENAI_API_KEY environment variables.

v2 changes:
  - Dynamic max_tokens based on task (raised default for tool calls)
  - Token estimation helpers for context window management
  - Context window limit awareness
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("promas-nl2repo")


@lru_cache(maxsize=1)
def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_model() -> str:
    return os.getenv("OPENAI_MODEL")


def get_context_limit() -> int:
    """Model's context window limit in tokens. Conservative default for DeepSeek V3."""
    return int(os.getenv("MODEL_CONTEXT_LIMIT", "125536"))


# ── Token estimation ──────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: ~3 chars per token for code (conservative)."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += estimate_tokens(content) + 4  # per-message overhead
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", ""))
                total += estimate_tokens(fn.get("arguments", ""))
    return total


# ── Data structures for tool-call responses ──────────────────────────────

@dataclass
class ToolCallResult:
    """A single tool call extracted from the LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from the LLM, may contain text and/or tool calls."""
    content: str = ""
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)
    finish_reason: str = "stop"  # "stop", "length", "tool_calls", etc.

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def was_truncated(self) -> bool:
        """True if the response was cut short by max_tokens."""
        return self.finish_reason == "length"

    def get_first_tool(self) -> ToolCallResult | None:
        return self.tool_calls[0] if self.tool_calls else None

    def raw_message_for_tool(self, tool_call_id: str) -> dict:
        """Return a copy of raw_message containing ONLY the specified tool_call."""
        raw = dict(self.raw_message)
        if "tool_calls" in raw and raw["tool_calls"]:
            raw["tool_calls"] = [
                tc for tc in raw["tool_calls"] if tc["id"] == tool_call_id
            ]
        return raw


# ── Main API functions ───────────────────────────────────────────────────

async def llm_chat(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 8192,
    model: str | None = None,
) -> str:
    """Send a plain chat completion request and return the response text."""
    client = _get_client()
    model = model or get_model()

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""


async def llm_chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 56384,
    model: str | None = None,
) -> LLMResponse:
    """Send a chat completion with function-calling tools.
    
    v2: default max_tokens raised to 16384 to accommodate large file
    generation via the `finish` tool (entire file content as argument).
    """
    client = _get_client()
    model = model or get_model()

    # Log context size for debugging
    ctx_tokens = estimate_messages_tokens(messages)
    logger.debug(f"LLM call: ~{ctx_tokens} input tokens, max_output={max_tokens}")

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    msg = choice.message
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

    result = LLMResponse(
        content=msg.content or "",
        finish_reason=finish_reason,
    )

    # Log truncation explicitly — this is the #1 cause of agent stuck loops
    if finish_reason == "length":
        logger.warning(
            f"⚠️  LLM output TRUNCATED (finish_reason=length, max_tokens={max_tokens}). "
            f"Content length: {len(msg.content or '')} chars, "
            f"tool_calls present: {bool(msg.tool_calls)}"
        )

    # Build the raw message dict for conversation history
    raw = {"role": "assistant"}
    if msg.content:
        raw["content"] = msg.content
    else:
        raw["content"] = None

    # Parse tool calls
    if msg.tool_calls:
        raw["tool_calls"] = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # Truncated JSON from output limit — try to salvage
                raw_args = tc.function.arguments or ""
                logger.warning(
                    f"⚠️  Tool call '{tc.function.name}' has unparseable arguments "
                    f"({len(raw_args)} chars). finish_reason={finish_reason}. "
                    f"Likely output truncation."
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
