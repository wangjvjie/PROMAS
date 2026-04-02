"""Agent — single-pass loop with persistent conversation (like claude-code's QueryEngine).

Key design (from claude-code):
  - ONE mutableMessages list persists across all run() calls
  - File reads are cached — no re-reading the same file
  - ContextWindow handles compaction when conversation grows too long
  - Each run() call APPENDS to the existing conversation, not replaces it
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from ..models import SSEvent, EventType
from ..llm.client import LLMClient
from ..tools.base import Tool
from .context_window import ContextWindow

logger = logging.getLogger("promas")

NO_TOOL_MAX = 2


class Agent:
    """Single-pass agent with persistent conversation state.

    Like claude-code's QueryEngine:
      - self.messages persists across run() calls (one long conversation)
      - First run() initializes with system prompt
      - Subsequent run() calls append user prompt and continue
      - ContextWindow compacts when approaching token limit
    """

    def __init__(
        self,
        tools: list[Tool],
        system_prompt: str,
        llm: LLMClient,
        context_window: ContextWindow,
        max_steps: int = 30,
        stage: str = "",
        file: str = "",
    ):
        self.tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.llm = llm
        self.cw = context_window
        self.max_steps = max_steps
        self.stage = stage
        self.file = file
        # Persistent conversation — accumulates across run() calls
        self.messages: list[dict] = []

    def update_tools(self, tools: list[Tool]):
        """Update the tool set (e.g., when WriteFileTool.current_file changes)."""
        self.tool_map = {t.name: t for t in tools}

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        user_prompt: str,
    ) -> AsyncGenerator[SSEvent, None]:
        """Run the agent loop. Appends to persistent conversation.

        First call: initializes with system prompt + user prompt.
        Subsequent calls: just appends user prompt and continues.
        """
        # Initialize on first call
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        # Append new user turn
        self.messages.append({"role": "user", "content": user_prompt})

        tool_schemas = [t.to_schema() for t in self.tool_map.values()]
        no_tool_count = 0

        for step in range(1, self.max_steps + 1):
            self.messages = self.cw.maybe_compact(self.messages)

            # ── LLM call ──────────────────────────────────────────────────
            try:
                response = await self.llm.call_with_tools(self.messages, tool_schemas)
            except Exception as e:
                yield self._evt(EventType.ERROR, f"LLM error at step {step}: {e}", step)
                return

            self.cw.track(response.prompt_tokens, response.completion_tokens)
            self.messages.append(response.raw_message)

            # Yield reasoning text (if any)
            if response.content:
                yield self._evt(EventType.AGENT_THINK, response.content, step)

            # ── No tool calls ─────────────────────────────────────────────
            if not response.tool_calls:
                no_tool_count += 1
                if no_tool_count > NO_TOOL_MAX:
                    yield self._evt(EventType.WARN, "No tool call after retries, stopping.", step)
                    return
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Make progress by calling a tool. "
                        "If you have enough context, call `finish` (or `verify_done`) now."
                    ),
                })
                continue

            no_tool_count = 0

            # ── Partition tool calls ──────────────────────────────────────
            read_calls = [
                tc for tc in response.tool_calls
                if self.tool_map.get(tc.name) and self.tool_map[tc.name].is_read_only
                and tc.name not in ("verify_done",)
            ]
            write_calls = [
                tc for tc in response.tool_calls
                if self.tool_map.get(tc.name) and not self.tool_map[tc.name].is_read_only
            ]
            unknown_calls = [tc for tc in response.tool_calls if tc.name not in self.tool_map]
            terminal_signals = [
                tc for tc in response.tool_calls if tc.name == "verify_done"
            ]

            results: dict[str, str] = {}

            # ── Concurrent read-only ──────────────────────────────────────
            if read_calls:
                outs = await asyncio.gather(
                    *[self._run_tool(tc) for tc in read_calls],
                    return_exceptions=True,
                )
                for tc, out in zip(read_calls, outs):
                    r = f"[Error] {out}" if isinstance(out, Exception) else str(out)
                    results[tc.id] = r
                    yield self._evt(EventType.AGENT_ACT, _fmt_call(tc.name, tc.arguments), step)
                    yield self._evt(EventType.AGENT_OBSERVE, r[:600], step)

            # ── Serial writes (pick_next_file, edit_file, finish) ─────────
            for tc in write_calls:
                file_label = tc.arguments.get("file_name", "") or self.file
                yield self._evt(EventType.AGENT_ACT, _fmt_call(tc.name, tc.arguments), step,
                                file=file_label)
                r = await self._run_tool(tc)
                results[tc.id] = r

                if tc.name == "finish":
                    if r.startswith("✅"):
                        # File written — record in cache and stop this run
                        file_key = self.tool_map["finish"].current_file  # type: ignore[attr-defined]
                        code = tc.arguments.get("code", "")
                        if code and file_key:
                            self.cw.record_written_file(file_key, code)
                        self.messages.append(_tool_msg(tc.id, r))
                        yield self._evt(EventType.FILE_WRITTEN, r, step, file=file_key)
                        return
                    else:
                        yield self._evt(EventType.AGENT_OBSERVE, r[:600], step, file=file_label)
                elif tc.name == "edit_file":
                    if "has been updated successfully" in r:
                        edit_path = tc.arguments.get("path", "")
                        yield self._evt(EventType.FILE_EDITED, r, step, file=edit_path)
                    else:
                        yield self._evt(EventType.AGENT_OBSERVE, r[:600], step)
                else:
                    yield self._evt(EventType.AGENT_OBSERVE, r[:600], step)

            # ── verify_done signal ────────────────────────────────────────
            for tc in terminal_signals:
                r = await self._run_tool(tc)
                results[tc.id] = r
                self.messages.append(_tool_msg(tc.id, r))
                summary = tc.arguments.get("summary", "")
                yield self._evt(EventType.LOG, f"Verify complete: {summary}", step)
                return

            # ── Unknown tools ─────────────────────────────────────────────
            for tc in unknown_calls:
                r = f"[Error] Unknown tool: {tc.name}. Available: {', '.join(self.tool_map)}"
                results[tc.id] = r
                yield self._evt(EventType.AGENT_OBSERVE, r, step)

            # ── Append all tool results to conversation ───────────────────
            for tc in response.tool_calls:
                if tc.id in results:
                    self.messages.append(_tool_msg(tc.id, results[tc.id]))

        yield self._evt(EventType.WARN, f"Reached max steps ({self.max_steps})")

    # ── Helpers ───────────────────────────────────────────���───────────────────

    def _evt(
        self, type: EventType, content: str, step: int = 0, file: str = ""
    ) -> SSEvent:
        return SSEvent(
            type=type,
            stage=self.stage,
            file=file or self.file,
            content=content,
            step=step,
        )

    async def _run_tool(self, tc) -> str:
        tool = self.tool_map.get(tc.name)
        if not tool:
            return f"[Error] Unknown tool: {tc.name}"
        try:
            return await tool.execute(**tc.arguments)
        except TypeError as e:
            return f"[Error] Bad arguments for {tc.name}: {e}"
        except Exception as e:
            logger.exception(f"Tool {tc.name} raised unexpectedly")
            return f"[Error] {tc.name} failed: {e}"


def _fmt_call(name: str, args: dict) -> str:
    if not args:
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if k.startswith("_"):
            continue
        sv = str(v)
        if len(sv) > 120:
            sv = sv[:120] + "..."
        parts.append(f"{k}={sv!r}")
    return f"{name}({', '.join(parts)})"


def _tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
