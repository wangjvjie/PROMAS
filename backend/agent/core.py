"""ReAct Agent Core — Two-Phase Think → Act loop.

v5 architecture:
  Each step is TWO separate LLM calls:
    1. THINK — LLM called WITHOUT tools → produces reasoning (text only)
    2. ACT   — LLM called WITH tools    → produces a tool call (no text)

  This eliminates the #1 failure mode of v3/v4: the model "thinking" for
  many steps without ever calling a tool, burning through max_steps.
  Now if ACT doesn't produce a tool call, we retry ACT directly (up to 2x),
  then force-finish. No ambiguity, no stuck loops.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ast
from typing import AsyncGenerator

from ..models import AgentStep, AgentResult, SSEvent, EventType, ToolCall
from .context import ContextManager
from .tools import ToolRouter, TOOL_DEFINITIONS, VERIFY_TOOL_DEFINITIONS
from .prompts import (
    REACT_SYSTEM_PROMPT, REACT_USER_PROMPT,
    REACT_PICK_SYSTEM_PROMPT, REACT_PICK_USER_PROMPT,
    VERIFY_SYSTEM_PROMPT, VERIFY_USER_PROMPT,
    THINK_FIRST, THINK_AFTER_OBSERVE,
    ACT_INSTRUCTION, ACT_RETRY, FORCE_FINISH,
)
from .llm import (
    llm_chat, llm_chat_with_tools, LLMResponse,
    estimate_messages_tokens, get_context_limit,
)

logger = logging.getLogger("promas-nl2repo")

# ── Constants ────────────────────────────────────────────────────────────
CONTEXT_PRUNE_THRESHOLD = 0.85
OUTPUT_TOKEN_BUDGET = 65536
ACT_MAX_RETRIES = 2  # retries if ACT phase doesn't produce a tool call


class ReactAgent:
    """Two-phase ReAct agent: Think (reason) → Act (tool call) → Observe."""

    def __init__(self, ctx: ContextManager, max_steps: int = 15,
                 user_message: str = "", prd: str = "", threat_model_text: str = ""):
        self.ctx = ctx
        self.tools = ToolRouter(ctx)
        self.max_steps = max_steps
        # Always-visible context injected into every system prompt
        self.user_message = user_message
        self.prd = prd
        self.threat_model_text = threat_model_text

    # ═══════════════════════════════════════════════════════════════════════
    # Core Primitives: _think() and _act()
    # ═══════════════════════════════════════════════════════════════════════

    async def _think(
        self,
        messages: list[dict],
        think_prompt: str,
    ) -> str:
        """Phase 1: Call LLM WITHOUT tools to get reasoning.

        Appends the think instruction as a user message, calls LLM (no tools),
        appends the assistant's reasoning, and returns the thought text.
        """
        messages.append({"role": "user", "content": think_prompt})

        thought = await llm_chat(
            messages=messages,
            temperature=0.3,
            max_tokens=1024,  # enough for complex cross-file reasoning
        )

        messages.append({"role": "assistant", "content": thought})
        return thought.strip()

    async def _act(
        self,
        messages: list[dict],
        tools: list[dict],
        max_retries: int = ACT_MAX_RETRIES,
    ) -> LLMResponse | None:
        """Phase 2: Call LLM WITH tools to get exactly one tool call.

        Appends the act instruction, calls LLM with tools.
        If no tool call is produced, retries up to max_retries times.
        Returns the LLMResponse (with tool call) or None if all retries fail.
        """
        messages.append({"role": "user", "content": ACT_INSTRUCTION})

        for attempt in range(1, max_retries + 2):  # 1 + max_retries
            try:
                resp = await llm_chat_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=OUTPUT_TOKEN_BUDGET,
                )
            except Exception as e:
                logger.error(f"ACT LLM call failed (attempt {attempt}): {e}")
                if attempt > max_retries:
                    return None
                continue

            if resp.get_first_tool():
                # Success — append raw message for conversation history
                messages.append(resp.raw_message)
                return resp

            # No tool call — retry
            if attempt <= max_retries:
                logger.warning(
                    f"ACT produced no tool call (attempt {attempt}/{max_retries + 1}), retrying"
                )
                # Append what the model said, then retry instruction
                messages.append(resp.raw_message)
                messages.append({"role": "user", "content": ACT_RETRY})
            else:
                messages.append(resp.raw_message)

        return None  # all retries exhausted

    # ═══════════════════════════════════════════════════════════════════════
    # Handle finish() tool call — shared logic
    # ═══════════════════════════════════════════════════════════════════════

    async def _handle_finish(
        self,
        file_name: str,
        tool_call,
        llm_resp: LLMResponse,
        messages: list[dict],
    ) -> str | None:
        """Process a finish() tool call. Returns code string or None if unusable.

        Handles: truncated JSON args, empty code, validation, review.
        If None is returned, appropriate error messages have been appended to messages.
        """
        code = tool_call.arguments.get("code", "")

        # ── Truncated JSON args ───────────────────────────────────────
        if tool_call.arguments.get("_parse_failed"):
            raw = tool_call.arguments.get("_raw_truncated", "")
            logger.warning(f"finish() had truncated JSON args ({len(raw)} chars)")
            code = _salvage_code_from_truncated_json(raw)
            if not code:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        "[Error] finish() was truncated. "
                        "Minimize comments, keep code concise. Try again."
                    ),
                })
                return None

        # ── Empty code ────────────────────────────────────────────────
        if not code:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "[Error] The `code` argument was empty. Call finish again with full content.",
            })
            return None

        # ── Validation + review ───────────────────────────────────────
        needs_review = False
        review_reason = ""

        if llm_resp.was_truncated:
            needs_review = True
            review_reason = f"Output truncated ({len(code)} chars)"
        else:
            validation = _validate_code(file_name, code)
            if not validation["ok"]:
                needs_review = True
                review_reason = f"Validation: {validation['error']}"

        if needs_review:
            logger.info(f"Review triggered for {file_name}: {review_reason}")
            code = await self._review_and_rewrite(file_name, code, review_reason)

        return code

    # ═══════════════════════════════════════════════════════════════════════
    # generate_file — single file, name given externally
    # ═══════════════════════════════════════════════════════════════════════

    async def generate_file(self, file_name: str) -> AsyncGenerator[SSEvent, None]:
        """Two-phase ReAct loop for a single file."""

        file_design = self.ctx.get_design_for_prompt(file_name)
        focused_arch = self.ctx.get_focused_architecture(file_name)

        messages: list[dict] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT.format(
                user_message=self.user_message,
                prd=self.prd,
                threat_model=self.threat_model_text,
            )},
            {"role": "user", "content": REACT_USER_PROMPT.format(
                file_name=file_name,
                full_architecture=focused_arch,
                file_design=file_design,
            )},
        ]

        last_observation: str | None = None

        for step_num in range(1, self.max_steps + 1):
            _maybe_prune_context(messages)

            # ── THINK ─────────────────────────────────────────────────
            think_prompt = (
                THINK_AFTER_OBSERVE.format(observation=last_observation)
                if last_observation
                else THINK_FIRST
            )

            try:
                thought = await self._think(messages, think_prompt)
            except Exception as e:
                yield SSEvent(type=EventType.ERROR, file=file_name,
                              content=f"Think failed at step {step_num}: {e}", step=step_num)
                return

            yield SSEvent(type=EventType.AGENT_THINK, file=file_name,
                          content=thought, step=step_num)

            # ── ACT ───────────────────────────────────────────────────
            llm_resp = await self._act(messages, TOOL_DEFINITIONS)

            if not llm_resp or not llm_resp.get_first_tool():
                yield SSEvent(type=EventType.WARN, file=file_name,
                              content=f"ACT produced no tool call after retries. Forcing finish.",
                              step=step_num)
                break  # fall through to force-finish

            tool_call = llm_resp.get_first_tool()

            # ── finish ────────────────────────────────────────────────
            if tool_call.name == "finish":
                yield SSEvent(type=EventType.AGENT_ACT, file=file_name,
                              content=f"finish({len(tool_call.arguments.get('code', ''))} chars)",
                              step=step_num)

                code = await self._handle_finish(file_name, tool_call, llm_resp, messages)
                if code is None:
                    last_observation = "finish() failed — see error above. Try again."
                    continue

                self._write_file(file_name, code)
                yield SSEvent(type=EventType.FILE_WRITTEN, file=file_name,
                              content=f"Written {len(code)} chars ({len(code.splitlines())} lines)",
                              step=step_num)
                return

            # ── Context-gathering tool ────────────────────────────────
            yield SSEvent(type=EventType.AGENT_ACT, file=file_name,
                          content=f"{tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)})",
                          step=step_num)

            observation = self.tools.execute(tool_call.name, tool_call.arguments)
            last_observation = observation

            yield SSEvent(type=EventType.AGENT_OBSERVE, file=file_name,
                          content=observation[:500] + ("..." if len(observation) > 500 else ""),
                          step=step_num)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation,
            })

        # ── Force finish ──────────────────────────────────────────────────
        yield SSEvent(type=EventType.WARN, file=file_name,
                      content=f"Max steps ({self.max_steps}) reached. Forcing finish.")

        code = await self._force_finish(file_name, messages)
        if code:
            self._write_file(file_name, code)
            yield SSEvent(type=EventType.FILE_WRITTEN, file=file_name,
                          content=f"Written {len(code)} chars (forced)")

    # ═══════════════════════════════════════════════════════════════════════
    # generate_next_file — agent picks which file, then writes it
    # ═══════════════════════════════════════════════════════════════════════

    async def generate_next_file(self) -> AsyncGenerator[SSEvent, None]:
        """Two-phase ReAct: pick a file, then write it."""

        file_index = self.ctx.get_file_index()
        remaining = self.ctx.get_remaining_files()
        dep_summary = self.ctx.get_dependency_summary()

        written_files = self.ctx.get_written_files()
        written_lines = []
        for wf_path in sorted(written_files.keys()):
            line_count = len(written_files[wf_path].splitlines())
            written_lines.append(f"  ✓ {wf_path} ({line_count} lines)")
        written_summary = "\n".join(written_lines) if written_lines else "  (none yet)"

        messages: list[dict] = [
            {"role": "system", "content": REACT_PICK_SYSTEM_PROMPT.format(
                user_message=self.user_message,
                prd=self.prd,
                full_architecture=self.ctx.get_full_architecture_summary(),
                threat_model=self.threat_model_text,
            )},
            {"role": "user", "content": REACT_PICK_USER_PROMPT.format(
                file_index=file_index,
                dep_summary=dep_summary,
                remaining_list=", ".join(remaining),
                written_summary=written_summary,
            )},
        ]

        current_file: str | None = None
        self.last_picked_file = None
        last_observation: str | None = None

        for step_num in range(1, self.max_steps + 1):
            _maybe_prune_context(messages)

            # ── THINK ─────────────────────────────────────────────────
            think_prompt = (
                THINK_AFTER_OBSERVE.format(observation=last_observation)
                if last_observation
                else THINK_FIRST
            )

            try:
                thought = await self._think(messages, think_prompt)
            except Exception as e:
                yield SSEvent(type=EventType.ERROR,
                              file=current_file or "(picking)",
                              content=f"Think failed at step {step_num}: {e}",
                              step=step_num)
                return

            yield SSEvent(type=EventType.AGENT_THINK,
                          file=current_file or "(picking)",
                          content=thought, step=step_num)

            # ── ACT ───────────────────────────────────────────────────
            llm_resp = await self._act(messages, TOOL_DEFINITIONS)

            if not llm_resp or not llm_resp.get_first_tool():
                yield SSEvent(type=EventType.WARN,
                              file=current_file or "(picking)",
                              content="ACT produced no tool call. Forcing finish.",
                              step=step_num)
                break

            tool_call = llm_resp.get_first_tool()

            # ── pick_next_file ────────────────────────────────────────
            if tool_call.name == "pick_next_file":
                file_name_arg = tool_call.arguments.get("file_name", "")
                reason = tool_call.arguments.get("reason", "")

                yield SSEvent(type=EventType.AGENT_ACT, file="(picking)",
                              content=f"pick_next_file({file_name_arg!r}, reason={reason!r})",
                              step=step_num)

                observation = self.tools.execute("pick_next_file", tool_call.arguments)

                if observation.startswith("✅"):
                    current_file = self.ctx._resolve_file_key(file_name_arg)
                    self.last_picked_file = current_file
                    # Inject focused architecture
                    focused = self.ctx.get_focused_architecture(current_file)
                    observation += f"\n\n## Focused Architecture for {current_file}\n{focused}"

                last_observation = observation

                yield SSEvent(type=EventType.AGENT_OBSERVE,
                              file=current_file or "(picking)",
                              content=observation[:500] + ("..." if len(observation) > 500 else ""),
                              step=step_num)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })
                continue

            # ── finish ────────────────────────────────────────────────
            if tool_call.name == "finish":
                if not current_file:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "[Error] Call `pick_next_file` before `finish`.",
                    })
                    last_observation = "Error: must pick a file first."
                    continue

                yield SSEvent(type=EventType.AGENT_ACT, file=current_file,
                              content=f"finish({len(tool_call.arguments.get('code', ''))} chars)",
                              step=step_num)

                code = await self._handle_finish(current_file, tool_call, llm_resp, messages)
                if code is None:
                    last_observation = "finish() failed — see error above."
                    continue

                self._write_file(current_file, code)
                yield SSEvent(type=EventType.FILE_WRITTEN, file=current_file,
                              content=f"Written {len(code)} chars ({len(code.splitlines())} lines)",
                              step=step_num)
                return

            # ── Other tools (read_file, search_code, etc.) ────────────
            yield SSEvent(type=EventType.AGENT_ACT,
                          file=current_file or "(picking)",
                          content=f"{tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)})",
                          step=step_num)

            observation = self.tools.execute(tool_call.name, tool_call.arguments)
            last_observation = observation

            yield SSEvent(type=EventType.AGENT_OBSERVE,
                          file=current_file or "(picking)",
                          content=observation[:500] + ("..." if len(observation) > 500 else ""),
                          step=step_num)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation,
            })

        # ── Force finish ──────────────────────────────────────────────────
        if not current_file:
            yield SSEvent(type=EventType.WARN, file="(picking)",
                          content=f"Max steps without picking a file.")
            return

        yield SSEvent(type=EventType.WARN, file=current_file,
                      content=f"Max steps reached for {current_file}. Forcing finish.")

        code = await self._force_finish(current_file, messages)
        if code:
            self._write_file(current_file, code)
            yield SSEvent(type=EventType.FILE_WRITTEN, file=current_file,
                          content=f"Written {len(code)} chars (forced)")

    # ═══════════════════════════════════════════════════════════════════════
    # verify_project — build → diagnose → fix loop
    # ═══════════════════════════════════════════════════════════════════════

    async def verify_project(self, max_steps: int = 20) -> AsyncGenerator[SSEvent, None]:
        """Two-phase ReAct for verification: Think → Act with verify tools."""

        file_index = self.ctx.get_file_index()

        messages: list[dict] = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": VERIFY_USER_PROMPT.format(file_index=file_index)},
        ]

        edits_made = 0
        last_observation: str | None = None

        for step_num in range(1, max_steps + 1):
            _maybe_prune_context(messages)

            # ── THINK ─────────────────────────────────────────────────
            think_prompt = (
                THINK_AFTER_OBSERVE.format(observation=last_observation)
                if last_observation
                else THINK_FIRST
            )

            try:
                thought = await self._think(messages, think_prompt)
            except Exception as e:
                yield SSEvent(type=EventType.ERROR, stage="verify",
                              content=f"Think failed: {e}", step=step_num)
                return

            yield SSEvent(type=EventType.AGENT_THINK, stage="verify",
                          content=thought, step=step_num)

            # ── ACT ───────────────────────────────────────────────────
            llm_resp = await self._act(messages, VERIFY_TOOL_DEFINITIONS)

            if not llm_resp or not llm_resp.get_first_tool():
                yield SSEvent(type=EventType.WARN, stage="verify",
                              content="ACT produced no tool call. Stopping verify.",
                              step=step_num)
                return

            tool_call = llm_resp.get_first_tool()

            # ── verify_done ───────────────────────────────────────────
            if tool_call.name == "verify_done":
                summary = tool_call.arguments.get("summary", "OK")
                msg = f"Verify complete: {summary}"
                if edits_made:
                    msg += f" ({edits_made} edit(s))"
                yield SSEvent(type=EventType.LOG, stage="verify",
                              content=msg, step=step_num)
                return

            # ── edit_file — track edits ───────────────────────────────
            if tool_call.name == "edit_file":
                edit_path = tool_call.arguments.get("path", "")
                yield SSEvent(type=EventType.AGENT_ACT, stage="verify",
                              file=edit_path, content=f"edit_file({edit_path})",
                              step=step_num)

                observation = self.tools.execute(tool_call.name, tool_call.arguments)
                last_observation = observation

                if observation.startswith("✅"):
                    edits_made += 1
                    yield SSEvent(type=EventType.FILE_EDITED, stage="verify",
                                  file=edit_path, content=observation, step=step_num)
                else:
                    yield SSEvent(type=EventType.AGENT_OBSERVE, stage="verify",
                                  content=observation[:500], step=step_num)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })
                continue

            # ── Other tools (run_command, read_file, search_code) ─────
            yield SSEvent(type=EventType.AGENT_ACT, stage="verify",
                          content=f"{tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)[:200]})",
                          step=step_num)

            observation = self.tools.execute(tool_call.name, tool_call.arguments)
            last_observation = observation

            yield SSEvent(type=EventType.AGENT_OBSERVE, stage="verify",
                          content=observation[:800] + ("..." if len(observation) > 800 else ""),
                          step=step_num)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation,
            })

        # Max steps
        msg = f"Verify reached max steps ({max_steps})"
        if edits_made:
            msg += f", {edits_made} edit(s) applied"
        yield SSEvent(type=EventType.WARN, stage="verify", content=msg)

    # ═══════════════════════════════════════════════════════════════════════
    # Force Finish (out of steps)
    # ═══════════════════════════════════════════════════════════════════════

    async def _force_finish(self, file_name: str, messages: list[dict]) -> str | None:
        """Last-resort: prune context aggressively, demand finish."""
        _prune_context_aggressive(messages)
        messages.append({
            "role": "user",
            "content": FORCE_FINISH.format(file_name=file_name),
        })

        try:
            resp = await llm_chat_with_tools(
                messages=messages,
                tools=[t for t in TOOL_DEFINITIONS if t["function"]["name"] == "finish"],
                max_tokens=OUTPUT_TOKEN_BUDGET,
            )
            tool_call = resp.get_first_tool()
            if tool_call and tool_call.name == "finish":
                code = tool_call.arguments.get("code", "")
                if not code and tool_call.arguments.get("_parse_failed"):
                    code = _salvage_code_from_truncated_json(
                        tool_call.arguments.get("_raw_truncated", ""))
                if code:
                    if resp.was_truncated:
                        code = await self._review_and_rewrite(
                            file_name, code, "Forced finish was truncated")
                    return code

            # Last resort: extract code from text
            if resp.content:
                code = _extract_code_from_text(resp.content)
                if code:
                    return code
        except Exception as e:
            logger.error(f"Force finish failed for {file_name}: {e}")

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Review & Rewrite (post-write quality check)
    # ═══════════════════════════════════════════════════════════════════════

    async def _review_and_rewrite(
        self,
        file_name: str,
        code: str,
        reason: str = "truncation suspected",
    ) -> str:
        """Post-write review: compare code against architecture, rewrite if incomplete.
        
        Shows the FULL generated code to the reviewer — no middle truncation.
        Modern models have enough context to review even large files.
        """
        file_design = self.ctx.get_design_for_prompt(file_name)

        # Also provide focused architecture so reviewer knows what other files export
        focused_arch = self.ctx.get_focused_architecture(file_name)

        review_messages = [
            {"role": "system", "content": (
                "You are a code review agent. Your job:\n"
                "1. Compare generated code against the file design AND architecture\n"
                "2. Check that ALL functions/classes from the design are implemented\n"
                "3. Check that ALL imports reference functions that actually exist "
                "in dependency files (see architecture)\n"
                "4. If COMPLETE and correct → call `finish` with the SAME code\n"
                "5. If INCOMPLETE or has wrong references → call `finish` with a COMPLETE rewrite\n\n"
                "Call `finish` exactly once. No text, no thinking — only the tool call."
            )},
            {"role": "user", "content": (
                f"## Review trigger\n{reason}\n\n"
                f"## File: {file_name}\n\n"
                f"## Expected Design\n{file_design}\n\n"
                f"## Project Architecture (for cross-file consistency check)\n{focused_arch}\n\n"
                f"## Generated Code\n```\n{code}\n```\n\n"
                f"Call `finish` with the same code (if complete) or corrected version."
            )},
        ]

        logger.info(f"Review for {file_name}: {reason} ({len(code)} chars)")

        try:
            resp = await llm_chat_with_tools(
                messages=review_messages,
                tools=[t for t in TOOL_DEFINITIONS if t["function"]["name"] == "finish"],
                max_tokens=OUTPUT_TOKEN_BUDGET,
            )

            tool = resp.get_first_tool()
            if tool and tool.name == "finish":
                new_code = tool.arguments.get("code", "")
                if tool.arguments.get("_parse_failed"):
                    logger.warning("Review finish() truncated — keeping original")
                    return code
                if not new_code:
                    return code
                if resp.was_truncated and len(new_code) < len(code) * 0.8:
                    logger.info("Review version shorter — keeping original")
                    return code
                if new_code.strip() != code.strip():
                    logger.info(f"Review rewrote {file_name}: {len(code)} → {len(new_code)} chars")
                return new_code or code
            else:
                return code
        except Exception as e:
            logger.error(f"Review failed for {file_name}: {e}")
            return code

    # ═══════════════════════════════════════════════════════════════════════
    # File I/O
    # ═══════════════════════════════════════════════════════════════════════

    def _write_file(self, file_name: str, code: str):
        """Write a file to disk and register it in context."""
        full_path = os.path.join(self.ctx.work_dir, file_name)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)
        self.ctx.register_written_file(file_name, code)


# ═══════════════════════════════════════════════════════════════════════════
# Context Window Management
# ═══════════════════════════════════════════════════════════════════════════

def _maybe_prune_context(messages: list[dict]):
    """Prune conversation history if context is getting too large.
    
    Key change: preserve read_file / read_architecture tool results in full,
    because those contain the API signatures needed for cross-file consistency.
    Only compress THINK/ACT messages (which are less critical).
    """
    ctx_tokens = estimate_messages_tokens(messages)
    limit = get_context_limit()
    threshold = int(limit * CONTEXT_PRUNE_THRESHOLD)

    if ctx_tokens < threshold:
        return

    logger.info(f"Pruning: ~{ctx_tokens} tokens > {threshold} ({len(messages)} msgs)")

    if len(messages) <= 10:
        return

    keep_head = 2   # system + initial user prompt
    keep_tail = 8   # keep more recent steps

    mid_messages = messages[keep_head:-keep_tail]
    
    # Separate: keep tool results that contain file reads (critical for consistency),
    # compress everything else
    preserved_tool_results = []
    summary_parts = []
    
    for i, msg in enumerate(mid_messages):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        
        if role == "tool" and len(content) > 200:
            # This is likely a read_file or read_architecture result — keep it
            # Check adjacent assistant message for tool name
            tool_name = ""
            if i > 0:
                prev = mid_messages[i - 1]
                if prev.get("tool_calls"):
                    tool_name = prev["tool_calls"][0].get("function", {}).get("name", "")
            
            if tool_name in ("read_file", "read_architecture", "search_code"):
                # Preserve as a standalone context block
                preserved_tool_results.append(
                    f"[Previously read — {tool_name}]\n{content}"
                )
                continue
        
        if role == "assistant" and msg.get("tool_calls"):
            names = [tc["function"]["name"] for tc in msg["tool_calls"]]
            summary_parts.append(f"  [act: {', '.join(names)}]")
        elif role == "tool":
            summary_parts.append(f"  [obs: {content[:150]}...]")
        elif content:
            summary_parts.append(f"  [{role}: {content[:150]}...]")

    summary = "[Conversation history compressed]\n" + "\n".join(summary_parts[-12:])
    
    # Add preserved tool results back as context
    if preserved_tool_results:
        summary += "\n\n[Key file contents from previous reads — DO NOT ignore these]\n"
        summary += "\n\n".join(preserved_tool_results[-5:])  # keep last 5 reads

    new_messages = (
        messages[:keep_head]
        + [{"role": "user", "content": summary}]
        + messages[-keep_tail:]
    )
    messages.clear()
    messages.extend(new_messages)
    logger.info(f"Pruned to {len(new_messages)} messages "
                f"(preserved {len(preserved_tool_results)} tool results)")


def _prune_context_aggressive(messages: list[dict]):
    """Aggressive pruning for force-finish: keep only head + tail."""
    if len(messages) <= 4:
        return
    summary = "[Previous steps removed to maximize output budget.]"
    new = messages[:2] + [{"role": "user", "content": summary}] + messages[-2:]
    messages.clear()
    messages.extend(new)


def _extract_code_from_text(text: str) -> str:
    """Last-resort: extract code from markdown code blocks in plain text."""
    m = re.search(r"```\w*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = text.strip().splitlines()
    if lines and not lines[0].startswith("#") and len(lines) > 5:
        return text.strip()
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Code Validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_code(file_name: str, code: str) -> dict:
    """Validate generated code. Returns {"ok": bool, "error": str, "likely_truncated": bool}."""
    result = {"ok": True, "error": "", "likely_truncated": False}

    if not code or not code.strip():
        result["ok"] = False
        result["error"] = "Empty code"
        return result

    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # Python: ast.parse()
    if ext == ".py":
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["ok"] = False
            result["error"] = f"SyntaxError at line {e.lineno}: {e.msg}"
            total_lines = len(code.splitlines())
            if e.lineno and e.lineno >= total_lines - 2:
                result["likely_truncated"] = True
            truncation_msgs = {
                "unexpected EOF while parsing",
                "EOF while scanning triple-quoted string literal",
                "EOF while scanning string literal",
                "expected an indented block after",
            }
            if e.msg in truncation_msgs:
                result["likely_truncated"] = True
            return result

        # Trailing truncation heuristic
        stripped = code.rstrip()
        if stripped:
            last_line = stripped.splitlines()[-1].rstrip()
            indicators = [
                last_line.endswith("=") and not last_line.endswith("=="),
                last_line.endswith(","),
                last_line.endswith("("),
                last_line.endswith(": ") or last_line.endswith(":\\"),
                last_line.endswith("def "),
                last_line.endswith("class "),
                last_line.endswith("import "),
            ]
            if any(indicators):
                result["ok"] = False
                result["error"] = f"Incomplete last line: '{last_line[-60:]}'"
                result["likely_truncated"] = True
        return result

    # JSON
    if ext == ".json":
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            result["ok"] = False
            result["error"] = f"JSON error: {e.msg} at line {e.lineno}"
            if "Unterminated" in e.msg or "Expecting" in e.msg:
                result["likely_truncated"] = True
        return result

    # Bracket balance for code files
    CODE_EXTS = {
        ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".swift",
    }
    if ext in CODE_EXTS:
        opens = code.count("(") + code.count("[") + code.count("{")
        closes = code.count(")") + code.count("]") + code.count("}")
        diff = opens - closes
        if diff >= 3:
            result["ok"] = False
            result["error"] = f"Bracket imbalance: {diff} more opens than closes"
            result["likely_truncated"] = True
        return result

    return result


def _salvage_code_from_truncated_json(raw_args: str) -> str:
    """Try to extract usable code from a truncated finish() JSON argument."""
    if not raw_args:
        return ""

    patterns = [r'"code"\s*:\s*"', r'"code"\s*:\s*\'']
    for pat in patterns:
        m = re.search(pat, raw_args)
        if m:
            code_start = m.end()
            code_raw = raw_args[code_start:]
            try:
                for suffix in ['"}\n', '"}', '"', '']:
                    try:
                        test = '{"code": "' + code_raw + suffix
                        parsed = json.loads(test)
                        if parsed.get("code"):
                            logger.info(f"Salvaged {len(parsed['code'])} chars from truncated JSON")
                            return parsed["code"]
                    except json.JSONDecodeError:
                        continue

                # Manual unescape
                code = code_raw.replace("\\n", "\n").replace("\\t", "\t")
                code = code.replace('\\"', '"').replace("\\\\", "\\")
                if code.endswith("\\"):
                    code = code[:-1]
                if len(code) > 50:
                    logger.info(f"Salvaged {len(code)} chars via manual unescape")
                    return code
            except Exception:
                pass
    return ""