"""Code generation stage — one continuous conversation for all files.

Like claude-code: one Agent (one mutableMessages) persists across all files.
The agent writes file A, conversation stays, then writes file B with full
context of what A looked like. No re-reading needed.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from ...models import SSEvent, EventType
from ...project.state import ProjectState
from ...llm.client import LLMClient
from ...engine.agent import Agent
from ...engine.context_window import ContextWindow
from ...tools import make_code_tools
from ...tools.write_tools import WriteFileTool
from ...tools.env_tools import detect_environment
from ...prompts import CODE_SYSTEM_PROMPT, CODE_USER_PROMPT

logger = logging.getLogger("promas")


async def run_code_gen(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
    max_steps: int = 25,
    max_retries: int = 3,
    context_limit: int = 128000,
) -> AsyncGenerator[SSEvent, None]:
    total = len(state.architecture.files)
    written = set(state.get_written())
    yield SSEvent(type=EventType.LOG, stage="code",
                  content=f"Generating {total} files ({len(written)} already done)")

    # Pre-flight: detect environment
    yield SSEvent(type=EventType.LOG, stage="code", content="Detecting environment...")
    env_info = await detect_environment()
    yield SSEvent(type=EventType.LOG, stage="code", content=env_info)

    system_prompt = CODE_SYSTEM_PROMPT.format(
        user_message=prompt,
        prd=state.prd,
        full_architecture=state.get_full_arch(),
        threat_model=state.threat_model.raw_text,
    )
    system_prompt += f"\n\n## Environment\n{env_info}"

    # ── ONE persistent agent for the entire code gen stage ──────���─────────────
    # Like claude-code's QueryEngine: messages accumulate across all files.
    # The agent remembers what it read and wrote — no re-reading needed.
    shared_cw = ContextWindow(max_tokens=context_limit)
    write_tool = WriteFileTool(state)
    tools = make_code_tools(state, write_tool)

    agent = Agent(
        tools=tools,
        system_prompt=system_prompt,
        llm=llm,
        context_window=shared_cw,
        max_steps=max_steps,
        stage="code",
    )

    failed: list[str] = []
    generated = 0

    while True:
        remaining = state.get_remaining()
        pickable = [f for f in remaining if f not in failed]
        if not pickable:
            break

        idx = total - len(pickable) + 1
        written_keys = sorted(state.get_written())
        written_summary = _fmt_list(written_keys, 8)
        remaining_summary = _fmt_list(pickable, 8)

        yield SSEvent(type=EventType.LOG, stage="code",
                      content=f"[{idx}/{total}] Selecting next file... "
                               f"Written: [{written_summary}] | Remaining: [{remaining_summary}]")

        file_key: str | None = None
        success = False

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                yield SSEvent(type=EventType.RETRY, stage="code",
                              content=f"Retry {attempt}/{max_retries}"
                                      + (f" for {file_key}" if file_key else ""))

            try:
                # Reset write_tool for this iteration (clear current_file)
                write_tool.current_file = ""

                # Build the turn prompt — just status update, no content preview needed
                # because the agent already has all prior reads/writes in its conversation
                user_prompt = CODE_USER_PROMPT.format(
                    file_index=state.get_file_index(),
                    written_summary=written_summary,
                    dep_summary=state.get_dep_summary(),
                    remaining_list=", ".join(pickable[:12])
                    + (f" (+{len(pickable) - 12} more)" if len(pickable) > 12 else ""),
                )

                # Continue the SAME conversation (agent.messages persists)
                async for event in agent.run(user_prompt):
                    yield event
                    if event.type == EventType.FILE_WRITTEN:
                        file_key = event.file

                if file_key and file_key in state.get_written():
                    success = True
                    generated += 1
                    state.save_manifest()
                    break

                file_key = file_key or write_tool.current_file or None
                raise ValueError(
                    f"Agent finished without writing"
                    + (f" {file_key}" if file_key else " (no file picked)")
                )

            except Exception as e:
                if attempt == max_retries:
                    failed_name = file_key or "(unknown)"
                    yield SSEvent(type=EventType.WARN, stage="code",
                                  content=f"[{idx}/{total}] {failed_name} failed after "
                                           f"{max_retries} attempts: {e} — skipping")
                    if failed_name != "(unknown)":
                        failed.append(failed_name)
                    else:
                        yield SSEvent(type=EventType.ERROR, stage="code",
                                      content=f"Cannot identify failed file — stopping. Error: {e}")
                        return

    summary = f"Code generation done: {generated} written"
    if failed:
        summary += f", {len(failed)} failed: {', '.join(failed[:5])}"
    yield SSEvent(type=EventType.LOG, stage="code", content=summary)


def _fmt_list(items: list[str], limit: int) -> str:
    if len(items) <= limit:
        return ", ".join(items)
    shown = ", ".join(items[-limit:])
    return f"({len(items) - limit} more), {shown}"
