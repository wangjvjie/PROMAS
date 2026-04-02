"""Verify stage — build → diagnose → fix loop with persistent conversation."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from ...models import SSEvent, EventType
from ...project.state import ProjectState
from ...llm.client import LLMClient
from ...engine.agent import Agent
from ...engine.context_window import ContextWindow
from ...tools import make_verify_tools
from ...tools.env_tools import detect_stack, detect_environment
from ...prompts import VERIFY_SYSTEM_PROMPT, VERIFY_USER_PROMPT

logger = logging.getLogger("promas")


async def run_verify(
    state: ProjectState,
    llm: LLMClient,
    max_steps: int = 25,
    max_rounds: int = 3,
    context_limit: int = 128000,
) -> AsyncGenerator[SSEvent, None]:
    total_files = len(state.get_written())
    yield SSEvent(type=EventType.LOG, stage="verify",
                  content=f"Starting verification for {total_files} files...")

    # Auto-detect tech stack
    stack = detect_stack(state.architecture.files)
    if stack["verify_instructions"]:
        yield SSEvent(type=EventType.LOG, stage="verify",
                      content=stack["verify_instructions"])

    env_info = await detect_environment()

    system_prompt = VERIFY_SYSTEM_PROMPT
    if stack["verify_instructions"]:
        system_prompt += f"\n\n## Project Tech Stack\n{stack['verify_instructions']}"
    system_prompt += f"\n\n## Environment\n{env_info}"

    # ONE persistent agent for all verify rounds (conversation carries over)
    tools = make_verify_tools(state)
    agent = Agent(
        tools=tools,
        system_prompt=system_prompt,
        llm=llm,
        context_window=ContextWindow(max_tokens=context_limit),
        max_steps=max_steps,
        stage="verify",
    )

    total_edits = 0

    for round_num in range(1, max_rounds + 1):
        yield SSEvent(type=EventType.LOG, stage="verify",
                      content=f"Round {round_num}/{max_rounds}")

        user_prompt = VERIFY_USER_PROMPT.format(file_index=state.get_file_index())
        if round_num > 1:
            user_prompt = (
                f"Previous round made {total_edits} edit(s). "
                f"Re-verify: run install + build again to check if fixes worked.\n\n"
                + user_prompt
            )

        round_edits = 0
        try:
            async for event in agent.run(user_prompt):
                yield event
                if event.type == EventType.FILE_EDITED:
                    round_edits += 1
                    total_edits += 1
                    state.save_manifest()
        except Exception as e:
            yield SSEvent(type=EventType.WARN, stage="verify",
                          content=f"Round {round_num} error: {e}")

        if round_edits == 0:
            yield SSEvent(type=EventType.LOG, stage="verify",
                          content=f"Build clean on round {round_num}.")
            break

        yield SSEvent(type=EventType.LOG, stage="verify",
                      content=f"Round {round_num}: {round_edits} edit(s), re-verifying...")

    yield SSEvent(type=EventType.LOG, stage="verify",
                  content=f"Verification complete: {total_edits} total edit(s)")
