"""PRD stage — single LLM call, returns the PRD text."""

from __future__ import annotations

from typing import AsyncGenerator

from ...models import SSEvent, EventType
from ...project.state import ProjectState
from ...llm.client import LLMClient
from ...prompts import WRITE_PRD_PROMPT


async def run_prd(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
) -> AsyncGenerator[SSEvent, None]:
    yield SSEvent(type=EventType.LOG, stage="prd", content="Generating PRD...")

    user_prompt = WRITE_PRD_PROMPT.format(msg=prompt)
    response = await llm.chat([{"role": "user", "content": user_prompt}])

    if not response or len(response.strip()) < 50:
        raise ValueError(f"PRD response too short ({len(response)} chars)")

    state.prd = response
    yield SSEvent(type=EventType.LOG, stage="prd", content=response[:400] + "...")
