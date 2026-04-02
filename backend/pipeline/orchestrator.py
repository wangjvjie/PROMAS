"""Pipeline orchestrator — runs PRD → Architecture → Threat Model → Code → Verify.

Improvements over old pipeline.py:
- Stages are standalone async generator functions, not methods on Pipeline
- LLMClient is shared across stages (one instance)
- Auto-detect resume point from workspace state
- Stage-level retry with clear error propagation
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from ..models import SSEvent, EventType, Stage, GenerateRequest
from ..project.state import ProjectState
from ..llm.client import LLMClient
from .stages.prd import run_prd
from .stages.architecture import run_architecture
from .stages.threat_model import run_threat_model
from .stages.code_gen import run_code_gen
from .stages.verify import run_verify

logger = logging.getLogger("promas")

STAGE_ORDER = [Stage.PRD, Stage.ARCHITECTURE, Stage.THREAT_MODEL, Stage.CODE, Stage.VERIFY]

# Stages where failure stops the pipeline (prerequisites for later stages)
CRITICAL_STAGES = {Stage.PRD, Stage.ARCHITECTURE}


class Pipeline:
    def __init__(self, config: GenerateRequest):
        self.config = config
        self.state = ProjectState(work_dir=config.work_dir)
        self.llm = LLMClient(model=config.model)

    async def run(self) -> AsyncGenerator[SSEvent, None]:
        self.state.load()

        begin = self.config.begin_stage
        if begin == Stage.AUTO:
            begin = self._detect_begin()
            yield SSEvent(type=EventType.LOG, stage="pipeline",
                          content=f"Resume point: {begin.value}")

        start_idx = STAGE_ORDER.index(begin)
        stages = STAGE_ORDER[start_idx:]

        for stage in stages:
            yield SSEvent(type=EventType.STAGE_START, stage=stage.value)

            prereq_err = self._check_prereqs(stage)
            if prereq_err:
                yield SSEvent(type=EventType.ERROR, stage=stage.value, content=prereq_err)
                yield SSEvent(type=EventType.DONE, content=f"Stopped: {prereq_err}")
                return

            success = False
            last_error = ""

            for attempt in range(1, self.config.max_retries + 1):
                if attempt > 1:
                    yield SSEvent(type=EventType.RETRY, stage=stage.value,
                                  content=f"Retry {attempt}/{self.config.max_retries}: {last_error[:200]}")
                try:
                    async for event in self._run_stage(stage):
                        yield event
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    yield SSEvent(type=EventType.WARN, stage=stage.value,
                                  content=f"Attempt {attempt} failed: {last_error[:300]}")

            if not success:
                yield SSEvent(type=EventType.ERROR, stage=stage.value,
                              content=f"Stage '{stage.value}' failed after {self.config.max_retries} attempts")
                if stage in CRITICAL_STAGES:
                    yield SSEvent(type=EventType.DONE, content=f"Pipeline stopped at '{stage.value}'")
                    return
                yield SSEvent(type=EventType.WARN, stage=stage.value,
                              content=f"Skipping '{stage.value}', continuing...")
                continue

            yield SSEvent(type=EventType.STAGE_END, stage=stage.value)
            self.state.save()

        yield SSEvent(type=EventType.DONE, content="All stages complete")

    async def _run_stage(self, stage: Stage) -> AsyncGenerator[SSEvent, None]:
        cfg = self.config
        context_limit = int(os.getenv("MODEL_CONTEXT_LIMIT", "128000"))

        if stage == Stage.PRD:
            async for e in run_prd(cfg.prompt, self.state, self.llm):
                yield e

        elif stage == Stage.ARCHITECTURE:
            async for e in run_architecture(cfg.prompt, self.state, self.llm, cfg.max_retries):
                yield e

        elif stage == Stage.THREAT_MODEL:
            async for e in run_threat_model(
                cfg.prompt, self.state, self.llm,
                mode=cfg.threat_model_mode,
                candidates=cfg.threat_model_candidates,
            ):
                yield e

        elif stage == Stage.CODE:
            async for e in run_code_gen(
                cfg.prompt, self.state, self.llm,
                max_steps=cfg.max_steps,
                max_retries=cfg.max_retries,
                context_limit=context_limit,
            ):
                yield e

        elif stage == Stage.VERIFY:
            async for e in run_verify(
                self.state, self.llm,
                max_steps=cfg.max_steps,
                max_rounds=cfg.max_verify_rounds,
                context_limit=context_limit,
            ):
                yield e

    def _check_prereqs(self, stage: Stage) -> str | None:
        if stage == Stage.ARCHITECTURE and not self.state.prd:
            return "PRD is empty — cannot design architecture"
        if stage == Stage.THREAT_MODEL and not self.state.architecture.files:
            return "Architecture is empty — cannot create threat model"
        if stage == Stage.CODE and not self.state.architecture.files:
            return "Architecture is empty — cannot generate code"
        return None

    def _detect_begin(self) -> Stage:
        if not self.state.prd or len(self.state.prd.strip()) < 50:
            logger.info("[auto-stage] No PRD → starting from PRD")
            return Stage.PRD
        if not self.state.architecture.files:
            logger.info("[auto-stage] No architecture → starting from ARCHITECTURE")
            return Stage.ARCHITECTURE
        if not self.state.threat_model.raw_text or len(self.state.threat_model.raw_text.strip()) < 50:
            logger.info("[auto-stage] No threat model → starting from THREAT_MODEL")
            return Stage.THREAT_MODEL
        remaining = self.state.get_remaining()
        if remaining:
            written = len(self.state.get_written())
            total = len(self.state.architecture.files)
            logger.info(f"[auto-stage] {written}/{total} files written → starting from CODE")
            return Stage.CODE
        logger.info("[auto-stage] All files written → starting from VERIFY")
        return Stage.VERIFY
