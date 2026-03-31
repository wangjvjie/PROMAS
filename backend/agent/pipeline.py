"""Pipeline orchestrator — runs PRD → Architecture → Threat Model → Code → Verify.

Reliability features:
  - Per-batch retry in architecture design (up to max_retries)
  - Per-file retry in code generation
  - Robust JSON/Pydantic parsing that coerces bad types instead of crashing
  - Auto-continue: a failed batch/file is skipped with a warning, not a full stop
  - Stage-level retry: entire stage retried if it fails completely
"""

from __future__ import annotations

import json
import re
import os
import traceback
from typing import AsyncGenerator

from ..models import (
    ArchitectureDesign, FileDesign, ThreatModel, ClassSpec, FunctionSpec,
    Parameter, SSEvent, EventType, Stage, GenerateRequest,
)
from .context import ContextManager
from .core import ReactAgent
from .prompts import (
    WRITE_PRD_PROMPT, WRITE_FILE_DESIGN_PROMPT,
    WRITE_ARCHITECTURE_BATCH_PROMPT,
    THREAT_MODELING_SIMPLE_PROMPT,
    TM_EXTRACT_CHAINS_PROMPT, TM_GENERATE_SC_AM_PROMPT,
    TM_ANALYZE_CHAIN_PROMPT, TM_JUDGE_PROMPT,
)
from .llm import llm_chat


class Pipeline:
    """Orchestrates the full generation pipeline with streaming events."""

    def __init__(self, config: GenerateRequest):
        self.config = config
        self.ctx = ContextManager(work_dir=config.work_dir)
        self.max_retries = max(1, config.max_retries)

    # ═══════════════════════════════════════════════════════════════════════
    # Main run loop with stage-level retry
    # ═══════════════════════════════════════════════════════════════════════

    async def run(self) -> AsyncGenerator[SSEvent, None]:
        stage_order = [Stage.PRD, Stage.ARCHITECTURE, Stage.THREAT_MODEL, Stage.CODE, Stage.VERIFY]

        self.ctx.load_state()

        # ── Auto-detect begin_stage ───────────────────────────────────
        begin_stage = self.config.begin_stage
        if begin_stage == Stage.AUTO:
            begin_stage = self._detect_begin_stage()
            yield SSEvent(type=EventType.LOG, stage="pipeline",
                           content=f"Auto-detected resume point: {begin_stage.value}")

        start_idx = stage_order.index(begin_stage)
        stages = stage_order[start_idx:]

        for stage in stages:
            yield SSEvent(type=EventType.STAGE_START, stage=stage.value)

            success = False
            last_error = ""

            for attempt in range(1, self.max_retries + 1):
                try:
                    handler = {
                        Stage.PRD: self._run_prd,
                        Stage.ARCHITECTURE: self._run_architecture,
                        Stage.THREAT_MODEL: self._run_threat_model,
                        Stage.CODE: self._run_code,
                        Stage.VERIFY: self._run_verify,
                    }.get(stage)

                    if not handler:
                        break

                    # Check prerequisites
                    prereq_error = self._check_prerequisites(stage)
                    if prereq_error:
                        yield SSEvent(type=EventType.ERROR, stage=stage.value, content=prereq_error)
                        yield SSEvent(type=EventType.DONE, content=f"Stopped: {prereq_error}")
                        return

                    if attempt > 1:
                        yield SSEvent(type=EventType.RETRY, stage=stage.value,
                                       content=f"Retry {attempt}/{self.max_retries} for stage '{stage.value}' — last error: {last_error[:200]}")

                    async for event in handler():
                        yield event

                    success = True
                    break

                except Exception as e:
                    last_error = str(e)
                    yield SSEvent(type=EventType.WARN, stage=stage.value,
                                   content=f"Stage '{stage.value}' attempt {attempt} failed: {last_error[:300]}")

            if not success:
                yield SSEvent(type=EventType.ERROR, stage=stage.value,
                               content=f"Stage '{stage.value}' failed after {self.max_retries} attempts")
                # Critical stages stop the pipeline; others can be skipped
                if stage in (Stage.PRD, Stage.ARCHITECTURE):
                    yield SSEvent(type=EventType.DONE, content=f"Pipeline stopped at '{stage.value}'")
                    return
                else:
                    yield SSEvent(type=EventType.WARN, stage=stage.value,
                                   content=f"Skipping failed stage '{stage.value}', continuing...")
                    continue

            yield SSEvent(type=EventType.STAGE_END, stage=stage.value)
            self.ctx.save_state()

        yield SSEvent(type=EventType.DONE, content="All stages complete")

    def _check_prerequisites(self, stage: Stage) -> str | None:
        if stage == Stage.ARCHITECTURE and not self.ctx.prd:
            return "PRD is empty — cannot design architecture"
        if stage == Stage.THREAT_MODEL and not self.ctx.architecture.files:
            return "Architecture is empty — cannot create threat model"
        return None

    def _detect_begin_stage(self) -> Stage:
        """Detect the first incomplete stage by checking workspace state.

        Called after ctx.load_state(), so self.ctx already has whatever was
        previously persisted to disk.

        Logic:
          1. No PRD                         → PRD
          2. No architecture files           → ARCHITECTURE
          3. No threat model                 → THREAT_MODEL
          4. Has remaining files to generate → CODE  (already-written files are auto-skipped)
          5. All files written               → VERIFY
        """
        import logging
        logger = logging.getLogger("promas-nl2repo")

        # 1. PRD
        if not self.ctx.prd or len(self.ctx.prd.strip()) < 50:
            logger.info("[auto-stage] No PRD found → starting from PRD")
            return Stage.PRD

        # 2. Architecture
        if not self.ctx.architecture.files:
            logger.info("[auto-stage] No architecture found → starting from ARCHITECTURE")
            return Stage.ARCHITECTURE

        # 3. Threat model
        if not self.ctx.threat_model.raw_text or len(self.ctx.threat_model.raw_text.strip()) < 50:
            logger.info("[auto-stage] No threat model found → starting from THREAT_MODEL")
            return Stage.THREAT_MODEL

        # 4. Code — check if there are remaining files to write
        remaining = self.ctx.get_remaining_files()
        written = self.ctx.get_written_files()
        total = len(self.ctx.architecture.files)

        if remaining:
            logger.info(
                f"[auto-stage] {len(written)}/{total} files written, "
                f"{len(remaining)} remaining → starting from CODE"
            )
            return Stage.CODE

        # 5. All files written → verify
        logger.info(f"[auto-stage] All {total} files written → starting from VERIFY")
        return Stage.VERIFY

    # ═══════════════════════════════════════════════════════════════════════
    # Stage: PRD
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_prd(self) -> AsyncGenerator[SSEvent, None]:
        yield SSEvent(type=EventType.LOG, stage="prd", content="Generating PRD...")

        prompt = WRITE_PRD_PROMPT.format(msg=self.config.prompt)
        response = await llm_chat([{"role": "user", "content": prompt}])

        if not response or len(response.strip()) < 50:
            raise ValueError(f"PRD response too short ({len(response)} chars), likely incomplete")

        self.ctx.prd = response
        yield SSEvent(type=EventType.LOG, stage="prd", content=response[:300] + "...")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage: Architecture (with per-batch retry)
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_architecture(self) -> AsyncGenerator[SSEvent, None]:
        # Step 1: File design
        yield SSEvent(type=EventType.LOG, stage="architecture", content="Designing file structure...")

        prompt = WRITE_FILE_DESIGN_PROMPT.format(msg=self.config.prompt, prd=self.ctx.prd)
        response = await llm_chat([{"role": "user", "content": prompt}])
        files_data = self._extract_json(response)
        raw_files = files_data.get("files", [])

        file_designs = []
        for f in raw_files:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            path = self._normalize_path(f.get("path", "./"))
            deps = f.get("dependencies", [])
            if not isinstance(deps, list):
                deps = [str(deps)] if deps else []
            file_designs.append(FileDesign(
                name=str(f["name"]),
                path=path,
                description=str(f.get("description", "")),
                dependencies=[str(d) for d in deps],
            ))

        if not file_designs:
            raise ValueError("File design returned 0 files")

        yield SSEvent(type=EventType.LOG, stage="architecture",
                       content=f"Designed {len(file_designs)} files")

        # Step 2: Batch architecture design with per-batch retry
        BATCH_SIZE = 4
        designed_files = []
        failed_batches = 0

        for batch_idx, i in enumerate(range(0, len(file_designs), BATCH_SIZE)):
            batch = file_designs[i:i + BATCH_SIZE]
            batch_names = [f.name for f in batch]
            batch_num = batch_idx + 1

            yield SSEvent(type=EventType.LOG, stage="architecture",
                           content=f"Designing APIs for batch {batch_num}: {', '.join(batch_names)}")

            batch_designed = None
            last_batch_error = ""

            for attempt in range(1, self.max_retries + 1):
                try:
                    if attempt > 1:
                        yield SSEvent(type=EventType.RETRY, stage="architecture",
                                       content=f"Batch {batch_num} retry {attempt}/{self.max_retries}: {last_batch_error[:150]}")

                    summary_lines = []
                    for df in designed_files:
                        exports = [c.class_name for c in df.classes] + [fn.name for fn in df.functions]
                        # Show full function signatures for cross-batch consistency
                        sig_parts = []
                        for fn in df.functions:
                            params = ", ".join(f"{p.name}: {p.type}" for p in fn.input_parameters)
                            sig_parts.append(f"{fn.name}({params})")
                        for cls in df.classes:
                            sig_parts.append(f"class {cls.class_name}")
                        summary_lines.append(
                            f"  {df.path}/{df.name}: {', '.join(sig_parts)}"
                        )
                    designed_summary = "\n".join(summary_lines) or "(none yet)"

                    prompt = WRITE_ARCHITECTURE_BATCH_PROMPT.format(
                        msg=self.config.prompt,
                        prd=self.ctx.prd,
                        files=json.dumps({"files": [
                            {"name": f.name, "path": f.path, "description": f.description}
                            for f in file_designs
                        ]}, indent=2),
                        designed_summary=designed_summary,
                        batch_files=json.dumps({"files": [
                            {"name": f.name, "path": f.path, "description": f.description,
                             "dependencies": f.dependencies}
                            for f in batch
                        ]}, indent=2),
                    )

                    resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=8192)
                    batch_data = self._extract_json(resp)

                    batch_designed = []
                    parse_warnings = []

                    for fd_raw in batch_data.get("files", []):
                        try:
                            fd = self._parse_file_design_safe(fd_raw, batch)
                            batch_designed.append(fd)
                        except Exception as pe:
                            parse_warnings.append(f"{fd_raw.get('name', '?')}: {pe}")

                    if parse_warnings:
                        yield SSEvent(type=EventType.WARN, stage="architecture",
                                       content=f"Batch {batch_num}: {len(parse_warnings)} parse issue(s): {'; '.join(parse_warnings[:3])}")

                    if not batch_designed:
                        raise ValueError(f"Batch {batch_num} produced 0 valid file designs")

                    break  # success

                except Exception as e:
                    last_batch_error = str(e)
                    batch_designed = None

            if batch_designed:
                designed_files.extend(batch_designed)
            else:
                # Fallback: create minimal stubs
                yield SSEvent(type=EventType.WARN, stage="architecture",
                               content=f"Batch {batch_num} failed — creating stubs for {len(batch)} files")
                for bf in batch:
                    designed_files.append(FileDesign(
                        name=bf.name, path=bf.path,
                        description=bf.description or "(design failed, using stub)",
                        dependencies=bf.dependencies,
                    ))
                failed_batches += 1

        if not designed_files:
            raise ValueError("Architecture produced 0 files across all batches")

        self.ctx.architecture = ArchitectureDesign(files=designed_files)
        self.ctx.build_index()
        self.ctx.build_dependency_graph()

        summary = f"Architecture complete: {len(designed_files)} files designed"
        if failed_batches:
            summary += f" ({failed_batches} batch(es) used stubs)"
        yield SSEvent(type=EventType.LOG, stage="architecture", content=summary)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage: Threat Model (simple or full mode)
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_threat_model(self) -> AsyncGenerator[SSEvent, None]:
        mode = self.config.threat_model_mode  # "simple" (default) or "full"
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Generating threat model (mode={mode})...")

        if mode == "full":
            async for event in self._run_threat_model_full():
                yield event
        else:
            async for event in self._run_threat_model_simple():
                yield event

    async def _run_threat_model_simple(self) -> AsyncGenerator[SSEvent, None]:
        """Single-pass threat model — lightweight default."""
        arch_summary = self.ctx.get_full_architecture_summary()
        prompt = THREAT_MODELING_SIMPLE_PROMPT.format(
            msg=self.config.prompt, prd=self.ctx.prd, arch=arch_summary)
        response = await llm_chat([{"role": "user", "content": prompt}], max_tokens=8192)

        if not response or len(response.strip()) < 100:
            raise ValueError(f"Threat model too short ({len(response)} chars)")

        self.ctx.threat_model = ThreatModel(raw_text=response)
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=response[:300] + "...")

    async def _run_threat_model_full(self) -> AsyncGenerator[SSEvent, None]:
        """Paper-aligned multi-step threat modeling:
        
        1. Extract entry interfaces + invocation chains from Arch
        2. Generate SC (Global Security Context) + AM (Attacker Model)
        3. Per-chain function-level threat analysis with iterative merge
        4. Generate k candidate threat models
        5. LLM-Judge scoring → select best
        """
        arch_summary = self.ctx.get_full_architecture_summary()
        k = getattr(self.config, 'threat_model_candidates', 3)

        # ── Step 1: Extract entry interfaces + invocation chains ──────
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content="Step 1/5: Extracting entry interfaces and invocation chains...")

        prompt_chains = TM_EXTRACT_CHAINS_PROMPT.format(arch=arch_summary)
        chains_resp = await llm_chat([{"role": "user", "content": prompt_chains}], max_tokens=4096)
        chains_data = self._extract_json(chains_resp)
        entry_interfaces = chains_data.get("entry_interfaces", [])

        if not entry_interfaces:
            yield SSEvent(type=EventType.WARN, stage="threat_model",
                           content="No entry interfaces extracted — falling back to simple mode")
            async for event in self._run_threat_model_simple():
                yield event
            return

        interface_summary = "\n".join(
            f"  - {ei.get('interface', '?')} → {ei.get('entry_function', '?')} "
            f"(chain: {len(ei.get('invocation_chain', []))} functions)"
            for ei in entry_interfaces
        )
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Found {len(entry_interfaces)} entry interfaces:\n{interface_summary}")

        # ── Step 2: Generate SC + AM ──────────────────────────────────
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content="Step 2/5: Generating Global Security Context + Attacker Model...")

        ei_text = json.dumps(entry_interfaces, indent=2, ensure_ascii=False)
        prompt_sc_am = TM_GENERATE_SC_AM_PROMPT.format(
            msg=self.config.prompt, prd=self.ctx.prd,
            arch=arch_summary, entry_interfaces=ei_text)
        sc_am_text = await llm_chat([{"role": "user", "content": prompt_sc_am}], max_tokens=4096)

        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"SC + AM generated ({len(sc_am_text)} chars)")

        # ── Step 3: Per-chain threat analysis with iterative merge ────
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Step 3/5: Analyzing {len(entry_interfaces)} invocation chains...")

        # Global function-level threat map: function -> list of threat dicts
        global_ft: dict[str, list[dict]] = {}

        for idx, ei in enumerate(entry_interfaces, 1):
            interface_name = ei.get("interface", f"interface_{idx}")
            chain = ei.get("invocation_chain", [])
            if not chain:
                continue

            yield SSEvent(type=EventType.LOG, stage="threat_model",
                           content=f"  Analyzing chain {idx}/{len(entry_interfaces)}: {interface_name}")

            # Build existing threats summary for this chain's functions
            existing_for_chain = {}
            for func in chain:
                if func in global_ft:
                    existing_for_chain[func] = global_ft[func]

            existing_text = json.dumps(existing_for_chain, indent=2, ensure_ascii=False) \
                if existing_for_chain else "(none — first analysis for these functions)"

            prompt_chain = TM_ANALYZE_CHAIN_PROMPT.format(
                sc_am=sc_am_text,
                interface=interface_name,
                chain=" → ".join(chain),
                arch_excerpt=arch_summary,
                existing_threats=existing_text,
            )

            try:
                chain_resp = await llm_chat(
                    [{"role": "user", "content": prompt_chain}], max_tokens=4096)
                chain_data = self._extract_json(chain_resp)

                # Merge into global FT
                for ft_entry in chain_data.get("function_threats", []):
                    func_name = ft_entry.get("function", "")
                    new_threats = ft_entry.get("threats", [])
                    if not func_name or not new_threats:
                        continue

                    if func_name not in global_ft:
                        global_ft[func_name] = new_threats
                    else:
                        # Merge: add threats with new CWEs not already present
                        existing_cwes = {t.get("cwe", "") for t in global_ft[func_name]}
                        for t in new_threats:
                            if t.get("cwe", "") not in existing_cwes:
                                global_ft[func_name].append(t)
                                existing_cwes.add(t.get("cwe", ""))

            except Exception as e:
                yield SSEvent(type=EventType.WARN, stage="threat_model",
                               content=f"  Chain {idx} analysis failed: {e}")

        total_threats = sum(len(ts) for ts in global_ft.values())
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Function-level threats: {len(global_ft)} functions, "
                               f"{total_threats} total threats identified")

        # ── Step 4: Assemble k candidate threat models ────────────────
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Step 4/5: Assembling {k} candidate threat models...")

        candidates: list[str] = []
        for ci in range(k):
            # Each candidate assembles SC + AM + FT into a coherent document
            ft_sections = []
            for func, threats in global_ft.items():
                threat_lines = []
                for t in threats:
                    threat_lines.append(
                        f"  - **{t.get('cwe', '?')} {t.get('name', '')}**: "
                        f"{t.get('attack_vector', '')}\n"
                        f"    Protection: {t.get('protection', '')}"
                    )
                ft_sections.append(f"#### {func}\n" + "\n".join(threat_lines))

            candidate_text = (
                f"# Threat Model (Candidate {ci + 1})\n\n"
                f"## Global Security Context & Attacker Model\n{sc_am_text}\n\n"
                f"## Entry Interfaces\n{interface_summary}\n\n"
                f"## Function-level Threats\n\n" + "\n\n".join(ft_sections)
            )
            candidates.append(candidate_text)

            # For k>1, re-analyze with temperature variation to get diversity
            if ci < k - 1 and len(entry_interfaces) > 0:
                # Re-run chain analysis with higher temperature for diversity
                global_ft_variant: dict[str, list[dict]] = {}
                for ei in entry_interfaces:
                    chain = ei.get("invocation_chain", [])
                    if not chain:
                        continue
                    existing_text = json.dumps(
                        {f: global_ft_variant.get(f, []) for f in chain if f in global_ft_variant},
                        indent=2, ensure_ascii=False) or "(none)"
                    prompt_chain = TM_ANALYZE_CHAIN_PROMPT.format(
                        sc_am=sc_am_text,
                        interface=ei.get("interface", ""),
                        chain=" → ".join(chain),
                        arch_excerpt=arch_summary,
                        existing_threats=existing_text,
                    )
                    try:
                        resp = await llm_chat(
                            [{"role": "user", "content": prompt_chain}],
                            max_tokens=4096, temperature=0.6 + ci * 0.15)
                        data = self._extract_json(resp)
                        for ft_entry in data.get("function_threats", []):
                            fn = ft_entry.get("function", "")
                            ts = ft_entry.get("threats", [])
                            if fn and ts:
                                if fn not in global_ft_variant:
                                    global_ft_variant[fn] = ts
                                else:
                                    existing_cwes = {t.get("cwe", "") for t in global_ft_variant[fn]}
                                    for t in ts:
                                        if t.get("cwe", "") not in existing_cwes:
                                            global_ft_variant[fn].append(t)
                                            existing_cwes.add(t.get("cwe", ""))
                    except Exception:
                        pass
                # Use variant for this candidate
                global_ft = global_ft_variant if global_ft_variant else global_ft

        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Generated {len(candidates)} candidates")

        # ── Step 5: LLM-Judge scoring ─────────────────────────────────
        if len(candidates) > 1:
            yield SSEvent(type=EventType.LOG, stage="threat_model",
                           content="Step 5/5: LLM-Judge scoring candidates...")

            arch_brief = self.ctx.get_file_index()
            best_idx = 0
            best_score = -1.0
            alpha, beta, gamma = 0.33, 0.33, 0.34

            for ci, cand in enumerate(candidates):
                prompt_judge = TM_JUDGE_PROMPT.format(
                    msg=self.config.prompt,
                    arch_summary=arch_brief,
                    candidate=cand,
                )
                try:
                    judge_resp = await llm_chat(
                        [{"role": "user", "content": prompt_judge}], max_tokens=512)
                    scores = self._extract_json(judge_resp)
                    s_r = float(scores.get("relevance", 5))
                    s_p = float(scores.get("impact", 5))
                    s_e = float(scores.get("exploitability", 5))
                    total = alpha * s_r + beta * s_p + gamma * s_e

                    yield SSEvent(type=EventType.LOG, stage="threat_model",
                                   content=f"  Candidate {ci + 1}: "
                                           f"R={s_r:.1f} I={s_p:.1f} E={s_e:.1f} → S={total:.2f}")

                    if total > best_score:
                        best_score = total
                        best_idx = ci
                except Exception as e:
                    yield SSEvent(type=EventType.WARN, stage="threat_model",
                                   content=f"  Judge failed for candidate {ci + 1}: {e}")

            yield SSEvent(type=EventType.LOG, stage="threat_model",
                           content=f"Selected candidate {best_idx + 1} (score={best_score:.2f})")
            final_text = candidates[best_idx]
        else:
            yield SSEvent(type=EventType.LOG, stage="threat_model",
                           content="Step 5/5: Single candidate — skipping judge")
            final_text = candidates[0]

        self.ctx.threat_model = ThreatModel(raw_text=final_text)
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                       content=f"Threat model complete ({len(final_text)} chars, "
                               f"{len(global_ft)} functions covered)")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage: Code Generation (with per-file retry)
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_code(self) -> AsyncGenerator[SSEvent, None]:
        files = self.ctx.architecture.files
        total = len(files)
        written = set(self.ctx.get_written_files().keys())

        yield SSEvent(type=EventType.LOG, stage="code",
                       content=f"Generating code for {total} files ({len(written)} already written)")

        agent = ReactAgent(
            ctx=self.ctx,
            max_steps=self.config.max_react_steps,
            user_message=self.config.prompt,
            prd=self.ctx.prd,
            threat_model_text=self.ctx.threat_model.raw_text,
        )
        failed_files: list[str] = []
        generated_count = 0

        while True:
            remaining = self.ctx.get_remaining_files()
            pickable = [f for f in remaining if f not in failed_files]

            if not pickable:
                break

            idx = total - len(pickable) + 1
            written_keys = sorted(self.ctx.get_written_files().keys())
            written_summary = ", ".join(written_keys[-8:]) if written_keys else "(none)"
            if len(written_keys) > 8:
                written_summary = f"({len(written_keys) - 8} more), " + written_summary
            remaining_summary = ", ".join(pickable[:8])
            if len(pickable) > 8:
                remaining_summary += f", (+{len(pickable) - 8} more)"
            yield SSEvent(type=EventType.LOG, stage="code",
                           content=f"[{idx}/{total}] Agent selecting next file... "
                                   f"Written: [{written_summary}] | Remaining: [{remaining_summary}]")

            file_key = None
            file_success = False

            for attempt in range(1, self.max_retries + 1):
                try:
                    if attempt > 1:
                        yield SSEvent(type=EventType.RETRY, stage="code",
                                       content=f"Retry {attempt}/{self.max_retries}"
                                                + (f" for {file_key}" if file_key else ""))

                    async for event in agent.generate_next_file():
                        event.stage = "code"
                        yield event
                        if event.type == EventType.FILE_WRITTEN:
                            file_key = event.file

                    if file_key and file_key in self.ctx.get_written_files():
                        file_success = True
                        written.add(file_key)
                        generated_count += 1
                        # Incremental save so progress survives interruptions
                        self.ctx.save_manifest()
                        break
                    else:
                        # Agent may have picked but failed to write — track what it picked
                        file_key = file_key or getattr(agent, 'last_picked_file', None)
                        raise ValueError(
                            f"Agent finished without writing"
                            + (f" {file_key}" if file_key else " (no file picked)"))

                except Exception as e:
                    if attempt == self.max_retries:
                        failed_name = file_key or getattr(agent, 'last_picked_file', None) or "(unknown)"
                        yield SSEvent(type=EventType.WARN, stage="code",
                                       content=f"[{idx}/{total}] {failed_name} failed after "
                                               f"{self.max_retries} attempts: {e} — skipping")
                        if failed_name and failed_name != "(unknown)":
                            failed_files.append(failed_name)
                        else:
                            # Can't identify which file failed — bail to avoid infinite loop
                            yield SSEvent(type=EventType.ERROR, stage="code",
                                           content=f"Cannot identify failed file — stopping code generation. Error: {e}")
                            # Must also break the outer while-True loop
                            pickable = []
                            break

            # If we cleared pickable above to signal a bail-out, stop
            if not pickable:
                break

        summary = f"Code generation done: {generated_count} written"
        if failed_files:
            summary += f", {len(failed_files)} failed: {', '.join(failed_files[:5])}"
            if len(failed_files) > 5:
                summary += f" (+{len(failed_files) - 5} more)"
        yield SSEvent(type=EventType.LOG, stage="code", content=summary)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage: Verify (build → fix → rebuild)
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_verify(self) -> AsyncGenerator[SSEvent, None]:
        total_files = len(self.ctx.get_written_files())
        yield SSEvent(type=EventType.LOG, stage="verify",
                       content=f"Starting verification for {total_files} files...")

        agent = ReactAgent(
            ctx=self.ctx,
            max_steps=self.config.max_react_steps,
            user_message=self.config.prompt,
            prd=self.ctx.prd,
            threat_model_text=self.ctx.threat_model.raw_text,
        )
        total_edits = 0

        for round_num in range(1, self.config.max_verify_rounds + 1):
            yield SSEvent(type=EventType.LOG, stage="verify",
                           content=f"Verify round {round_num}/{self.config.max_verify_rounds}")

            round_edits = 0
            try:
                async for event in agent.verify_project(
                    max_steps=self.config.max_react_steps,
                ):
                    event.stage = "verify"
                    yield event
                    if event.type == EventType.FILE_EDITED:
                        round_edits += 1
                        total_edits += 1
                        self.ctx.save_manifest()
            except Exception as e:
                yield SSEvent(type=EventType.WARN, stage="verify",
                               content=f"Verify round {round_num} error: {e}")

            # If no edits were needed, build was clean — stop
            if round_edits == 0:
                yield SSEvent(type=EventType.LOG, stage="verify",
                               content=f"Build clean on round {round_num}, no further fixes needed.")
                break

            yield SSEvent(type=EventType.LOG, stage="verify",
                           content=f"Round {round_num}: {round_edits} edit(s) applied, "
                                   f"re-verifying...")

        summary = f"Verification complete: {total_edits} total edit(s) across all rounds"
        yield SSEvent(type=EventType.LOG, stage="verify", content=summary)

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response. Handles code blocks, unbalanced braces, trailing commas."""
        # Strategy 1: code block
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: balanced brace finder
        start = text.find("{")
        if start != -1:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Fix trailing commas
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Could not extract JSON (len={len(text)}): {text[:300]}...")

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = str(path or "").strip().replace("\\", "/")
        if path in ("", ".", "./"):
            return "./"
        path = path.strip("/")
        if not path.startswith("./"):
            path = f"./{path}"
        return path

    @staticmethod
    def _parse_file_design_safe(raw: dict, batch: list[FileDesign]) -> FileDesign:
        """Parse raw JSON into FileDesign with aggressive type coercion."""
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Missing 'name'")
        path = Pipeline._normalize_path(raw.get("path", "./"))

        # Classes — per-class recovery
        classes = []
        for c in raw.get("classes", []):
            if not isinstance(c, dict):
                continue
            try:
                members = Pipeline._parse_parameters(c.get("members", []))
                methods = c.get("methods", [])
                if not isinstance(methods, list):
                    methods = [str(methods)]
                methods = [str(m) for m in methods]
                classes.append(ClassSpec(
                    class_name=str(c.get("class_name", "Unknown")),
                    members=members, methods=methods,
                ))
            except Exception:
                classes.append(ClassSpec(class_name=str(c.get("class_name", "Unknown"))))

        # Functions — per-function recovery
        functions = []
        for f in raw.get("functions", []):
            if not isinstance(f, dict):
                continue
            try:
                functions.append(FunctionSpec(
                    name=str(f.get("name", "unknown")),
                    input_parameters=Pipeline._parse_parameters(f.get("input_parameters", [])),
                    output_parameters=Pipeline._parse_parameters(f.get("output_parameters", [])),
                    description=str(f.get("description", "")),
                ))
            except Exception:
                functions.append(FunctionSpec(name=str(f.get("name", "unknown"))))

        deps = raw.get("dependencies", [])
        if not isinstance(deps, list):
            deps = [str(deps)] if deps else []
        deps = [str(d) for d in deps]

        for bf in batch:
            if bf.name == name:
                deps = deps or bf.dependencies
                break

        return FileDesign(
            name=name, path=path,
            description=str(raw.get("description", "")),
            classes=classes, functions=functions, dependencies=deps,
        )

    @staticmethod
    def _parse_parameters(params_raw) -> list[Parameter]:
        """Robustly parse parameters. Handles all weird LLM formats:
        
        Normal:  [{"name": "x", "type": "str"}]
        Dict:    {"x": "str", "y": "int"}
        Nested:  [{"name": "x", "type": {"a": "str"}}]  ← the bug you hit
        Bare:    ["x", "y"]
        None:    null
        """
        if params_raw is None:
            return []

        # Dict shorthand: {"arg1": "str", "arg2": "int"}
        if isinstance(params_raw, dict):
            return [
                Parameter(
                    name=str(k),
                    type=str(v) if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                )
                for k, v in params_raw.items()
            ]

        if not isinstance(params_raw, list):
            return []

        result = []
        for item in params_raw:
            if isinstance(item, str):
                result.append(Parameter(name=item, type="any"))
            elif isinstance(item, dict):
                name = str(item.get("name", ""))
                raw_type = item.get("type", "any")
                if isinstance(raw_type, dict):
                    type_str = json.dumps(raw_type, ensure_ascii=False)
                elif isinstance(raw_type, list):
                    type_str = ", ".join(str(t) for t in raw_type)
                elif raw_type is None:
                    type_str = "any"
                else:
                    type_str = str(raw_type)
                result.append(Parameter(name=name, type=type_str))
        return result