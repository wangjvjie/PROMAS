from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agents import CodeWorkerAgent, SystemDesignConsistencyAgent, ThreatModelAgent
from .api import LLMClient
from .memory import HashEmbeddingProvider, SharedMemoryPool
from .models import CodeTask, ProjectState, TaskStatus
from .prompts import (
    COMPILE_ERROR_TASKS_PROMPT,
    GLOBAL_PROJECT_REVIEW_PROMPT,
    TASK_PLANNING_PROMPT,
    WRITE_FILE_DESIGN_PROMPT,
    WRITE_PRD_PROMPT,
    WRITE_README_PROMPT,
    WRITE_SYSTEM_DESIGN_PROMPT,
)
from .scheduler import ParallelTaskScheduler
from .utils import dump_json, extract_json_object, safe_rel_path
from .workspace import Workspace


class _LLMEmbeddingProvider:
    """Optional embedding provider backed by model API."""

    def __init__(self, llm: LLMClient, dim: int = 1536) -> None:
        self.llm = llm
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        try:
            emb = self.llm.embed(text[:8000])
            if emb:
                return emb
        except Exception:
            pass
        return [0.0] * self.dim


class ParallelMASecDev:
    def __init__(
        self,
        *,
        work_dir: str = "./promas-parallel-workspace",
        threat_model_mode: str = "call_chain",
        threat_model_modular: bool = True,
        threat_module_max_files: int = 12,
        threat_model_workers: int = 4,
        threat_context_max_files: int = 32,
        code_workers: int = 4,
        system_design_workers: int = 4,
        reuse_existing_arch_design: bool = True,
        enable_system_design_consistency_check: bool = True,
        auto_apply_consistency_fixes: bool = True,
        consistency_max_updates: int = 20,
        enable_global_code_check: bool = True,
        global_code_check_rounds: int = 2,
        enable_global_project_review: bool = True,
        enable_compile_fix_loop: bool = True,
        compile_fix_rounds: int = 3,
        compile_timeout_seconds: int = 240,
        use_llm_embeddings: bool = False,
        llm_client: LLMClient | None = None,
        feedback_callback: Callable[[dict], None] | None = None,
        persist_feedback: bool = True,
    ) -> None:
        self.workspace = Workspace(work_dir)
        self.llm = llm_client or LLMClient()
        provider = _LLMEmbeddingProvider(self.llm) if use_llm_embeddings else HashEmbeddingProvider()
        self.memory = SharedMemoryPool(provider)

        self.state = ProjectState()
        self.threat_model_mode = threat_model_mode
        self.threat_model_modular = threat_model_modular
        self.threat_module_max_files = max(1, threat_module_max_files)
        self.threat_model_workers = max(1, threat_model_workers)
        self.threat_context_max_files = max(8, threat_context_max_files)
        self.code_workers = max(1, code_workers)
        self.system_design_workers = max(1, system_design_workers)
        self.reuse_existing_arch_design = reuse_existing_arch_design
        self.enable_system_design_consistency_check = enable_system_design_consistency_check
        self.auto_apply_consistency_fixes = auto_apply_consistency_fixes
        self.consistency_max_updates = max(0, consistency_max_updates)
        self.enable_global_code_check = enable_global_code_check
        self.global_code_check_rounds = max(1, global_code_check_rounds)
        self.enable_global_project_review = enable_global_project_review
        self.enable_compile_fix_loop = enable_compile_fix_loop
        self.compile_fix_rounds = max(1, compile_fix_rounds)
        self.compile_timeout_seconds = max(30, compile_timeout_seconds)
        self.feedback_callback = feedback_callback
        self.persist_feedback = persist_feedback
        self._threat_module_lock = threading.Lock()
        self._live_threat_modules: dict[str, dict[str, Any]] = {}

    @property
    def work_dir(self) -> str:
        return self.workspace.os_path()

    def run(self, user_msg: str, begin_stage: str = "auto") -> None:
        self.state.user_msg = user_msg
        self.load_existing_state()
        self.refresh_workspace_manifest()

        stage_order = ["prd", "system_design", "threat_model", "code", "readme"]
        requested_begin_stage = begin_stage
        if begin_stage == "auto":
            begin_stage = self._infer_begin_stage()
            self._emit_feedback(
                "begin_stage_auto_selected",
                "begin stage auto selected",
                requested_stage=requested_begin_stage,
                selected_stage=begin_stage,
            )

        if begin_stage not in stage_order:
            raise ValueError(f"Invalid begin_stage: {begin_stage}")

        self._emit_feedback(
            "run_started",
            "parallel run started",
            begin_stage=begin_stage,
            requested_stage=requested_begin_stage,
            workspace=self.work_dir,
            existing_files=len(self.state.workspace_manifest),
        )

        for stage in stage_order[stage_order.index(begin_stage) :]:
            self._emit_feedback("stage_started", f"stage {stage} started", stage=stage)
            try:
                if stage == "prd":
                    self.write_prd()
                elif stage == "system_design":
                    if not self.state.prd:
                        raise RuntimeError("PRD is empty but system_design requested")
                    self.write_system_design()
                elif stage == "threat_model":
                    if not self.state.arch.get("files"):
                        raise RuntimeError("Architecture is empty but threat_model requested")
                    self.write_threat_modeling()
                elif stage == "code":
                    if not self.state.threat_model:
                        raise RuntimeError("Threat_model is empty but code generation requested")
                    self.write_code_parallel()
                elif stage == "readme":
                    self.write_readme()
            except Exception as exc:
                self._emit_feedback("stage_failed", f"stage {stage} failed", stage=stage, error=str(exc))
                raise
            else:
                self._emit_feedback("stage_done", f"stage {stage} completed", stage=stage)

        self._emit_feedback("run_done", "parallel run completed")

    def _infer_begin_stage(self) -> str:
        if not self.state.prd.strip():
            return "prd"

        arch_files = self._arch_file_paths()
        if not arch_files:
            return "system_design"

        if not self.state.threat_model.strip():
            return "threat_model"

        existing_files = {
            safe_rel_path(item["path"])
            for item in self.state.workspace_manifest
            if item.get("path")
        }
        missing_arch_files = [rel for rel in sorted(arch_files) if rel not in existing_files]
        if missing_arch_files:
            return "code"

        if not self.state.code:
            return "code"

        if not self.workspace.exists("README.md"):
            return "readme"

        # Fully built workspace: resume from code so new requirements can be applied incrementally.
        return "code"

    def write_prd(self) -> None:
        prompt = WRITE_PRD_PROMPT.format(msg=self.state.user_msg)
        self.state.prd = self.llm.chat(prompt)
        self.memory.add("prd", self.state.prd, {"kind": "prd"})
        self._persist_state_files()

    def write_system_design(self) -> None:
        file_design_prompt = WRITE_FILE_DESIGN_PROMPT.format(msg=self.state.user_msg, prd=self.state.prd)
        file_design_response = self.llm.chat(file_design_prompt)
        files_obj = self._parse_file_design_response(file_design_response, file_design_prompt)
        files = files_obj.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("File design response missing files")

        old_arch_files = list(self.state.arch.get("files", []))
        old_arch_map = {self._arch_file_key(item): item for item in old_arch_files}
        self.state.arch = {"files": []}

        progress = self._init_arch_progress(files)
        self._persist_arch_progress(progress)

        ordered_keys: list[str] = []
        designed_map: dict[str, dict[str, Any]] = {}
        pending: list[tuple[dict[str, Any], str]] = []

        for file in files:
            file_name = str(file.get("name", "")).strip()
            file_path = str(file.get("path", "./")).strip()
            key = self._arch_file_key({"name": file_name, "path": file_path})
            ordered_keys.append(key)

            reused = self.reuse_existing_arch_design and key in old_arch_map
            if reused:
                designed = dict(old_arch_map[key])
                if "name" not in designed:
                    designed["name"] = file_name
                if "path" not in designed:
                    designed["path"] = file_path
                designed_map[key] = designed
                self._mark_arch_progress(progress, key=key, state="reused")
                self.state.arch["files"] = self._ordered_arch_files(ordered_keys, designed_map)
                self._persist_state_files()
                self._persist_arch_progress(progress)
                self._emit_feedback(
                    "system_design_progress",
                    "system design file reused",
                    file=f"{file_path}/{file_name}" if file_path not in {"", ".", "./"} else file_name,
                    completed=progress["completed"],
                    total=progress["total"],
                    remaining=progress["remaining"],
                    reused=progress["reused"],
                    failed=progress["failed"],
                )
            else:
                pending.append((file, key))

        pending.sort(key=lambda x: self._system_design_priority(x[0]))

        def _design_one(file_item: dict[str, Any], history_for_prompt: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
            file_name = str(file_item.get("name", "")).strip()
            file_path = str(file_item.get("path", "./")).strip()
            display = f"### File Name:{file_name}\n### File Path:{file_path}"
            prompt = WRITE_SYSTEM_DESIGN_PROMPT.format(
                msg=self.state.user_msg,
                prd=self.state.prd,
                api=dump_json(history_for_prompt),
                files=dump_json({"files": files}),
                file_name=display,
            )
            response = self.llm.chat(prompt)
            try:
                designed = extract_json_object(response)
                error: str | None = None
            except Exception as exc:
                designed = {
                    "name": file_name,
                    "path": file_path,
                    "classes": [],
                    "functions": [],
                    "description": file_item.get("description", ""),
                }
                error = str(exc)

            if "name" not in designed:
                designed["name"] = file_name
            if "path" not in designed:
                designed["path"] = file_path
            return designed, error

        if pending:
            wave_idx = 0
            while pending:
                wave_idx += 1
                wave = pending[: self.system_design_workers]
                pending = pending[self.system_design_workers :]
                self._emit_feedback(
                    "system_design_wave_started",
                    "system design wave started",
                    wave=wave_idx,
                    batch_size=len(wave),
                    remaining_batches=((len(pending) + self.system_design_workers - 1) // self.system_design_workers),
                )

                with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                    futures = {
                        executor.submit(
                            _design_one,
                            file_item,
                            self._select_system_design_history_context(file_item, designed_map, ordered_keys),
                        ): (file_item, key)
                        for file_item, key in wave
                    }
                    for future in as_completed(futures):
                        file_item, key = futures[future]
                        file_name = str(file_item.get("name", "")).strip()
                        file_path = str(file_item.get("path", "./")).strip()
                        pretty = f"{file_path}/{file_name}" if file_path not in {"", ".", "./"} else file_name
                        try:
                            designed, error = future.result()
                        except Exception as exc:
                            designed = {
                                "name": file_name,
                                "path": file_path,
                                "classes": [],
                                "functions": [],
                                "description": file_item.get("description", ""),
                            }
                            error = str(exc)

                        designed_map[key] = designed
                        if error:
                            self._mark_arch_progress(progress, key=key, state="failed", error=error)
                        else:
                            self._mark_arch_progress(progress, key=key, state="done")

                        self.state.arch["files"] = self._ordered_arch_files(ordered_keys, designed_map)
                        self._persist_state_files()
                        self._persist_arch_progress(progress)
                        self._emit_feedback(
                            "system_design_progress",
                            "system design file completed",
                            file=pretty,
                            status="failed" if error else "done",
                            error=error or "",
                            wave=wave_idx,
                            completed=progress["completed"],
                            total=progress["total"],
                            remaining=progress["remaining"],
                            reused=progress["reused"],
                            failed=progress["failed"],
                        )

        progress["status"] = "done" if progress["remaining"] == 0 else "partial"
        progress["updated_at"] = self._utc_now()
        self._persist_arch_progress(progress)

        if self.enable_system_design_consistency_check:
            self.run_system_design_consistency_check()

        self.memory.add("arch", dump_json(self.state.arch), {"kind": "architecture"})
        self._persist_state_files()

    def write_threat_modeling(self) -> None:
        with self._threat_module_lock:
            self._live_threat_modules = {}
            self.workspace.save_state_file(
                "threat_model_modules.json",
                dump_json(
                    {
                        "status": "running",
                        "total": 0,
                        "completed": 0,
                        "running": 0,
                        "modules": [],
                        "updated_at": self._utc_now(),
                    }
                ),
            )

        tm_agent = ThreatModelAgent(
            self.llm,
            self.state,
            mode=self.threat_model_mode,
            modular=self.threat_model_modular,
            module_max_files=self.threat_module_max_files,
            max_workers=self.threat_model_workers,
            context_max_files=self.threat_context_max_files,
            event_callback=self._on_threat_event,
        )
        threat_text, module_reports = tm_agent.generate_with_modules()
        self.state.threat_model = threat_text
        self.state.threat_model_modules = module_reports
        with self._threat_module_lock:
            self._live_threat_modules = {
                f"{int(m.get('index', i+1))}:{str(m.get('module', ''))}": {
                    **m,
                    "status": "done",
                }
                for i, m in enumerate(module_reports)
                if isinstance(m, dict)
            }
        self.state.threat_model_json = tm_agent.to_structured_json(self.state.threat_model)
        self.memory.add("threat", self.state.threat_model, {"kind": "threat_model"})
        if module_reports:
            self.memory.add("threat_modules", dump_json({"modules": module_reports}), {"kind": "threat_modules"})
        self.memory.add("threat_json", dump_json(self.state.threat_model_json), {"kind": "threat_model_json"})
        self._persist_state_files()

    def run_system_design_consistency_check(self) -> None:
        self._emit_feedback(
            "system_design_consistency_started",
            "system design consistency check started",
            file_count=len(self.state.arch.get("files", [])),
        )
        reviewer = SystemDesignConsistencyAgent(self.llm, self.state)
        try:
            report = reviewer.review()
        except Exception as exc:
            self._emit_feedback(
                "system_design_consistency_failed",
                "system design consistency check failed",
                error=str(exc),
            )
            return

        updates = report.get("file_updates", [])
        issues = report.get("issues", [])
        if not isinstance(updates, list):
            updates = []
        if not isinstance(issues, list):
            issues = []

        applied = 0
        if self.auto_apply_consistency_fixes and updates:
            applied = self._apply_arch_consistency_updates(updates[: self.consistency_max_updates])

        output = {
            "summary": report.get("summary", ""),
            "consistency_score": report.get("consistency_score"),
            "issue_count": len(issues),
            "issues": issues,
            "suggested_update_count": len(updates),
            "applied_update_count": applied,
            "auto_applied": self.auto_apply_consistency_fixes,
        }
        self.workspace.save_state_file("arch.consistency.json", dump_json(output))
        self._emit_feedback(
            "system_design_consistency_done",
            "system design consistency check completed",
            issue_count=len(issues),
            suggested_update_count=len(updates),
            applied_update_count=applied,
            consistency_score=report.get("consistency_score"),
        )

    def _apply_arch_consistency_updates(self, updates: list[dict[str, Any]]) -> int:
        current_files = list(self.state.arch.get("files", []))
        by_key = {self._arch_file_key(item): item for item in current_files}
        applied = 0

        for upd in updates:
            if not isinstance(upd, dict):
                continue
            replace_design = upd.get("replace_design")
            if not isinstance(replace_design, dict):
                continue

            file_ref = str(upd.get("file", "")).strip()
            target_key = ""
            if file_ref:
                try:
                    target_key = safe_rel_path(file_ref)
                except Exception:
                    target_key = ""
            if not target_key:
                try:
                    target_key = self._arch_file_key(replace_design)
                except Exception:
                    target_key = ""
            if not target_key:
                continue
            if target_key not in by_key:
                continue

            base = dict(by_key[target_key])
            base.update(replace_design)
            if "name" not in base or "path" not in base:
                continue
            by_key[target_key] = base
            applied += 1

        if applied:
            ordered: list[dict[str, Any]] = []
            for item in current_files:
                key = self._arch_file_key(item)
                if key in by_key:
                    ordered.append(by_key[key])
            self.state.arch["files"] = ordered
            self._persist_state_files()
        return applied

    def write_code_parallel(self) -> None:
        self.refresh_workspace_manifest()
        self._preload_memory_with_workspace_files()
        tasks = self.plan_code_tasks()

        self._emit_feedback(
            "planning_done",
            "task planning completed",
            task_count=len(tasks),
            tasks=[{"task_id": t.task_id, "file": t.file, "mode": t.mode, "source": t.source} for t in tasks],
        )

        scheduler = ParallelTaskScheduler(
            tasks,
            worker_count=self.code_workers,
            agent_factory=self._build_worker,
            max_attempts=4,
            on_event=self._on_scheduler_event,
        )
        scheduler.run()

        failures = [t for t in scheduler.tasks.values() if t.status == TaskStatus.FAILED]

        if failures:
            failed = "; ".join(f"{t.task_id}:{t.file}:{t.last_error}" for t in failures)
            raise RuntimeError(f"Some code tasks failed: {failed}")

        if self.enable_global_project_review:
            # Keep only one final LLM gate: whole-project review.
            self.run_global_project_review_and_fix()
        else:
            if self.enable_global_code_check:
                self.run_global_code_check_and_fix()
            if self.enable_compile_fix_loop:
                self.run_compile_fix_loop()

        self.refresh_workspace_manifest()
        self.state.code = self.workspace.load_code_blocks()
        self._persist_state_files()

    def plan_code_tasks(self) -> list[CodeTask]:
        arch_files = self._arch_file_paths()
        existing_files = {safe_rel_path(item["path"]) for item in self.state.workspace_manifest if item.get("path")}
        planning_scope = set(arch_files) | set(existing_files)

        prompt = TASK_PLANNING_PROMPT.format(
            msg=self.state.user_msg,
            api=dump_json(self._compact_arch_for_planning()),
            threat=dump_json(self._compact_threat_for_planning()),
            workspace_manifest=dump_json({"files": self.state.workspace_manifest}),
        )

        planned_tasks: list[CodeTask] = []
        task_by_id: dict[str, CodeTask] = {}

        try:
            response = self.llm.chat(prompt)
            obj = extract_json_object(response)
            raw_tasks = obj.get("tasks", [])
        except Exception:
            raw_tasks = []

        for i, raw in enumerate(raw_tasks):
            file_path = safe_rel_path(str(raw.get("file", ""))) if raw.get("file") else ""
            if file_path not in planning_scope:
                continue
            task_id = str(raw.get("task_id") or f"T{i+1}")
            if task_id in task_by_id:
                continue
            depends = {str(d) for d in raw.get("depends_on", []) if isinstance(d, str)}
            mode = str(raw.get("mode", "update_or_create"))
            if mode not in {"create_only", "update_only", "update_or_create"}:
                mode = "update_or_create"
            source = str(raw.get("source", "planned"))
            task = CodeTask(
                task_id=task_id,
                file=file_path,
                goal=str(raw.get("goal", f"Implement {file_path}")),
                depends_on=depends,
                priority=int(raw.get("priority", i + 1)),
                mode=mode,
                source=source,
            )
            planned_tasks.append(task)
            task_by_id[task_id] = task

        planned_files = {t.file for t in planned_tasks}

        # Ensure every architecture file has work assigned.
        for i, rel in enumerate(sorted(arch_files)):
            if rel in planned_files:
                continue
            exists = rel in existing_files
            task_id = f"AUTO_ARCH_{i+1}"
            planned_tasks.append(
                CodeTask(
                    task_id=task_id,
                    file=rel,
                    goal=(
                        f"Refactor and align existing {rel} with architecture, threat model, and user requirement"
                        if exists
                        else f"Create and implement {rel} according to architecture and threat model"
                    ),
                    depends_on=set(),
                    priority=100 + i,
                    mode="update_or_create" if exists else "create_only",
                    source="arch",
                )
            )

        # If no architecture exists, still support incremental refactor on existing workspace.
        if not arch_files and existing_files and not planned_tasks:
            for i, rel in enumerate(sorted(existing_files)):
                planned_tasks.append(
                    CodeTask(
                        task_id=f"AUTO_EXIST_{i+1}",
                        file=rel,
                        goal=f"Incrementally improve {rel} to satisfy user requirement while preserving behavior",
                        depends_on=set(),
                        priority=200 + i,
                        mode="update_only",
                        source="existing",
                    )
                )

        valid_ids = {t.task_id for t in planned_tasks}
        for task in planned_tasks:
            task.depends_on = {d for d in task.depends_on if d in valid_ids and d != task.task_id}

        def _prio_boost(path: str) -> int:
            low = path.lower()
            if low.endswith(("requirements.txt", "pyproject.toml", "package.json", "dockerfile")):
                return -50
            if low.endswith(("main.py", "app.py", "server.py", "index.ts", "index.js")):
                return -20
            return 0

        for task in planned_tasks:
            task.priority = task.priority + _prio_boost(task.file)

        planned_tasks.sort(key=lambda t: (t.priority, t.task_id))
        return planned_tasks

    def run_global_code_check_and_fix(self) -> None:
        code_files = sorted(self.workspace.list_code_files())
        if not code_files:
            self._emit_feedback(
                "global_code_check_done",
                "global code check completed",
                rounds=0,
                file_count=0,
                issue_files=0,
                fixed_files=0,
                unresolved_files=0,
            )
            return

        checker = self._build_worker("global-checker")
        report: dict[str, Any] = {
            "rounds": [],
            "max_rounds": self.global_code_check_rounds,
            "file_count": len(code_files),
        }

        total_issue_files = 0
        total_fixed_files = 0
        unresolved_files: set[str] = set()

        self._emit_feedback(
            "global_code_check_started",
            "global code check started",
            file_count=len(code_files),
            max_rounds=self.global_code_check_rounds,
        )

        for round_idx in range(1, self.global_code_check_rounds + 1):
            issue_files = 0
            fixed_files = 0
            unresolved_this_round: list[dict[str, Any]] = []

            self._emit_feedback(
                "global_code_check_round_started",
                "global code check round started",
                round=round_idx,
                file_count=len(code_files),
            )

            for i, rel in enumerate(code_files, start=1):
                task = CodeTask(
                    task_id=f"GLOBAL_R{round_idx}_F{i}",
                    file=rel,
                    goal="Resolve global compile/linkage/consistency issues with minimal edits",
                    mode="update_only",
                    source="global_check",
                )
                deps = self._select_global_check_dependency_files(rel, code_files)
                check = checker.run_file_check(task, dependency_files=deps, deep=True)
                if not check.get("needs_fix"):
                    continue

                issue_files += 1
                total_issue_files += 1
                self._emit_feedback(
                    "global_code_check_issue",
                    "global code issue detected",
                    round=round_idx,
                    file=rel,
                    issue_count=len(check.get("issues", [])),
                    summary=check.get("short_summary", ""),
                )

                memory_ctx = [
                    str(check.get("report", ""))[:3000],
                    "Global pass: keep edits minimal and compile-oriented.",
                ]
                edit_result = checker.run_file_edit(task, dependency_files=deps, memory_snippets=memory_ctx)
                if edit_result.status != TaskStatus.DONE:
                    unresolved_this_round.append(
                        {
                            "file": rel,
                            "reason": edit_result.summary or "edit failed",
                        }
                    )
                    unresolved_files.add(rel)
                    self._emit_feedback(
                        "global_code_fix_failed",
                        "global code fix failed",
                        round=round_idx,
                        file=rel,
                        reason=edit_result.summary,
                    )
                    continue

                post = checker.run_file_check(task, dependency_files=deps, deep=False)
                if post.get("needs_fix"):
                    unresolved_this_round.append(
                        {
                            "file": rel,
                            "reason": post.get("short_summary", "still has issues"),
                        }
                    )
                    unresolved_files.add(rel)
                    self._emit_feedback(
                        "global_code_fix_partial",
                        "global code fix incomplete",
                        round=round_idx,
                        file=rel,
                        issue_count=len(post.get("issues", [])),
                        summary=post.get("short_summary", ""),
                    )
                    continue

                fixed_files += 1
                total_fixed_files += 1
                if rel in unresolved_files:
                    unresolved_files.discard(rel)
                self._emit_feedback(
                    "global_code_fix_done",
                    "global code fix applied",
                    round=round_idx,
                    file=rel,
                )

            round_payload = {
                "round": round_idx,
                "issue_files": issue_files,
                "fixed_files": fixed_files,
                "unresolved": unresolved_this_round,
            }
            report["rounds"].append(round_payload)
            self._emit_feedback(
                "global_code_check_round_done",
                "global code check round completed",
                round=round_idx,
                issue_files=issue_files,
                fixed_files=fixed_files,
                unresolved_files=len(unresolved_this_round),
            )

            if issue_files == 0:
                break
            if fixed_files == 0:
                # no progress in this round; stop to avoid useless loops
                break

            code_files = sorted(self.workspace.list_code_files())

        report["issue_files"] = total_issue_files
        report["fixed_files"] = total_fixed_files
        report["unresolved_files"] = sorted(unresolved_files)
        report["completed_at"] = self._utc_now()
        self.workspace.save_state_file("code.global_check.json", dump_json(report))
        self._emit_feedback(
            "global_code_check_done",
            "global code check completed",
            rounds=len(report["rounds"]),
            file_count=report["file_count"],
            issue_files=total_issue_files,
            fixed_files=total_fixed_files,
            unresolved_files=len(unresolved_files),
        )

    def run_global_project_review_and_fix(self) -> None:
        review_files = self._list_review_files(max_files=64)
        if not review_files:
            self._emit_feedback(
                "global_project_review_done",
                "global project review completed",
                requested_fixes=0,
                applied_fixes=0,
                failed_fixes=0,
                skipped=True,
            )
            return

        self._emit_feedback(
            "global_project_review_started",
            "global project review started",
            file_count=len(review_files),
        )
        prompt = GLOBAL_PROJECT_REVIEW_PROMPT.format(
            msg=self.state.user_msg,
            api=dump_json(self._compact_arch_for_planning()),
            threat=dump_json(self._compact_threat_for_planning()),
            workspace_manifest=dump_json({"files": self.state.workspace_manifest}),
            file_snapshots=self._build_file_snapshots(review_files, max_chars_per_file=1400),
        )

        summary = ""
        requested = 0
        plan_fixes: list[dict[str, Any]] = []
        parse_error = ""
        try:
            raw = self.llm.chat(prompt, temperature=0.1)
            obj = extract_json_object(raw)
            summary = str(obj.get("summary", ""))
            fixes = obj.get("priority_fixes", [])
            if isinstance(fixes, list):
                plan_fixes = [x for x in fixes if isinstance(x, dict)]
        except Exception as exc:
            parse_error = str(exc)
            plan_fixes = []

        requested = len(plan_fixes)
        fix_result = self._apply_llm_fix_tasks(plan_fixes, source="global_review", round_idx=1)
        output = {
            "summary": summary,
            "requested_fixes": requested,
            "applied_fixes": fix_result["applied_count"],
            "failed_fixes": len(fix_result["failed"]),
            "failed": fix_result["failed"],
            "parse_error": parse_error,
            "completed_at": self._utc_now(),
        }
        self.workspace.save_state_file("code.global_review.json", dump_json(output))
        self._emit_feedback(
            "global_project_review_done",
            "global project review completed",
            requested_fixes=requested,
            applied_fixes=fix_result["applied_count"],
            failed_fixes=len(fix_result["failed"]),
            parse_error=parse_error,
        )

    def run_compile_fix_loop(self) -> None:
        targets = self._detect_compile_targets()
        if not targets:
            output = {
                "status": "skipped",
                "reason": "no supported compile target detected",
                "max_rounds": self.compile_fix_rounds,
                "rounds": [],
                "completed_at": self._utc_now(),
            }
            self.workspace.save_state_file("code.compile_check.json", dump_json(output))
            self._emit_feedback(
                "compile_fix_done",
                "compile fix loop completed",
                status="skipped",
                rounds=0,
                unresolved_targets=0,
            )
            return

        self._emit_feedback(
            "compile_fix_started",
            "compile fix loop started",
            target_count=len(targets),
            max_rounds=self.compile_fix_rounds,
        )
        report: dict[str, Any] = {
            "status": "running",
            "targets": [{"label": t["label"], "command": t["command"]} for t in targets],
            "max_rounds": self.compile_fix_rounds,
            "rounds": [],
            "final_failures": [],
        }

        success = False
        final_failures: list[dict[str, Any]] = []
        applied_in_last_round = False

        for round_idx in range(1, self.compile_fix_rounds + 1):
            failures = self._run_compile_targets_once(targets, round_idx=round_idx, phase="main")
            if not failures:
                success = True
                report["rounds"].append(
                    {
                        "round": round_idx,
                        "failed_targets": 0,
                        "fix_tasks": 0,
                        "applied_fixes": 0,
                        "failed_fixes": 0,
                    }
                )
                break

            compile_output = self._merge_compile_failures(failures, max_chars=18000)
            planned_tasks = self._plan_compile_fix_tasks(compile_output)
            fix_result = self._apply_llm_fix_tasks(planned_tasks, source="compile_fix", round_idx=round_idx)
            applied_in_last_round = fix_result["applied_count"] > 0
            round_payload = {
                "round": round_idx,
                "failed_targets": len(failures),
                "first_failure": failures[0]["label"] if failures else "",
                "fix_tasks": len(planned_tasks),
                "applied_fixes": fix_result["applied_count"],
                "failed_fixes": len(fix_result["failed"]),
                "failed_fix_items": fix_result["failed"],
                "planning_mode": "llm_global",
            }
            report["rounds"].append(round_payload)
            if not applied_in_last_round:
                final_failures = failures
                break

        if not success and applied_in_last_round:
            verify_failures = self._run_compile_targets_once(targets, round_idx=self.compile_fix_rounds + 1, phase="verify")
            if not verify_failures:
                success = True
            else:
                final_failures = verify_failures

        if success:
            report["status"] = "success"
            report["final_failures"] = []
        else:
            if not final_failures:
                final_failures = self._run_compile_targets_once(
                    targets,
                    round_idx=self.compile_fix_rounds + 1,
                    phase="final_check",
                )
            report["status"] = "failed"
            report["final_failures"] = [
                {"label": x["label"], "returncode": x["returncode"], "output_preview": x["output"][:1200]}
                for x in final_failures
            ]

        report["completed_at"] = self._utc_now()
        self.workspace.save_state_file("code.compile_check.json", dump_json(report))
        self._emit_feedback(
            "compile_fix_done",
            "compile fix loop completed",
            status=report["status"],
            rounds=len(report["rounds"]),
            unresolved_targets=len(report["final_failures"]),
        )

        if report["status"] != "success":
            first = report["final_failures"][0] if report["final_failures"] else {}
            label = str(first.get("label", "unknown-target"))
            detail = str(first.get("output_preview", "compile failed"))[:500]
            raise RuntimeError(f"Compile gate failed on {label}: {detail}")

    def _apply_llm_fix_tasks(
        self,
        raw_tasks: list[dict[str, Any]],
        *,
        source: str,
        round_idx: int,
    ) -> dict[str, Any]:
        if not raw_tasks:
            return {"applied_count": 0, "failed": []}

        checker = self._build_worker(f"{source}-r{round_idx}")
        code_files = sorted(self.workspace.list_code_files())
        applied_count = 0
        failed: list[dict[str, Any]] = []

        for i, raw in enumerate(raw_tasks[:30], start=1):
            file_path = str(raw.get("file", "")).strip()
            if not file_path:
                continue
            try:
                rel = safe_rel_path(file_path)
            except Exception:
                continue

            mode = self._normalize_fix_mode(raw.get("mode"), file_path=rel)
            goal = str(raw.get("goal") or raw.get("reason") or f"Fix compile/runtime issue in {rel}").strip()
            if not goal:
                goal = f"Fix compile/runtime issue in {rel}"
            task = CodeTask(
                task_id=f"{source.upper()}_R{round_idx}_T{i}",
                file=rel,
                goal=goal,
                mode=mode,
                source=source,
            )
            deps = self._select_global_check_dependency_files(rel, code_files)
            self._emit_feedback(
                "llm_fix_task_started",
                "llm fix task started",
                source=source,
                round=round_idx,
                task_id=task.task_id,
                file=rel,
                mode=mode,
            )

            try:
                result = checker.execute(task, dependency_files=deps, completed_files=code_files)
            except Exception as exc:
                failed.append({"file": rel, "reason": str(exc), "mode": mode})
                self._emit_feedback(
                    "llm_fix_task_failed",
                    "llm fix task failed",
                    source=source,
                    round=round_idx,
                    task_id=task.task_id,
                    file=rel,
                    reason=str(exc),
                )
                continue

            if result.status != TaskStatus.DONE:
                reason = result.summary or result.wait_reason or "worker returned non-done status"
                failed.append({"file": rel, "reason": reason, "mode": mode})
                self._emit_feedback(
                    "llm_fix_task_failed",
                    "llm fix task failed",
                    source=source,
                    round=round_idx,
                    task_id=task.task_id,
                    file=rel,
                    reason=reason,
                )
                continue

            applied_count += 1
            code_files = sorted(self.workspace.list_code_files())
            self._emit_feedback(
                "llm_fix_task_done",
                "llm fix task completed",
                source=source,
                round=round_idx,
                task_id=task.task_id,
                file=rel,
            )

        self.refresh_workspace_manifest()
        return {
            "applied_count": applied_count,
            "failed": failed,
        }

    def _normalize_fix_mode(self, mode: Any, *, file_path: str) -> str:
        mode_text = str(mode or "").strip().lower()
        exists = self.workspace.exists(file_path)
        if mode_text not in {"create_only", "update_only", "update_or_create"}:
            return "update_only" if exists else "update_or_create"
        if mode_text == "create_only" and exists:
            return "update_only"
        if mode_text == "update_only" and not exists:
            return "update_or_create"
        return mode_text

    def _detect_compile_targets(self) -> list[dict[str, Any]]:
        root = Path(self.workspace.os_path())

        custom_targets = self._load_custom_compile_targets(root)
        if custom_targets:
            return custom_targets

        discovered: list[dict[str, Any]] = []
        discovered.extend(self._detect_jvm_compile_targets(root))
        discovered.extend(self._detect_js_compile_targets(root))
        discovered.extend(self._detect_python_compile_targets(root))
        discovered.extend(self._detect_go_compile_targets(root))
        discovered.extend(self._detect_rust_compile_targets(root))
        discovered.extend(self._detect_dotnet_compile_targets(root))

        # Deduplicate while preserving order.
        dedup: list[dict[str, Any]] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for t in discovered:
            cmd = [str(x).strip() for x in t.get("command", []) if str(x).strip()]
            cwd = str(t.get("cwd", self.workspace.os_path()))
            label = str(t.get("label", "compile-target"))
            if not cmd:
                continue
            key = (label, cwd, tuple(cmd))
            if key in seen:
                continue
            seen.add(key)
            dedup.append({"label": label, "cwd": cwd, "command": cmd})
        return dedup[:12]

    def _load_custom_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        config_names = [".promas_compile_targets.json", "compile.targets.json"]
        for name in config_names:
            cfg = root / name
            if not cfg.exists():
                continue
            try:
                obj = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception as exc:
                self._emit_feedback(
                    "compile_target_config_invalid",
                    "compile target config parse failed",
                    file=name,
                    error=str(exc),
                )
                return []

            items = obj.get("targets", []) if isinstance(obj, dict) else []
            if not isinstance(items, list):
                self._emit_feedback(
                    "compile_target_config_invalid",
                    "compile target config has invalid schema",
                    file=name,
                )
                return []

            out: list[dict[str, Any]] = []
            for idx, item in enumerate(items[:12], start=1):
                if not isinstance(item, dict):
                    continue
                raw_cmd = item.get("command")
                cmd: list[str] = []
                if isinstance(raw_cmd, list):
                    cmd = [str(x).strip() for x in raw_cmd if str(x).strip()]
                elif isinstance(raw_cmd, str):
                    try:
                        cmd = [x for x in shlex.split(raw_cmd) if x]
                    except Exception:
                        cmd = []
                if not cmd:
                    continue

                raw_cwd = str(item.get("cwd", ".")).strip() or "."
                try:
                    abs_cwd = (root / raw_cwd).resolve()
                    if not str(abs_cwd).startswith(str(root)):
                        continue
                except Exception:
                    continue
                label = str(item.get("label") or f"custom:{idx}").strip() or f"custom:{idx}"
                out.append(
                    {
                        "label": label,
                        "cwd": str(abs_cwd),
                        "command": cmd,
                    }
                )

            if out:
                self._emit_feedback(
                    "compile_target_config_loaded",
                    "custom compile targets loaded",
                    file=name,
                    target_count=len(out),
                )
            return out
        return []

    def _detect_jvm_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []

        root_mvnw = root / "mvnw"
        mvn_bin = shutil.which("mvn")
        root_pom = root / "pom.xml"

        if root_pom.exists():
            if root_mvnw.exists():
                targets.append(
                    {
                        "label": "maven:root",
                        "cwd": str(root),
                        "command": ["bash", "./mvnw", "-DskipTests", "compile"],
                    }
                )
            elif mvn_bin:
                targets.append(
                    {
                        "label": "maven:root",
                        "cwd": str(root),
                        "command": [mvn_bin, "-DskipTests", "compile"],
                    }
                )
            return targets

        pom_files = sorted(
            p for p in root.rglob("pom.xml")
            if not self._is_ignored_project_path(p)
        )
        if pom_files:
            for pom in pom_files[:4]:
                rel_pom = str(pom.relative_to(root)).replace("\\", "/")
                if root_mvnw.exists():
                    cmd = ["bash", "./mvnw", "-f", rel_pom, "-DskipTests", "compile"]
                    cwd = str(root)
                elif mvn_bin:
                    cmd = [mvn_bin, "-f", rel_pom, "-DskipTests", "compile"]
                    cwd = str(root)
                else:
                    break
                targets.append(
                    {
                        "label": f"maven:{rel_pom}",
                        "cwd": cwd,
                        "command": cmd,
                    }
                )
            if targets:
                return targets

        root_gradlew = root / "gradlew"
        gradle_bin = shutil.which("gradle")
        if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            if root_gradlew.exists():
                targets.append(
                    {
                        "label": "gradle:root",
                        "cwd": str(root),
                        "command": ["bash", "./gradlew", "classes", "--no-daemon"],
                    }
                )
            elif gradle_bin:
                targets.append(
                    {
                        "label": "gradle:root",
                        "cwd": str(root),
                        "command": [gradle_bin, "classes", "--no-daemon"],
                    }
                )
        return targets

    def _detect_js_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        package_json_files = sorted(
            p for p in root.rglob("package.json")
            if not self._is_ignored_project_path(p)
        )
        if not package_json_files:
            return targets

        for pkg_json in package_json_files[:4]:
            pkg_dir = pkg_json.parent
            try:
                obj = json.loads(pkg_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            scripts = obj.get("scripts", {}) if isinstance(obj, dict) else {}
            if not isinstance(scripts, dict):
                continue
            selected_script = ""
            for name in ("build", "typecheck", "compile", "check"):
                if name in scripts and str(scripts.get(name, "")).strip():
                    selected_script = name
                    break
            if not selected_script:
                continue
            cmd = self._js_script_command(pkg_dir, selected_script)
            if not cmd:
                continue
            rel = str(pkg_dir.relative_to(root)).replace("\\", "/")
            rel = rel if rel else "."
            targets.append(
                {
                    "label": f"node:{rel}:{selected_script}",
                    "cwd": str(pkg_dir),
                    "command": cmd,
                }
            )
        return targets

    def _js_script_command(self, pkg_dir: Path, script_name: str) -> list[str]:
        pnpm = shutil.which("pnpm")
        yarn = shutil.which("yarn")
        npm = shutil.which("npm")
        if (pkg_dir / "pnpm-lock.yaml").exists() and pnpm:
            return [pnpm, "run", script_name]
        if (pkg_dir / "yarn.lock").exists() and yarn:
            return [yarn, script_name]
        if npm:
            return [npm, "run", script_name, "--silent"]
        if yarn:
            return [yarn, script_name]
        if pnpm:
            return [pnpm, "run", script_name]
        return []

    def _detect_python_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        py = shutil.which("python3") or shutil.which("python")
        if not py:
            return []
        has_py = any(
            p.suffix.lower() == ".py" and not self._is_ignored_project_path(p)
            for p in root.rglob("*.py")
        )
        has_project = any((root / n).exists() for n in ("pyproject.toml", "requirements.txt", "setup.py"))
        if not has_py and not has_project:
            return []
        return [
            {
                "label": "python:compileall",
                "cwd": str(root),
                "command": [py, "-m", "compileall", "-q", "."],
            }
        ]

    def _detect_go_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        go = shutil.which("go")
        if not go:
            return []
        go_mods = sorted(
            p for p in root.rglob("go.mod")
            if not self._is_ignored_project_path(p)
        )
        targets: list[dict[str, Any]] = []
        for mod in go_mods[:3]:
            mod_dir = mod.parent
            rel = str(mod_dir.relative_to(root)).replace("\\", "/")
            rel = rel if rel else "."
            targets.append(
                {
                    "label": f"go:{rel}",
                    "cwd": str(mod_dir),
                    "command": [go, "build", "./..."],
                }
            )
        return targets

    def _detect_rust_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        cargo = shutil.which("cargo")
        if not cargo:
            return []
        cargo_tomls = sorted(
            p for p in root.rglob("Cargo.toml")
            if not self._is_ignored_project_path(p)
        )
        targets: list[dict[str, Any]] = []
        for toml in cargo_tomls[:3]:
            crate_dir = toml.parent
            rel = str(crate_dir.relative_to(root)).replace("\\", "/")
            rel = rel if rel else "."
            targets.append(
                {
                    "label": f"rust:{rel}",
                    "cwd": str(crate_dir),
                    "command": [cargo, "check"],
                }
            )
        return targets

    def _detect_dotnet_compile_targets(self, root: Path) -> list[dict[str, Any]]:
        dotnet = shutil.which("dotnet")
        if not dotnet:
            return []
        targets: list[dict[str, Any]] = []
        slns = sorted(
            p for p in root.rglob("*.sln")
            if not self._is_ignored_project_path(p)
        )
        if slns:
            for sln in slns[:3]:
                sln_dir = sln.parent
                rel_sln = str(sln.relative_to(root)).replace("\\", "/")
                targets.append(
                    {
                        "label": f"dotnet:{rel_sln}",
                        "cwd": str(sln_dir),
                        "command": [dotnet, "build", sln.name, "--nologo"],
                    }
                )
            return targets

        csprojs = sorted(
            p for p in root.rglob("*.csproj")
            if not self._is_ignored_project_path(p)
        )
        for proj in csprojs[:3]:
            proj_dir = proj.parent
            rel_proj = str(proj.relative_to(root)).replace("\\", "/")
            targets.append(
                {
                    "label": f"dotnet:{rel_proj}",
                    "cwd": str(proj_dir),
                    "command": [dotnet, "build", proj.name, "--nologo"],
                }
            )
        return targets

    def _is_ignored_project_path(self, path: Path) -> bool:
        ignored = {
            ".git",
            "target",
            "node_modules",
            ".next",
            ".nuxt",
            "dist",
            "build",
            "__pycache__",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
        }
        return any(part in ignored for part in path.parts)

    def _run_compile_targets_once(
        self,
        targets: list[dict[str, Any]],
        *,
        round_idx: int,
        phase: str,
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        self._emit_feedback(
            "compile_round_started",
            "compile round started",
            round=round_idx,
            phase=phase,
            target_count=len(targets),
        )
        for target in targets:
            result = self._run_compile_target(target)
            if result["ok"]:
                self._emit_feedback(
                    "compile_target_done",
                    "compile target succeeded",
                    round=round_idx,
                    phase=phase,
                    target=target["label"],
                    command=" ".join(target["command"]),
                )
                continue
            failures.append(result)
            self._emit_feedback(
                "compile_target_failed",
                "compile target failed",
                round=round_idx,
                phase=phase,
                target=target["label"],
                returncode=result["returncode"],
                output_preview=result["output"][:600],
            )
        self._emit_feedback(
            "compile_round_done",
            "compile round completed",
            round=round_idx,
            phase=phase,
            failed_targets=len(failures),
        )
        return failures

    def _run_compile_target(self, target: dict[str, Any]) -> dict[str, Any]:
        command = list(target.get("command", []))
        cwd = str(target.get("cwd", self.workspace.os_path()))
        label = str(target.get("label", "compile-target"))
        if not command:
            return {"ok": False, "label": label, "returncode": -1, "output": "empty command"}
        try:
            proc = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.compile_timeout_seconds,
                check=False,
            )
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            output = self._clip_text_tail(output, max_chars=24000)
            return {
                "ok": proc.returncode == 0,
                "label": label,
                "returncode": proc.returncode,
                "output": output,
            }
        except Exception as exc:
            return {
                "ok": False,
                "label": label,
                "returncode": -1,
                "output": f"compile command execution error: {exc}",
            }

    def _merge_compile_failures(self, failures: list[dict[str, Any]], *, max_chars: int) -> str:
        blocks: list[str] = []
        for item in failures:
            blocks.append(
                "\n".join(
                    [
                        f"### Target: {item.get('label', '')}",
                        f"### Return Code: {item.get('returncode', '')}",
                        str(item.get("output", "")),
                    ]
                )
            )
        merged = "\n\n".join(blocks)
        if len(merged) > max_chars:
            return merged[-max_chars:]
        return merged

    def _plan_compile_fix_tasks(self, compile_output: str) -> list[dict[str, Any]]:
        review_files = self._list_review_files(max_files=72)
        prompt = COMPILE_ERROR_TASKS_PROMPT.format(
            msg=self.state.user_msg,
            compile_output=compile_output[:18000],
            workspace_manifest=dump_json({"files": self.state.workspace_manifest}),
            file_snapshots=self._build_file_snapshots(review_files, max_chars_per_file=1200),
        )
        try:
            raw = self.llm.chat(prompt, temperature=0.1)
            obj = extract_json_object(raw)
            tasks = obj.get("tasks", [])
            if not isinstance(tasks, list):
                return []
            return [x for x in tasks[:20] if isinstance(x, dict)]
        except Exception:
            return []

    def _list_review_files(self, max_files: int = 80) -> list[str]:
        root = Path(self.workspace.os_path())
        files = set(self.workspace.list_code_files())

        extra_names = {
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
            "mvnw",
            "gradlew",
            "docker-compose.yml",
            "docker-compose.yaml",
            "dockerfile",
            ".env",
            ".env.example",
        }
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if ".git" in p.parts or "__pycache__" in p.parts or "node_modules" in p.parts:
                continue
            name = p.name.lower()
            if name not in extra_names:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            files.add(rel)

        ordered = sorted(files, key=self._review_file_priority)
        return ordered[:max_files]

    def _review_file_priority(self, rel_path: str) -> tuple[int, str]:
        low = rel_path.lower()
        if low.endswith(("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties")):
            return (0, rel_path)
        if low.endswith(("application.yml", "application.yaml", "application.properties", "docker-compose.yml", "docker-compose.yaml")):
            return (1, rel_path)
        if low.endswith(("main.java", "app.py", "main.py", "server.py", "index.ts", "index.js")):
            return (2, rel_path)
        if any(x in low for x in ("/controller", "/service", "/repository", "/entity", "/model", "/config")):
            return (3, rel_path)
        return (5, rel_path)

    def _build_file_snapshots(self, files: list[str], *, max_chars_per_file: int) -> str:
        blocks: list[str] = []
        for rel in files:
            try:
                text = self.workspace.read_text(rel)
            except Exception:
                continue
            snippet = self._clip_text_head(text, max_chars=max_chars_per_file)
            blocks.append(f"### File: {rel}\n{snippet}")
        return "\n\n".join(blocks) if blocks else "(none)"

    def _clip_text_head(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...(truncated)"

    def _clip_text_tail(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return "...(truncated)\n" + text[-max_chars:]

    def _select_global_check_dependency_files(self, rel_path: str, all_files: list[str], max_files: int = 10) -> list[str]:
        rel_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
        top = rel_path.split("/", 1)[0]
        low_target = rel_path.lower()

        scored: list[tuple[int, str]] = []
        for cand in all_files:
            if cand == rel_path:
                continue
            score = 0
            if rel_dir and cand.startswith(rel_dir + "/"):
                score += 40
            if cand.split("/", 1)[0] == top:
                score += 20
            low = cand.lower()
            if any(k in low for k in ("config", "setting", "model", "entity", "service", "controller", "router")):
                score += 8
            if Path(cand).suffix.lower() == Path(rel_path).suffix.lower():
                score += 4
            if low_target.endswith(("pom.xml", "package.json", "docker-compose.yml")) and low.endswith(("pom.xml", "package.json", "docker-compose.yml")):
                score += 15
            scored.append((score, cand))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, p in scored[:max_files]]

    def write_readme(self) -> None:
        if not self.state.code:
            self.state.code = self.workspace.load_code_blocks()

        prompt = WRITE_README_PROMPT.format(code="\n\n".join(self.state.code))
        self.state.readme = self.llm.chat(prompt, temperature=0.3)
        self.workspace.write_text("README.md", self.state.readme)

    def load_existing_state(self) -> None:
        prd = self.workspace.load_state_file("prd.txt")
        arch_json = self.workspace.load_state_file("arch.json")
        arch_txt = self.workspace.load_state_file("arch.txt")
        threat = self.workspace.load_state_file("threat_model.txt")
        threat_json = self.workspace.load_state_file("threat_model.json")
        threat_modules_json = self.workspace.load_state_file("threat_model_modules.json")

        if prd:
            self.state.prd = prd
        if arch_json:
            try:
                self.state.arch = json.loads(arch_json)
            except Exception:
                pass
        elif arch_txt:
            try:
                self.state.arch = json.loads(arch_txt)
            except Exception:
                pass
        if threat:
            self.state.threat_model = threat
        if threat_json:
            try:
                self.state.threat_model_json = json.loads(threat_json)
            except Exception:
                self.state.threat_model_json = {}
        if threat_modules_json:
            try:
                obj = json.loads(threat_modules_json)
                mods = obj.get("modules", []) if isinstance(obj, dict) else []
                self.state.threat_model_modules = mods if isinstance(mods, list) else []
            except Exception:
                self.state.threat_model_modules = []

        self.state.code = self.workspace.load_code_blocks()

    def refresh_workspace_manifest(self) -> None:
        manifest = self.workspace.scan_workspace_manifest()
        self.state.workspace_manifest = manifest
        self.workspace.save_state_file("workspace_manifest.json", dump_json({"files": manifest}))
        self._emit_feedback(
            "workspace_scanned",
            "workspace manifest refreshed",
            file_count=len(manifest),
        )

    def _persist_state_files(self) -> None:
        self.workspace.save_state_file("prd.txt", self.state.prd)
        self.workspace.save_state_file("arch.json", dump_json(self.state.arch))
        self.workspace.save_state_file("arch.txt", dump_json(self.state.arch))
        self.workspace.save_state_file("threat_model.txt", self.state.threat_model)
        self.workspace.save_state_file("threat_model.json", dump_json(self.state.threat_model_json))
        self.workspace.save_state_file("threat_model_modules.json", dump_json(self._build_threat_modules_payload_from_state()))
        self.workspace.save_state_file("workspace_manifest.json", dump_json({"files": self.state.workspace_manifest}))

    def _init_arch_progress(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        items = []
        for file_item in files:
            name = str(file_item.get("name", "")).strip()
            path = str(file_item.get("path", "./")).strip()
            key = self._arch_file_key({"name": name, "path": path})
            pretty = f"{path}/{name}" if path not in {"", ".", "./"} else name
            items.append(
                {
                    "key": key,
                    "file": pretty,
                    "state": "pending",
                    "error": "",
                }
            )

        return {
            "status": "running",
            "total": len(items),
            "completed": 0,
            "remaining": len(items),
            "reused": 0,
            "failed": 0,
            "files": items,
            "updated_at": self._utc_now(),
        }

    def _mark_arch_progress(self, progress: dict[str, Any], *, key: str, state: str, error: str = "") -> None:
        for item in progress.get("files", []):
            if item.get("key") != key:
                continue
            item["state"] = state
            item["error"] = error[:500]
            break
        self._recalc_arch_progress(progress)

    def _recalc_arch_progress(self, progress: dict[str, Any]) -> None:
        files = progress.get("files", [])
        done = sum(1 for x in files if x.get("state") in {"done", "reused", "failed"})
        reused = sum(1 for x in files if x.get("state") == "reused")
        failed = sum(1 for x in files if x.get("state") == "failed")
        total = int(progress.get("total", len(files)))
        progress["completed"] = done
        progress["remaining"] = max(total - done, 0)
        progress["reused"] = reused
        progress["failed"] = failed
        progress["updated_at"] = self._utc_now()

    def _persist_arch_progress(self, progress: dict[str, Any]) -> None:
        self.workspace.save_state_file("arch.progress.json", dump_json(progress))

    def _arch_file_key(self, file_item: dict[str, Any]) -> str:
        name = str(file_item.get("name", "")).strip()
        path = str(file_item.get("path", "./")).strip()
        if not name:
            name = "__unknown__"
        if path in {"", ".", "./"}:
            rel = name
        else:
            rel = f"{path.rstrip('/')}/{name}"
        return safe_rel_path(rel)

    def _ordered_arch_files(self, ordered_keys: list[str], designed_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in ordered_keys:
            if key in designed_map:
                out.append(designed_map[key])
        return out

    def _select_system_design_history_context(
        self,
        file_item: dict[str, Any],
        designed_map: dict[str, dict[str, Any]],
        ordered_keys: list[str],
        max_files: int = 10,
    ) -> dict[str, Any]:
        all_designed = self._ordered_arch_files(ordered_keys, designed_map)
        if not all_designed:
            return {"files": []}

        target_key = self._arch_file_key(file_item)
        target_dir = target_key.rsplit("/", 1)[0] if "/" in target_key else ""
        target_top = target_key.split("/", 1)[0]

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in all_designed:
            key = self._arch_file_key(item)
            score = 0
            if key == target_key:
                score += 100
            if target_dir and key.startswith(target_dir + "/"):
                score += 40
            if key.split("/", 1)[0] == target_top:
                score += 20
            low = key.lower()
            if any(k in low for k in ("config", "setting", "schema", "model", "type", "service", "core")):
                score += 12
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [self._compact_arch_item_for_prompt(item) for _, item in scored[:max_files]]
        return {"files": selected}

    def _compact_arch_item_for_prompt(self, item: dict[str, Any]) -> dict[str, Any]:
        out = {
            "name": item.get("name", ""),
            "path": item.get("path", "./"),
            "description": item.get("description", ""),
            "classes": [],
            "functions": [],
        }
        classes = item.get("classes", [])
        if isinstance(classes, list):
            for cls in classes[:8]:
                if not isinstance(cls, dict):
                    continue
                out["classes"].append(
                    {
                        "class_name": cls.get("class_name", ""),
                        "members": cls.get("members", [])[:10] if isinstance(cls.get("members", []), list) else [],
                    }
                )

        funcs = item.get("functions", [])
        if isinstance(funcs, list):
            for fn in funcs[:16]:
                if not isinstance(fn, dict):
                    continue
                out["functions"].append(
                    {
                        "name": fn.get("name", ""),
                        "input_parameters": fn.get("input_parameters", [])[:8] if isinstance(fn.get("input_parameters", []), list) else [],
                        "output_parameters": fn.get("output_parameters", [])[:8] if isinstance(fn.get("output_parameters", []), list) else [],
                    }
                )
        return out

    def _compact_arch_for_planning(self, max_files: int = 100) -> dict[str, Any]:
        files = self.state.arch.get("files", [])
        if not isinstance(files, list):
            return {"files": []}
        compact = [self._compact_arch_item_for_prompt(item) for item in files if isinstance(item, dict)]
        return {"files": compact}

    def _compact_threat_for_planning(self, max_entries: int = 120) -> dict[str, Any]:
        if isinstance(self.state.threat_model_json, dict) and self.state.threat_model_json:
            entries = self.state.threat_model_json.get("functions", [])
            if not isinstance(entries, list):
                entries = []
            return {
                "global_context": self.state.threat_model_json.get("global_context", {}),
                "attacker_model": self.state.threat_model_json.get("attacker_model", {}),
                "functions": entries[:max_entries],
            }
        if self.state.threat_model_modules:
            mini = []
            for mod in self.state.threat_model_modules[:40]:
                if not isinstance(mod, dict):
                    continue
                mini.append(
                    {
                        "module": mod.get("module", ""),
                        "file_count": mod.get("file_count", 0),
                        "files": mod.get("files", [])[:20] if isinstance(mod.get("files", []), list) else [],
                        "threat_excerpt": str(mod.get("threat_model", ""))[:1200],
                    }
                )
            return {"modules": mini}
        return {"raw_excerpt": self.state.threat_model[:8000]}

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _system_design_priority(self, file_item: dict[str, Any]) -> tuple[int, str]:
        name = str(file_item.get("name", "")).lower()
        path = str(file_item.get("path", "./")).lower().strip()
        rel = name if path in {"", ".", "./"} else f"{path.rstrip('/')}/{name}"

        # Lower number = designed earlier so later waves can reference stable contracts.
        if any(rel.endswith(x) for x in ("package.json", "pyproject.toml", "requirements.txt", "dockerfile", "docker-compose.yml")):
            return (0, rel)
        if any(x in rel for x in ("/config", "/settings", "config.", "settings.")):
            return (1, rel)
        if any(x in rel for x in ("/model", "/models", "/schema", "/schemas", "/entity", "/entities", "/types", "/dto")):
            return (2, rel)
        if any(x in rel for x in ("/core", "/service", "/services", "/repo", "/repository", "/db")):
            return (3, rel)
        if any(x in rel for x in ("/api", "/route", "/routes", "/controller", "/handler")):
            return (4, rel)
        if any(x in rel for x in ("/ui", "/components", "/pages", "/frontend")):
            return (5, rel)
        if any(x in rel for x in ("/test", "/tests", "test_", "_test.")):
            return (8, rel)
        return (6, rel)

    def _arch_file_paths(self) -> set[str]:
        out: set[str] = set()
        for f in self.state.arch.get("files", []):
            name = str(f.get("name", "")).strip()
            path = str(f.get("path", "./")).strip()
            if not name:
                continue
            if path in {".", "./", ""}:
                rel = name
            else:
                rel = f"{path.rstrip('/')}/{name}"
            out.add(safe_rel_path(rel))
        return out

    def _build_worker(self, worker_id: str) -> CodeWorkerAgent:
        return CodeWorkerAgent(
            worker_id=worker_id,
            llm=self.llm,
            state=self.state,
            workspace=self.workspace,
            memory=self.memory,
            event_callback=self._on_worker_event,
        )

    def _preload_memory_with_workspace_files(self) -> None:
        self.memory.add("user_msg", self.state.user_msg, {"kind": "user_msg"})
        if self.state.prd:
            self.memory.add("prd", self.state.prd, {"kind": "prd"})
        if self.state.threat_model:
            self.memory.add("threat", self.state.threat_model, {"kind": "threat_model"})
        if self.state.threat_model_modules:
            self.memory.add(
                "threat_modules",
                dump_json({"modules": self.state.threat_model_modules}),
                {"kind": "threat_modules"},
            )
        if self.state.threat_model_json:
            self.memory.add("threat_json", dump_json(self.state.threat_model_json), {"kind": "threat_model_json"})
        if self.state.arch:
            self.memory.add("arch", dump_json(self.state.arch), {"kind": "architecture"})
        if self.state.workspace_manifest:
            self.memory.add(
                "workspace_manifest",
                dump_json({"files": self.state.workspace_manifest}),
                {"kind": "workspace_manifest"},
            )

        for rel in self.workspace.list_code_files():
            try:
                text = self.workspace.read_text(rel)
            except Exception:
                continue
            self.memory.add(
                entry_id=f"existing:{rel}",
                text=f"File: {rel}\n{text[:4000]}",
                metadata={"kind": "existing_code", "file": rel},
            )

    def _on_scheduler_event(self, event: dict) -> None:
        et = str(event.get("type", "scheduler_event"))
        msg_map = {
            "task_assigned": "task assigned",
            "task_waiting": "task waiting",
            "task_done": "task completed",
            "task_failed": "task failed",
            "task_blocked": "task blocked",
            "scheduler_deadlock": "scheduler deadlock",
        }
        self._emit_feedback(et, msg_map.get(et, et), **event)

    def _on_worker_event(self, event: dict) -> None:
        et = str(event.get("type", "worker_event"))
        msg_map = {
            "worker_task_started": "worker task started",
            "worker_task_skipped": "worker task skipped",
            "worker_task_failed": "worker task failed",
            "worker_warm_memory": "worker warm memory loaded",
            "worker_step": "worker step",
            "worker_memory_search": "worker searched memory",
            "worker_read_file": "worker read file",
            "worker_run_command": "worker ran command",
            "worker_check_done": "worker check done",
            "worker_edit_skipped": "worker edit skipped",
            "worker_edit_done": "worker edit done",
            "worker_edit_failed": "worker edit failed",
            "worker_edit_noop": "worker edit noop",
            "worker_edit_fallback_write": "worker edit fallback write",
            "worker_write_failed": "worker write failed",
            "worker_write_done": "worker write done",
        }
        self._emit_feedback(et, msg_map.get(et, et), **event)

    def _on_threat_event(self, event: dict) -> None:
        et = str(event.get("type", "threat_event"))
        event_for_log = dict(event)
        threat_model_text = event_for_log.pop("threat_model", None)

        if et in {"threat_module_started", "threat_module_done"}:
            self._persist_live_threat_module_event(event_for_log, threat_model_text)

        msg_map = {
            "threat_module_started": "threat model module started",
            "threat_module_done": "threat model module done",
        }
        self._emit_feedback(et, msg_map.get(et, et), **event_for_log)

    def _emit_feedback(self, event_type: str, message: str, **details: Any) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "message": message,
            "details": details,
        }
        self.state.feedback_events.append(event)

        if self.persist_feedback:
            self.workspace.append_text("progress.jsonl", json.dumps(event, ensure_ascii=False) + "\n")

        if self.feedback_callback is not None:
            try:
                self.feedback_callback(event)
            except Exception:
                pass

    def _parse_file_design_response(self, response_text: str, original_prompt: str) -> dict[str, Any]:
        try:
            return extract_json_object(response_text)
        except Exception as first_exc:
            self._emit_feedback(
                "system_design_file_design_retry",
                "file design JSON parse failed, attempting repair",
                error=str(first_exc),
            )
            repair_prompt = "\n".join(
                [
                    "Your previous output is not strict valid JSON.",
                    "Convert it into strict JSON object with schema:",
                    '{"files":[{"name":"string","path":"./or/subdir","description":"string"}]}',
                    "Return JSON only, no markdown, no explanation.",
                    "",
                    "Previous output:",
                    response_text,
                    "",
                    "Original task prompt:",
                    original_prompt,
                ]
            )
            repaired = self.llm.chat(repair_prompt, temperature=0.1)
            try:
                return extract_json_object(repaired)
            except Exception as second_exc:
                preview = repaired[:500].replace("\n", "\\n")
                raise ValueError(
                    f"Failed to parse file design JSON after repair. "
                    f"first_error={first_exc}; second_error={second_exc}; repaired_preview={preview}"
                ) from second_exc

    def _persist_live_threat_module_event(self, event: dict[str, Any], threat_model_text: str | None) -> None:
        module = str(event.get("module", "")).strip()
        idx = int(event.get("index", 0) or 0)
        total = int(event.get("total", 0) or 0)
        file_count = int(event.get("file_count", 0) or 0)
        files = event.get("files", [])
        if not isinstance(files, list):
            files = []
        et = str(event.get("type", ""))
        if not module:
            return

        key = f"{idx}:{module}" if idx else module
        with self._threat_module_lock:
            current = self._live_threat_modules.get(
                key,
                {
                    "module": module,
                    "index": idx,
                    "total": total,
                    "file_count": file_count,
                    "files": files[:100],
                    "status": "pending",
                    "threat_model": "",
                },
            )
            current["module"] = module
            current["index"] = idx
            current["total"] = total
            current["file_count"] = file_count or current.get("file_count", 0)
            current["files"] = files[:100] or current.get("files", [])
            if et == "threat_module_started":
                current["status"] = "running"
            elif et == "threat_module_done":
                current["status"] = "done"
                if threat_model_text:
                    current["threat_model"] = threat_model_text

            self._live_threat_modules[key] = current
            modules = list(self._live_threat_modules.values())
            modules.sort(key=lambda x: (int(x.get("index", 0)), str(x.get("module", ""))))
            completed = sum(1 for m in modules if m.get("status") == "done")
            running = sum(1 for m in modules if m.get("status") == "running")
            payload = {
                "status": "running",
                "total": max(total, len(modules)),
                "completed": completed,
                "running": running,
                "modules": modules,
                "updated_at": self._utc_now(),
            }
            self.workspace.save_state_file("threat_model_modules.json", dump_json(payload))

    def _build_threat_modules_payload_from_state(self) -> dict[str, Any]:
        modules: list[dict[str, Any]] = []
        raw = self.state.threat_model_modules
        if isinstance(raw, list):
            for i, item in enumerate(raw, start=1):
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("index", i))
                mod = dict(item)
                mod["index"] = idx
                mod["status"] = str(mod.get("status", "done"))
                modules.append(mod)
        modules.sort(key=lambda x: (int(x.get("index", 0)), str(x.get("module", ""))))
        total = len(modules)
        completed = sum(1 for m in modules if str(m.get("status", "")) == "done")
        running = sum(1 for m in modules if str(m.get("status", "")) == "running")
        status = "done" if total and completed == total else ("running" if running else "idle")
        return {
            "status": status,
            "total": total,
            "completed": completed,
            "running": running,
            "modules": modules,
            "updated_at": self._utc_now(),
        }


# Backward-compatible alias
MASecDev = ParallelMASecDev
