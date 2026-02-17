from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from .api import LLMClient
from .memory import SharedMemoryPool
from .models import CodeTask, ProjectState, TaskStatus, TaskWaiting, WorkerResult
from .prompts import (
    COMBINE_THREAT_MODELS_PROMPT,
    FILE_CHECK_PROMPT,
    FILE_EDIT_PROMPT,
    FILE_EDIT_REPAIR_PROMPT,
    FILE_IMPLEMENT_PROMPT,
    GET_CALL_CHAIN_PROMPT,
    SYSTEM_DESIGN_CONSISTENCY_PROMPT,
    THREAT_MODEL_TO_JSON_PROMPT,
    THREAT_MODELING_PER_CHAIN_PROMPT,
    THREAT_MODELING_PROMPT,
    WORKER_REFLECTION_PROMPT,
)
from .utils import (
    apply_line_edits,
    dump_json,
    extract_json_object,
    parse_file_block,
    safe_rel_path,
    strip_markdown_code_fence,
    with_line_numbers,
)
from .workspace import Workspace


class ThreatModelAgent:
    def __init__(
        self,
        llm: LLMClient,
        state: ProjectState,
        mode: str = "call_chain",
        *,
        modular: bool = False,
        module_max_files: int = 12,
        max_workers: int = 4,
        context_max_files: int = 32,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.llm = llm
        self.state = state
        self.mode = mode
        self.modular = modular
        self.module_max_files = max(1, module_max_files)
        self.max_workers = max(1, max_workers)
        self.context_max_files = max(8, context_max_files)
        self.event_callback = event_callback
        # Keep combine prompts well below common provider context limits.
        self.combine_api_max_files = 20
        self.combine_prompt_char_budget = 52000
        self.combine_fragment_char_cap = 5000
        self.combine_round_output_cap = 14000

    def generate(self) -> str:
        final_text, _ = self.generate_with_modules()
        return final_text

    def generate_with_modules(self) -> tuple[str, list[dict]]:
        if not self.modular:
            arch_ctx = self._build_module_arch_context(self.state.arch.get("files", []), "all")
            return self._generate_for_arch(arch_ctx), []

        modules = self._build_modules(self.state.arch)
        if not modules:
            arch_ctx = self._build_module_arch_context(self.state.arch.get("files", []), "all")
            return self._generate_for_arch(arch_ctx), []

        outputs: list[dict] = []
        worker_count = min(self.max_workers, len(modules))
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {
                ex.submit(self._generate_module_threat, idx, mod, len(modules)): idx
                for idx, mod in enumerate(modules, start=1)
            }
            for future in as_completed(futures):
                try:
                    outputs.append(future.result())
                except Exception:
                    continue
        outputs.sort(key=lambda x: int(x.get("index", 0)))

        if len(outputs) == 1:
            return outputs[0]["threat_model"], outputs

        combined_input = []
        for out in outputs:
            combined_input.append(
                "\n".join(
                    [
                        f"### Module: {out['module']}",
                        f"### Files: {', '.join(out['files'][:20])}",
                        out["threat_model"],
                    ]
                )
            )

        final_text = self._combine_threat_fragments(
            combined_input,
            self._compact_arch_for_prompt(
                self.state.arch,
                max_files=min(self.combine_api_max_files, self.context_max_files),
            ),
        )
        return final_text, outputs

    def _generate_for_arch(self, arch: dict) -> str:
        if self.mode == "simple":
            prompt = THREAT_MODELING_PROMPT.format(
                msg=self.state.user_msg,
                api=dump_json(self._compact_arch_for_prompt(arch, max_files=self.context_max_files)),
            )
            return self.llm.chat(prompt)

        arch_ctx = self._compact_arch_for_prompt(arch, max_files=max(self.context_max_files, 40))
        interfaces = self._extract_interfaces(arch_ctx)
        if not interfaces:
            prompt = THREAT_MODELING_PROMPT.format(msg=self.state.user_msg, api=dump_json(arch_ctx))
            return self.llm.chat(prompt)

        per_chain: list[str] = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [
                ex.submit(self._build_chain_threat, interface, arch_ctx)
                for interface in interfaces
            ]
            for future in as_completed(futures):
                try:
                    per_chain.append(future.result())
                except Exception:
                    continue

        if not per_chain:
            prompt = THREAT_MODELING_PROMPT.format(msg=self.state.user_msg, api=dump_json(arch_ctx))
            return self.llm.chat(prompt)

        return self._combine_threat_fragments(per_chain, arch_ctx)

    def _combine_threat_fragments(self, fragments: list[str], api_ctx: dict) -> str:
        working = [
            self._cap_text(str(item).strip(), self.combine_fragment_char_cap)
            for item in fragments
            if str(item).strip()
        ]
        if not working:
            return ""
        if len(working) == 1:
            return working[0]

        api_text = dump_json(api_ctx)
        rounds = 0
        while len(working) > 1 and rounds < 8:
            rounds += 1
            batches = self._batch_fragments_for_combine(working, api_text)

            # No progress possible with current budget, aggressively shrink context.
            if len(batches) == len(working):
                if api_text != "{}":
                    api_text = "{}"
                working = [self._cap_text(text, max(700, len(text) // 2)) for text in working]
                continue

            next_round: list[str] = []
            for batch in batches:
                if len(batch) == 1:
                    next_round.append(batch[0])
                    continue
                try:
                    merged = self._combine_fragment_batch(batch, api_text)
                except Exception as exc:
                    if "maximum context length" in str(exc).lower():
                        reduced = [self._cap_text(text, max(700, len(text) // 2)) for text in batch]
                        try:
                            merged = self._combine_fragment_batch(reduced, "{}")
                        except Exception:
                            merged = self._cap_text("\n\n".join(reduced), self.combine_round_output_cap)
                    else:
                        raise
                next_round.append(self._cap_text(merged, self.combine_round_output_cap))
            working = [item for item in next_round if item.strip()]
            if not working:
                return ""

        if len(working) > 1:
            return self._cap_text("\n\n".join(working), self.combine_round_output_cap * 2)
        return working[0]

    def _batch_fragments_for_combine(self, fragments: list[str], api_text: str) -> list[list[str]]:
        base_prompt = COMBINE_THREAT_MODELS_PROMPT.format(
            msg=self.state.user_msg,
            api=api_text,
            threat_models="",
        )
        model_budget = max(1800, self.combine_prompt_char_budget - len(base_prompt))
        target_batch_size = max(2, min(4, len(fragments)))
        per_item_cap = max(700, (model_budget // target_batch_size) - 180)
        capped_items = [self._cap_text(text, per_item_cap) for text in fragments]

        batches: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for text in capped_items:
            item_len = len(text) + 60
            if current and (len(current) >= target_batch_size or current_len + item_len > model_budget):
                batches.append(current)
                current = []
                current_len = 0
            current.append(text)
            current_len += item_len
        if current:
            batches.append(current)
        return batches

    def _combine_fragment_batch(self, batch: list[str], api_text: str) -> str:
        tagged = [f"### Fragment {idx}\n{text}" for idx, text in enumerate(batch, start=1)]
        prompt = COMBINE_THREAT_MODELS_PROMPT.format(
            msg=self.state.user_msg,
            api=api_text,
            threat_models="\n\n".join(tagged),
        )
        return self.llm.chat(prompt)

    def _cap_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 32:
            return text[:max_chars]
        return f"{text[: max_chars - 20]}\n\n[truncated]"

    def _generate_module_threat(self, idx: int, mod: dict, total: int) -> dict:
        module_name = mod["module"]
        file_items = mod["files"]
        file_paths = [self._file_rel_path(x) for x in file_items]
        self._emit(
            {
                "type": "threat_module_started",
                "module": module_name,
                "index": idx,
                "total": total,
                "file_count": len(file_items),
                "files": file_paths,
            }
        )
        arch_ctx = self._build_module_arch_context(file_items, module_name)
        text = self._generate_for_arch(arch_ctx)
        self._emit(
            {
                "type": "threat_module_done",
                "module": module_name,
                "index": idx,
                "total": total,
                "file_count": len(file_items),
                "files": file_paths,
                "threat_model": text,
            }
        )
        return {
            "module": module_name,
            "index": idx,
            "file_count": len(file_items),
            "files": file_paths,
            "threat_model": text,
        }

    def _extract_interfaces(self, arch: dict) -> list[dict]:
        prompt = GET_CALL_CHAIN_PROMPT.format(api=dump_json(arch))
        response = self.llm.chat(prompt)
        if self.state.call_chain:
            self.state.call_chain = f"{self.state.call_chain}\n\n{response}"
        else:
            self.state.call_chain = response
        try:
            obj = extract_json_object(response)
            return obj.get("interfaces", [])
        except Exception:
            return []

    def _build_chain_threat(self, interface: dict, arch: dict) -> str:
        prompt = THREAT_MODELING_PER_CHAIN_PROMPT.format(
            msg=self.state.user_msg,
            prd=self.state.prd,
            api=dump_json(arch),
            interface=json.dumps(interface, ensure_ascii=False, indent=2),
        )
        return self.llm.chat(prompt)

    def _build_modules(self, arch: dict) -> list[dict]:
        files = arch.get("files", [])
        if not isinstance(files, list) or not files:
            return []

        grouped: dict[str, list[dict]] = {}
        order: list[str] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = self._file_rel_path(item)
            top = rel.split("/", 1)[0] if "/" in rel else "root"
            if top not in grouped:
                grouped[top] = []
                order.append(top)
            grouped[top].append(item)

        modules: list[dict] = []
        for group in order:
            chunk = grouped[group]
            for i in range(0, len(chunk), self.module_max_files):
                part = chunk[i : i + self.module_max_files]
                module_name = group if len(chunk) <= self.module_max_files else f"{group}:{(i // self.module_max_files) + 1}"
                modules.append({"module": module_name, "files": part})
        return modules

    def _build_module_arch_context(self, file_items: list[dict], module_name: str) -> dict:
        all_files = self.state.arch.get("files", [])
        all_index = []
        if isinstance(all_files, list):
            for item in all_files[:300]:
                if not isinstance(item, dict):
                    continue
                all_index.append(self._file_rel_path(item))
        return {
            "module": module_name,
            "all_files_index": all_index,
            "files": [
                self._compact_arch_item(item)
                for item in file_items[: self.context_max_files]
                if isinstance(item, dict)
            ],
        }

    def _compact_arch_for_prompt(self, arch: dict, max_files: int) -> dict:
        files = arch.get("files", [])
        if not isinstance(files, list):
            return {"files": []}
        return {
            "module": arch.get("module", "all"),
            "all_files_index": arch.get("all_files_index", []),
            "files": [
                self._compact_arch_item(item)
                for item in files[:max_files]
                if isinstance(item, dict)
            ],
        }

    def _compact_arch_item(self, item: dict) -> dict:
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

    def _file_rel_path(self, file_item: dict) -> str:
        name = str(file_item.get("name", "")).strip()
        path = str(file_item.get("path", "./")).strip()
        rel = name if path in {"", ".", "./"} else f"{path.rstrip('/')}/{name}"
        try:
            return safe_rel_path(rel)
        except Exception:
            return rel

    def _emit(self, event: dict) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event)
        except Exception:
            pass

    def to_structured_json(self, threat_text: str) -> dict:
        prompt = THREAT_MODEL_TO_JSON_PROMPT.format(threat_model=threat_text)
        try:
            return extract_json_object(self.llm.chat(prompt, temperature=0.2))
        except Exception:
            return self._fallback_parse_structured(threat_text)

    def _fallback_parse_structured(self, threat_text: str) -> dict:
        import re

        functions: list[dict] = []
        fn_blocks = re.findall(
            r"### Function:\s*(.+?)\n(.*?)(?=\n### Function:|\n## 3\.|\Z)",
            threat_text,
            flags=re.DOTALL,
        )
        for fn, body in fn_blocks:
            fn = fn.strip()
            file_path = fn.split(":", 1)[0].strip()
            threats = re.findall(r"-\s*(?:T\d+:\s*)?(.+)", body)
            functions.append(
                {
                    "function": fn,
                    "file": file_path,
                    "role": "",
                    "untrusted_inputs": [],
                    "security_operations": [],
                    "threats": [t.strip() for t in threats[:20]],
                    "protections": [],
                }
            )

        return {
            "global_context": {
                "overall_purpose": "",
                "trust_boundaries": [],
                "assets_and_privileges": [],
            },
            "functions": functions,
            "attacker_model": {
                "capabilities": [],
                "goals": [],
            },
            "raw_excerpt": threat_text[:4000],
        }


class SystemDesignConsistencyAgent:
    def __init__(self, llm: LLMClient, state: ProjectState) -> None:
        self.llm = llm
        self.state = state

    def review(self) -> dict:
        prompt = SYSTEM_DESIGN_CONSISTENCY_PROMPT.format(
            msg=self.state.user_msg,
            prd=self.state.prd,
            api=self.state.arch,
        )
        response = self.llm.chat(prompt, temperature=0.2)
        return extract_json_object(response)


class CodeWorkerAgent:
    def __init__(
        self,
        worker_id: str,
        llm: LLMClient,
        state: ProjectState,
        workspace: Workspace,
        memory: SharedMemoryPool,
        *,
        max_steps: int = 8,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.llm = llm
        self.state = state
        self.workspace = workspace
        self.memory = memory
        self.max_steps = max_steps
        self.event_callback = event_callback

    def execute(
        self,
        task: CodeTask,
        *,
        dependency_files: Iterable[str],
        completed_files: Iterable[str],
    ) -> WorkerResult:
        file_exists = self.workspace.exists(task.file)
        if task.mode == "create_only" and file_exists:
            self._emit(
                {
                    "type": "worker_task_skipped",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "reason": "create_only and target exists",
                }
            )
            return WorkerResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.DONE,
                produced_file=task.file,
                summary="create_only skipped because file already exists",
            )

        if task.mode == "update_only" and not file_exists:
            self._emit(
                {
                    "type": "worker_task_failed",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "reason": "update_only and target missing",
                }
            )
            return WorkerResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                produced_file=None,
                summary="update_only task failed because target file does not exist",
            )

        dep_files = list(dependency_files)
        missing_dep = [f for f in dep_files if not self.workspace.exists(f)]
        if missing_dep:
            raise TaskWaiting(f"dependencies not ready: {', '.join(missing_dep)}")

        self._emit(
            {
                "type": "worker_task_started",
                "worker": self.worker_id,
                "task_id": task.task_id,
                "file": task.file,
                "mode": task.mode,
                "dependency_files": dep_files,
            }
        )

        memory_snippets: list[str] = []
        file_snippets: list[str] = []
        loaded_file_snippets: set[str] = set()
        run_command_count = 0
        search_memory_count = 0
        read_target_counts: dict[str, int] = {}
        if task.file:
            warm_hits = self.memory.query(f"{task.file} {task.goal}", top_k=4)
            memory_snippets.extend(
                self._render_memory_snippet(score, item.text, item.metadata)
                for score, item in warm_hits
            )
            self._emit(
                {
                    "type": "worker_warm_memory",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "hit_count": len(warm_hits),
                }
            )

        for dep in dep_files:
            if dep in loaded_file_snippets:
                continue
            file_snippets.append(self._render_file_snippet(dep))
            loaded_file_snippets.add(dep)

        for step in range(1, self.max_steps + 1):
            action, args, debug = self._next_action(
                task=task,
                memory_snippets=memory_snippets,
                file_snippets=file_snippets,
                completed_files=list(completed_files),
            )
            self._emit(
                {
                    "type": "worker_step",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "step": step,
                    "action": action,
                    "args": self._shorten_dict(args, 240),
                    "reason": debug.get("reason", ""),
                    "raw_response_preview": debug.get("raw_preview", ""),
                }
            )

            if action == "wait":
                raise TaskWaiting(args.get("reason", "worker requested wait"))

            if action == "search_memory":
                search_memory_count += 1
                if search_memory_count > 2:
                    self._emit(
                        {
                            "type": "worker_action_redirected",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "from_action": "search_memory",
                            "reason": "too many search_memory steps; forcing implementation",
                        }
                    )
                    if self.workspace.exists(task.file):
                        return self._edit_target_file(task, dep_files, memory_snippets)
                    return self._write_target_file(task, dep_files, memory_snippets)
                query = args.get("query") or task.goal
                hits = self.memory.query(query, top_k=5)
                memory_snippets = [
                    self._render_memory_snippet(score, item.text, item.metadata)
                    for score, item in hits
                ]
                self._emit(
                    {
                        "type": "worker_memory_search",
                        "worker": self.worker_id,
                        "task_id": task.task_id,
                        "file": task.file,
                        "query": str(query)[:200],
                        "hit_count": len(hits),
                    }
                )
                continue

            # --- inside CodeWorkerAgent.execute(), in the main for-step loop ---

            if action == "read_file":
                # 1) 必须显式指定 args.file：不再默认回退到 task.file，避免无限 read 同一个文件
                raw_target = str(args.get("file") or "").strip()
                if not raw_target:
                    self._emit(
                        {
                            "type": "worker_read_file",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "target_file": "",
                            "status": "missing_arg_file",
                        }
                    )
                    # 给 LLM 明确反馈：你要读就说清楚读哪个
                    memory_snippets.append(
                        "[read_file_invalid] args.file is required. Choose a concrete existing file path to read. "
                        "If you already have enough context, choose write_file/edit_file/finish."
                    )

                    # 两种策略二选一：
                    # A) 保守：继续下一轮，让 LLM 重新决策（推荐先用这个）
                    continue

                    # B) 激进：直接强制推进实现（如果你不想浪费 token）
                    # if self.workspace.exists(task.file):
                    #     return self._edit_target_file(task, dep_files, memory_snippets)
                    # return self._write_target_file(task, dep_files, memory_snippets)

                # 2) 路径安全校验
                try:
                    target_file = safe_rel_path(raw_target)
                except Exception:
                    self._emit(
                        {
                            "type": "worker_read_file",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "target_file": raw_target[:240],
                            "status": "invalid_path",
                        }
                    )
                    memory_snippets.append(
                        f"[read_file_invalid_path] rejected unsafe file path: {raw_target[:200]}"
                    )
                    continue

                # 3) 去重：已加载过的文件，不再重复读（强力防 loop）
                if target_file in loaded_file_snippets:
                    self._emit(
                        {
                            "type": "worker_read_file",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "target_file": target_file,
                            "status": "already_loaded",
                        }
                    )
                    memory_snippets.append(
                        f"[read_file_skipped] {target_file} already loaded in context. "
                        "Do not request it again; proceed with write_file/edit_file/finish."
                    )

                    # 如果它在“卡住”，可以选择在这里直接推进（可选）
                    # if self.workspace.exists(task.file):
                    #     return self._edit_target_file(task, dep_files, memory_snippets)
                    # return self._write_target_file(task, dep_files, memory_snippets)

                    continue

                # 4) 计数：同一文件最多读 N 次（防止“换着花样读同一个”）
                read_target_counts[target_file] = read_target_counts.get(target_file, 0) + 1
                if read_target_counts[target_file] > 2:
                    self._emit(
                        {
                            "type": "worker_action_redirected",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "from_action": "read_file",
                            "reason": f"repeated reads on {target_file}; forcing implementation",
                        }
                    )
                    memory_snippets.append(
                        f"[read_file_guard] already read {target_file} multiple times; stop reading and implement now."
                    )
                    if self.workspace.exists(task.file):
                        return self._edit_target_file(task, dep_files, memory_snippets)
                    return self._write_target_file(task, dep_files, memory_snippets)

                # 5) 读取逻辑：存在就加载 snippet；不存在就提示并引导推进
                if self.workspace.exists(target_file):
                    file_snippets.append(self._render_file_snippet(target_file))
                    loaded_file_snippets.add(target_file)

                    self._emit(
                        {
                            "type": "worker_read_file",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "target_file": target_file,
                            "status": "ok",
                        }
                    )

                    # 读完之后继续下一轮，让 LLM 基于新上下文选 write/edit/finish
                    continue

                # 6) 文件缺失：记录一次，避免死循环
                self._emit(
                    {
                        "type": "worker_read_file",
                        "worker": self.worker_id,
                        "task_id": task.task_id,
                        "file": task.file,
                        "target_file": target_file,
                        "status": "missing",
                    }
                )
                memory_snippets.append(
                    f"[read_file_missing] {target_file} does not exist. "
                    "Choose a different existing file, or proceed with write_file/edit_file based on current context."
                )

                # 如果它请求读的就是“目标文件”但目标文件不存在：直接写（保留你原来的好逻辑）
                if target_file == task.file:
                    memory_snippets.append(
                        f"[read_file_missing_target] target file {target_file} missing; switching to write_file"
                    )
                    return self._write_target_file(task, dep_files, memory_snippets)

                continue


            # if action == "run_command":
            #     run_command_count += 1
            #     if run_command_count > 2:
            #         self._emit(
            #             {
            #                 "type": "worker_action_redirected",
            #                 "worker": self.worker_id,
            #                 "task_id": task.task_id,
            #                 "file": task.file,
            #                 "from_action": "run_command",
            #                 "reason": "too many run_command steps; forcing implementation",
            #             }
            #         )
            #         if self.workspace.exists(task.file):
            #             return self._edit_target_file(task, dep_files, memory_snippets)
            #         return self._write_target_file(task, dep_files, memory_snippets)
            #     command = args.get("command") or "ls"
            #     output = self._run_readonly_command(command)
            #     memory_snippets.append(f"[command] {command}\n{output[:2000]}")
            #     self._emit(
            #         {
            #             "type": "worker_run_command",
            #             "worker": self.worker_id,
            #             "task_id": task.task_id,
            #             "file": task.file,
            #             "command": str(command)[:200],
            #             "output_preview": output[:300],
            #         }
            #     )
            #     continue

            if action == "edit_file":
                pass

            if action == "write_file":
                result = self._write_target_file(task, dep_files, memory_snippets)
                return result

            if action == "finish":
                if self.workspace.exists(task.file):
                    if task.mode in {"update_only", "update_or_create"}:
                        self._emit(
                            {
                                "type": "worker_finish_redirected",
                                "worker": self.worker_id,
                                "task_id": task.task_id,
                                "file": task.file,
                                "reason": "finish requested on update task; redirecting to edit_file",
                            }
                        )
                        memory_snippets.append(
                            "[finish_guard] update task cannot finish without an edit attempt; redirecting to edit_file"
                        )
                        result = self._edit_target_file(task, dep_files, memory_snippets)
                        return result
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        status=TaskStatus.DONE,
                        produced_file=task.file,
                        summary="completed",
                    )
                result = self._write_target_file(task, dep_files, memory_snippets)
                return result

        result = self._write_target_file(task, dep_files, memory_snippets)
        return result

    def _next_action(
        self,
        *,
        task: CodeTask,
        memory_snippets: list[str],
        file_snippets: list[str],
        completed_files: list[str],
    ) -> tuple[str, dict, dict]:
        arch_ctx = self._select_arch_context(task.file) if task.file else {"files": []}
        arch_snippet = dump_json(arch_ctx)
        code_tree = self.workspace.render_code_tree()
        prompt = WORKER_REFLECTION_PROMPT.format(
            worker_id=self.worker_id,
            task_id=task.task_id,
            api=arch_snippet,
            code_tree=code_tree,
            file_path=task.file,
            goal=task.goal,
            dependencies=sorted(task.depends_on),
            completed_files=completed_files,
            memory_snippets="\n\n".join(memory_snippets) or "(none)",
            file_snippets="\n\n".join(file_snippets) or "(none)",
        )

        try:
            response = self.llm.chat(prompt, temperature=0.2)
            obj = extract_json_object(response)
            action = str(obj.get("action", "write_file"))
            args = obj.get("args", {})
            if action not in {"search_memory", "read_file", "edit_file", "write_file", "wait", "finish"}:
                fallback = "edit_file" if self.workspace.exists(task.file) else "write_file"
                return fallback, {}, {"reason": "invalid_action", "raw_preview": response[:240]}
            if not isinstance(args, dict):
                return action, {}, {"reason": "invalid_args", "raw_preview": response[:240]}
            return action, args, {"reason": "llm_ok", "raw_preview": response[:240]}
        except Exception as exc:
            if not memory_snippets:
                return "search_memory", {"query": task.goal}, {"reason": f"fallback_no_memory:{exc}", "raw_preview": ""}
            fallback = "edit_file" if self.workspace.exists(task.file) else "write_file"
            return fallback, {}, {"reason": f"fallback_with_memory:{exc}", "raw_preview": ""}

    def _write_target_file(
        self,
        task: CodeTask,
        dependency_files: list[str],
        memory_snippets: list[str],
    ) -> WorkerResult:
        dependency_context: list[str] = []
        for dep in dependency_files:
            if self.workspace.exists(dep):
                dependency_context.append(self._render_file_snippet(dep))

        target_design = self._target_file_design(task.file)
        existing_file_context = "(file does not exist yet)"
        if self.workspace.exists(task.file):
            existing_file_context = self._render_file_snippet(task.file)

        direct_context = "\n\n".join(
            [
                f"Task mode: {task.mode}",
                "Target file design from architecture:",
                target_design,
                "Current target file content:",
                existing_file_context,
            ]
        )

        prompt = FILE_IMPLEMENT_PROMPT.format(
            msg=self.state.user_msg,
            api=dump_json(self._select_arch_context(task.file)),
            threat=dump_json(self._select_threat_context(task.file)),
            task_id=task.task_id,
            file_path=task.file,
            goal=task.goal,
            direct_context=direct_context,
            dependency_context="\n\n".join(dependency_context) or "(none)",
            memory_context="\n\n".join(memory_snippets) or "(none)",
        )

        raw = self.llm.chat(prompt)
        model_file, code = parse_file_block(raw)
        target_file = task.file

        if model_file and model_file != task.file:
            # worker must respect the scheduler-assigned file boundary
            target_file = task.file

        if self._should_strip_code_fence(target_file):
            code = strip_markdown_code_fence(code)

        if not code.strip():
            self._emit(
                {
                    "type": "worker_write_failed",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "reason": "empty code returned",
                    "raw_preview": raw[:300],
                }
            )
            return WorkerResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                produced_file=None,
                summary="empty code returned",
            )

        self.workspace.write_text(target_file, code)
        self.memory.add(
            entry_id=f"file:{target_file}",
            text=f"File: {target_file}\nGoal: {task.goal}\n\n{code[:4000]}",
            metadata={"file": target_file, "task_id": task.task_id, "worker": self.worker_id},
        )
        self._emit(
            {
                "type": "worker_write_done",
                "worker": self.worker_id,
                "task_id": task.task_id,
                "file": task.file,
                "target_file": target_file,
                "chars": len(code),
            }
        )

        return WorkerResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            status=TaskStatus.DONE,
            produced_file=target_file,
            summary="file generated",
        )

    def _edit_target_file(
        self,
        task: CodeTask,
        dependency_files: list[str],
        memory_snippets: list[str],
    ) -> WorkerResult:
        if not self.workspace.exists(task.file):
            return WorkerResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                produced_file=None,
                summary="target file does not exist for edit",
            )

        dependency_context: list[str] = []
        for dep in dependency_files:
            if self.workspace.exists(dep):
                dependency_context.append(self._render_file_snippet(dep))

        current_file_text = self.workspace.read_text(task.file)
        target_design = self._target_file_design(task.file)
        direct_context = "\n\n".join(
            [
                f"Task mode: {task.mode}",
                "Target file design from architecture:",
                target_design,
            ]
        )

        base_prompt = FILE_EDIT_PROMPT.format(
            msg=self.state.user_msg,
            api=dump_json(self._select_arch_context(task.file)),
            threat=dump_json(self._select_threat_context(task.file)),
            task_id=task.task_id,
            file_path=task.file,
            goal=task.goal,
            direct_context=direct_context,
            dependency_context="\n\n".join(dependency_context) or "(none)",
            memory_context="\n\n".join(memory_snippets) or "(none)",
            current_file_numbered=with_line_numbers(current_file_text),
        )

        last_error = ""
        last_raw = ""
        summary = ""
        new_text = current_file_text
        edit_count = 0
        for attempt in range(1, 4):
            if attempt == 1:
                prompt = base_prompt
            else:
                prompt = FILE_EDIT_REPAIR_PROMPT.format(
                    error=last_error or "unknown apply error",
                    previous_response=last_raw[:4000] or "(empty response)",
                    current_file_numbered=with_line_numbers(current_file_text),
                )

            raw = self.llm.chat(prompt)
            last_raw = raw
            try:
                obj = extract_json_object(raw)
                edits = obj.get("edits", [])
                summary = str(obj.get("summary", ""))
                if not isinstance(edits, list) or not edits:
                    raise ValueError("edit response contains no edits")
                if self._should_strip_code_fence(task.file):
                    for edit in edits:
                        if not isinstance(edit, dict):
                            continue
                        if "replacement" not in edit:
                            continue
                        edit["replacement"] = strip_markdown_code_fence(str(edit.get("replacement", "")))
                new_text, edit_count = apply_line_edits(current_file_text, edits)
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt >= 3:
                    self._emit(
                        {
                            "type": "worker_edit_failed",
                            "worker": self.worker_id,
                            "task_id": task.task_id,
                            "file": task.file,
                            "reason": f"parse/apply edit failed after retries: {last_error}",
                            "raw_preview": raw[:300],
                        }
                    )
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        status=TaskStatus.FAILED,
                        produced_file=None,
                        summary=f"edit parse/apply failed after retries: {last_error}",
                    )

        if new_text == current_file_text:
            self._emit(
                {
                    "type": "worker_edit_noop",
                    "worker": self.worker_id,
                    "task_id": task.task_id,
                    "file": task.file,
                    "edit_count": edit_count,
                }
            )
            return WorkerResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.DONE,
                produced_file=task.file,
                summary="edit produced no changes",
            )

        self.workspace.write_text(task.file, new_text)
        self.memory.add(
            entry_id=f"file:{task.file}",
            text=f"File: {task.file}\nGoal: {task.goal}\n\n{new_text[:4000]}",
            metadata={"file": task.file, "task_id": task.task_id, "worker": self.worker_id, "mode": "edit"},
        )
        self._emit(
            {
                "type": "worker_edit_done",
                "worker": self.worker_id,
                "task_id": task.task_id,
                "file": task.file,
                "edit_count": edit_count,
                "summary": summary[:240],
            }
        )
        return WorkerResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            status=TaskStatus.DONE,
            produced_file=task.file,
            summary=f"edited file with {edit_count} line-range edits",
        )

    def run_file_check(self, task: CodeTask, *, dependency_files: Iterable[str], deep: bool = True) -> dict:
        dep_files = list(dependency_files)
        check = self._check_target_file(task, dep_files, deep=deep)
        self.memory.add(
            entry_id=f"check:{task.task_id}:{task.file}",
            text=check["report"][:4000],
            metadata={
                "kind": "check_report",
                "task_id": task.task_id,
                "file": task.file,
                "worker": self.worker_id,
                "deep": deep,
                "needs_fix": check["needs_fix"],
                "issue_count": len(check["issues"]),
            },
        )
        self._emit(
            {
                "type": "worker_check_done",
                "worker": self.worker_id,
                "task_id": task.task_id,
                "file": task.file,
                "deep": deep,
                "needs_fix": check["needs_fix"],
                "issue_count": len(check["issues"]),
            }
        )
        return check

    def run_file_edit(
        self,
        task: CodeTask,
        *,
        dependency_files: Iterable[str],
        memory_snippets: list[str] | None = None,
    ) -> WorkerResult:
        return self._edit_target_file(task, list(dependency_files), memory_snippets or [])

    def _check_target_file(self, task: CodeTask, dependency_files: list[str], *, deep: bool) -> dict:
        if not self.workspace.exists(task.file):
            issue = {"severity": "high", "line": 1, "message": "target file is missing"}
            report = f"[check][{task.file}] needs_fix=true\n- HIGH line 1: target file is missing"
            return {
                "needs_fix": True,
                "issues": [issue],
                "report": report,
                "short_summary": "target file is missing",
            }

        text = self.workspace.read_text(task.file)
        issues: list[dict] = []
        issues.extend(self._local_syntax_issues(task.file, text))
        issues.extend(self._local_linkage_issues(task.file, text))

        if deep:
            issues.extend(self._llm_consistency_issues(task, text, dependency_files))

        dedup: list[dict] = []
        seen: set[tuple[str, int, str]] = set()
        for issue in issues:
            sev = str(issue.get("severity", "medium")).lower()
            if sev not in {"high", "medium", "low"}:
                sev = "medium"
            line = int(issue.get("line", 1) or 1)
            msg = str(issue.get("message", "")).strip()
            key = (sev, line, msg)
            if not msg or key in seen:
                continue
            seen.add(key)
            dedup.append({"severity": sev, "line": line, "message": msg})

        needs_fix = any(x["severity"] in {"high", "medium"} for x in dedup)
        top = dedup[:8]
        if not top:
            report = f"[check][{task.file}] needs_fix=false\n- no issues found"
            short_summary = "no issues"
        else:
            lines = [
                f"[check][{task.file}] needs_fix={'true' if needs_fix else 'false'}",
            ]
            for it in top:
                lines.append(f"- {it['severity'].upper()} line {it['line']}: {it['message']}")
            report = "\n".join(lines)
            short_summary = top[0]["message"][:160]

        return {
            "needs_fix": needs_fix,
            "issues": dedup,
            "report": report,
            "short_summary": short_summary,
        }

    def _local_syntax_issues(self, rel_path: str, text: str) -> list[dict]:
        issues: list[dict] = []
        suffix = Path(rel_path).suffix.lower()

        if suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                issues.append(
                    {
                        "severity": "high",
                        "line": int(exc.lineno or 1),
                        "message": f"python syntax error: {exc.msg}",
                    }
                )
            return issues

        if suffix == ".json":
            try:
                json.loads(text)
            except Exception as exc:
                issues.append(
                    {
                        "severity": "high",
                        "line": 1,
                        "message": f"json parse error: {exc}",
                    }
                )
            return issues

        if suffix in {".js", ".mjs", ".cjs"}:
            node = shutil.which("node")
            if not node:
                return issues
            try:
                abs_file = str(self.workspace.resolve(rel_path))
                proc = subprocess.run(
                    [node, "--check", abs_file],
                    cwd=self.workspace.os_path(),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if proc.returncode != 0:
                    raw = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                    m = re.search(r":(\d+)(?::\d+)?", raw)
                    line = int(m.group(1)) if m else 1
                    issues.append(
                        {
                            "severity": "high",
                            "line": line,
                            "message": f"javascript syntax check failed: {raw[:220]}",
                        }
                    )
            except Exception as exc:
                issues.append(
                    {
                        "severity": "low",
                        "line": 1,
                        "message": f"javascript syntax check unavailable: {exc}",
                    }
                )
            return issues

        return issues

    def _local_linkage_issues(self, rel_path: str, text: str) -> list[dict]:
        suffix = Path(rel_path).suffix.lower()
        if suffix == ".py":
            return self._python_relative_import_issues(rel_path, text)
        if suffix == ".java":
            return self._java_local_import_issues(rel_path, text)
        if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue"}:
            return self._js_relative_import_issues(rel_path, text)
        return []

    def _python_relative_import_issues(self, rel_path: str, text: str) -> list[dict]:
        issues: list[dict] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return issues

        rel_parent = Path(rel_path).parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            level = int(getattr(node, "level", 0) or 0)
            if level <= 0:
                continue
            module = str(getattr(node, "module", "") or "").replace(".", "/")
            base = rel_parent
            for _ in range(max(0, level - 1)):
                base = base.parent
            target = base / module if module else base
            candidates = [
                str((target.with_suffix(".py")).as_posix()),
                str((target / "__init__.py").as_posix()),
            ]
            if any(self.workspace.exists(safe_rel_path(x)) for x in candidates):
                continue
            issues.append(
                {
                    "severity": "medium",
                    "line": int(getattr(node, "lineno", 1) or 1),
                    "message": f"unresolved relative import target: from {'.' * level}{module.replace('/', '.')}",
                }
            )
        return issues

    def _js_relative_import_issues(self, rel_path: str, text: str) -> list[dict]:
        issues: list[dict] = []
        pattern = re.compile(
            r"(?:import|export)\s+(?:[\w*\s{},]*\s+from\s+)?['\"]([^'\"]+)['\"]|"
            r"require\(\s*['\"]([^'\"]+)['\"]\s*\)|"
            r"import\(\s*['\"]([^'\"]+)['\"]\s*\)"
        )

        for m in pattern.finditer(text):
            mod = m.group(1) or m.group(2) or m.group(3) or ""
            mod = mod.strip()
            if not mod.startswith("."):
                continue
            if self._resolve_local_module(rel_path, mod):
                continue
            line = text.count("\n", 0, m.start()) + 1
            issues.append(
                {
                    "severity": "medium",
                    "line": line,
                    "message": f"unresolved relative import path: {mod}",
                }
            )
        return issues

    def _java_local_import_issues(self, rel_path: str, text: str) -> list[dict]:
        issues: list[dict] = []
        lines = text.splitlines()
        local_prefix = self._infer_java_local_prefix(rel_path, text)
        src_root = self._java_source_root(rel_path)
        all_code_files = self.workspace.list_code_files()

        import_pattern = re.compile(r"^\s*import\s+(?:static\s+)?([\w\.\*]+)\s*;")
        for idx, line in enumerate(lines, start=1):
            m = import_pattern.match(line)
            if not m:
                continue
            imp = m.group(1).strip()
            if not imp:
                continue

            # Spring Boot 3.x stacks are jakarta-based. javax imports are high-risk compatibility issues.
            if imp.startswith("javax."):
                issues.append(
                    {
                        "severity": "high",
                        "line": idx,
                        "message": f"legacy javax import may be incompatible with modern Spring Boot: {imp}",
                    }
                )
                continue

            if not local_prefix or not imp.startswith(local_prefix + "."):
                continue

            if imp.endswith(".*"):
                pkg_dir = imp[:-2].replace(".", "/")
                cands: list[str] = []
                if src_root:
                    cands.append(f"{src_root}/{pkg_dir}")

                exists = False
                for cand in cands:
                    try:
                        p = self.workspace.resolve(safe_rel_path(cand))
                    except Exception:
                        continue
                    if p.is_dir() and any(x.suffix.lower() == ".java" for x in p.rglob("*.java")):
                        exists = True
                        break

                if not exists:
                    suffix_dir = "/" + pkg_dir + "/"
                    exists = any(f.endswith(".java") and suffix_dir in f for f in all_code_files)

                if not exists:
                    issues.append(
                        {
                            "severity": "medium",
                            "line": idx,
                            "message": f"unresolved local wildcard import: {imp}",
                        }
                    )
                continue

            java_suffix = imp.replace(".", "/") + ".java"
            cands = []
            if src_root:
                cands.append(f"{src_root}/{java_suffix}")

            exists = any(
                self.workspace.exists(safe_rel_path(cand))
                for cand in cands
            )
            if not exists:
                exists = any(f.endswith("/" + java_suffix) or f == java_suffix for f in all_code_files)

            if not exists:
                issues.append(
                    {
                        "severity": "medium",
                        "line": idx,
                        "message": f"unresolved local java import: {imp}",
                    }
                )

        return issues

    def _java_source_root(self, rel_path: str) -> str:
        path = safe_rel_path(rel_path)
        parts = path.split("/")
        for i in range(len(parts) - 2):
            if parts[i:i + 3] == ["src", "main", "java"]:
                return "/".join(parts[: i + 3])
        return ""

    def _infer_java_local_prefix(self, rel_path: str, text: str) -> str:
        m = re.search(r"^\s*package\s+([\w\.]+)\s*;", text, flags=re.MULTILINE)
        if m:
            pkg = m.group(1).strip()
            parts = pkg.split(".")
            if len(parts) >= 2:
                return ".".join(parts[:2])
            return pkg

        src_root = self._java_source_root(rel_path)
        if not src_root:
            return ""
        path_parts = src_root.split("/")
        if len(path_parts) < 2:
            return ""
        # src/main/java/<a>/<b>/...
        # derive a coarse local prefix a.b for project-local package checks
        try:
            idx = path_parts.index("java")
        except ValueError:
            return ""
        tail = path_parts[idx + 1:]
        if len(tail) >= 2:
            return ".".join(tail[:2])
        if len(tail) == 1:
            return tail[0]
        return ""

    def _resolve_local_module(self, file_rel: str, spec: str) -> bool:
        base = Path(file_rel).parent
        try:
            raw = safe_rel_path(str((base / spec).as_posix()))
        except Exception:
            return False
        cands = [raw]

        ext = Path(raw).suffix.lower()
        if not ext:
            cands.extend(
                [
                    raw + ".js",
                    raw + ".ts",
                    raw + ".tsx",
                    raw + ".jsx",
                    raw + ".vue",
                    raw + ".json",
                    raw + ".py",
                    raw + "/index.js",
                    raw + "/index.ts",
                    raw + "/index.vue",
                    raw + "/__init__.py",
                ]
            )

        for cand in cands:
            try:
                if self.workspace.exists(safe_rel_path(cand)):
                    return True
            except Exception:
                continue
        return False

    def _llm_consistency_issues(self, task: CodeTask, current_file_text: str, dependency_files: list[str]) -> list[dict]:
        dependency_context: list[str] = []
        for dep in dependency_files:
            if self.workspace.exists(dep):
                dependency_context.append(self._render_file_snippet(dep))

        numbered = with_line_numbers(current_file_text)
        lines = numbered.splitlines()
        if len(lines) > 450:
            numbered = "\n".join(lines[:450])

        prompt = FILE_CHECK_PROMPT.format(
            msg=self.state.user_msg,
            api=dump_json(self._select_arch_context(task.file)),
            threat=dump_json(self._select_threat_context(task.file)),
            task_id=task.task_id,
            file_path=task.file,
            goal=task.goal,
            dependency_context="\n\n".join(dependency_context) or "(none)",
            current_file_numbered=numbered,
        )
        try:
            raw = self.llm.chat(prompt, temperature=0.1)
            obj = extract_json_object(raw)
        except Exception:
            return []

        out: list[dict] = []
        for item in obj.get("issues", []):
            if not isinstance(item, dict):
                continue
            msg = str(item.get("message", "")).strip()
            if not msg:
                continue
            sev = str(item.get("severity", "medium")).lower()
            if sev not in {"high", "medium", "low"}:
                sev = "medium"
            try:
                line = int(item.get("line", 1) or 1)
            except Exception:
                line = 1
            out.append(
                {
                    "severity": sev,
                    "line": line,
                    "message": msg,
                }
            )
        return out

    def _render_memory_snippet(self, score: float, text: str, metadata: dict) -> str:
        return f"[score={score:.3f}] metadata={metadata}\n{text[:1200]}"

    def _render_file_snippet(self, rel_path: str) -> str:
        try:
            content = self.workspace.read_text(rel_path)
        except Exception:
            return f"File: {rel_path}\n(unreadable)"
        return f"File: {rel_path}\n{content}"

    def _target_file_design(self, rel_path: str) -> str:
        for item in self.state.arch.get("files", []):
            name = str(item.get("name", "")).strip()
            path = str(item.get("path", "./")).strip()
            if not name:
                continue
            expected = name if path in {"./", ".", ""} else f"{path.rstrip('/')}/{name}"
            try:
                expected = safe_rel_path(expected)
            except Exception:
                continue
            if expected == rel_path:
                return json.dumps(item, ensure_ascii=False, indent=2)
        return "(not found in architecture; infer from user goal and existing code)"

    def _select_arch_context(self, rel_path: str, max_files: int = 30) -> dict:
        files = self.state.arch.get("files", [])
        if not files:
            return {"files": []}

        rel_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
        target_top = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path

        scored: list[tuple[int, dict]] = []
        for item in files:
            name = str(item.get("name", "")).strip()
            path = str(item.get("path", "./")).strip()
            if not name:
                continue
            candidate = name if path in {"", ".", "./"} else f"{path.rstrip('/')}/{name}"
            try:
                candidate = safe_rel_path(candidate)
            except Exception:
                continue

            score = 0
            if candidate == rel_path:
                score += 100
            if rel_dir and candidate.startswith(rel_dir + "/"):
                score += 40
            if candidate.split("/", 1)[0] == target_top:
                score += 20
            low = candidate.lower()
            if any(k in low for k in ("config", "setting", "schema", "model", "type", "service", "core")):
                score += 12
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [self._compact_arch_item(item) for _, item in scored[:max_files]]
        all_files_index: list[str] = []
        for item in files[:2000]:
            name = str(item.get("name", "")).strip()
            path = str(item.get("path", "./")).strip()
            if not name:
                continue
            rel = name if path in {"", ".", "./"} else f"{path.rstrip('/')}/{name}"
            try:
                all_files_index.append(safe_rel_path(rel))
            except Exception:
                continue
        return {"all_files_index": all_files_index, "files": selected}

    def _compact_arch_item(self, item: dict) -> dict:
        out = {
            "name": item.get("name", ""),
            "path": item.get("path", "./"),
            "description": item.get("description", ""),
            "classes": [],
            "functions": [],
        }

        classes = item.get("classes", [])
        if isinstance(classes, list):
            for cls in classes[:40]:
                if not isinstance(cls, dict):
                    continue
                out["classes"].append(
                    {
                        "class_name": cls.get("class_name", ""),
                        "members": cls.get("members", [])[:60] if isinstance(cls.get("members", []), list) else [],
                    }
                )

        funcs = item.get("functions", [])
        if isinstance(funcs, list):
            for fn in funcs[:120]:
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

    def _select_threat_context(self, rel_path: str, max_entries: int = 10) -> dict:
        tm_json = self.state.threat_model_json or {}
        if not isinstance(tm_json, dict) or not tm_json:
            if self.state.threat_model_modules:
                relevant_modules: list[dict] = []
                rel_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
                for mod in self.state.threat_model_modules:
                    if not isinstance(mod, dict):
                        continue
                    files = mod.get("files", [])
                    if not isinstance(files, list):
                        files = []
                    hit = rel_path in files or any(rel_dir and f.startswith(rel_dir + "/") for f in files)
                    if hit:
                        relevant_modules.append(
                            {
                                "module": mod.get("module", ""),
                                "files": files[:20],
                                "threat_excerpt": str(mod.get("threat_model", ""))[:1600],
                            }
                        )
                    if len(relevant_modules) >= 4:
                        break
                if relevant_modules:
                    return {"modules": relevant_modules}
            return {"raw_excerpt": self.state.threat_model[:4000], "functions": []}

        entries = tm_json.get("functions", [])
        if not isinstance(entries, list):
            entries = []

        rel_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
        scored: list[tuple[int, dict]] = []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            fpath = str(ent.get("file", "")).strip()
            score = 0
            if fpath == rel_path:
                score += 100
            if rel_dir and fpath.startswith(rel_dir + "/"):
                score += 30
            if rel_path.split("/", 1)[0] == fpath.split("/", 1)[0] and fpath:
                score += 10
            scored.append((score, ent))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [ent for _, ent in scored[:max_entries]]
        return {
            "global_context": tm_json.get("global_context", {}),
            "attacker_model": tm_json.get("attacker_model", {}),
            "functions": selected,
        }

    def _run_readonly_command(self, command: str) -> str:
        allowed = {"ls", "rg", "cat", "find", "wc", "head", "tail"}
        try:
            args = shlex.split(command)
        except Exception:
            return "invalid command format"
        if not args:
            return "empty command"
        if args[0] not in allowed:
            return f"command not allowed: {args[0]}"
        try:
            proc = subprocess.run(
                args,
                cwd=self.workspace.os_path(),
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception as exc:
            return f"command error: {exc}"
        output = (proc.stdout or "") + (proc.stderr or "")
        return output.strip() or f"exit={proc.returncode}"

    def _emit(self, event: dict) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event)
        except Exception:
            pass

    def _shorten_dict(self, value: dict, max_len: int) -> dict:
        out: dict = {}
        for k, v in value.items():
            if isinstance(v, str) and len(v) > max_len:
                out[k] = v[:max_len] + "..."
            else:
                out[k] = v
        return out

    def _should_strip_code_fence(self, rel_path: str) -> bool:
        low = rel_path.lower()
        return not low.endswith((".md", ".markdown", ".mdx"))
