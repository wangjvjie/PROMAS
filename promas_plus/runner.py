from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

try:
    from .api import LLMClient
    from .orchestrator import ParallelMASecDev
except ImportError:
    # Allow direct execution: python src/promas_parallel/runner.py
    from promas_parallel.api import LLMClient
    from promas_parallel.orchestrator import ParallelMASecDev


@dataclass
class RunnerConfig:
    prompt: str = "在这里写你的需求"
    begin_stage: str = "auto"
    work_dir: str = "./promas-parallel-workspace"
    workers: int = 4
    system_design_workers: int = 4
    threat_mode: str = "call_chain"
    threat_model_modular: bool = True
    threat_module_max_files: int = 12
    threat_model_workers: int = 4
    threat_context_max_files: int = 32
    reuse_existing_arch_design: bool = True
    enable_system_design_consistency_check: bool = False
    auto_apply_consistency_fixes: bool = True
    consistency_max_updates: int = 20
    enable_global_code_check: bool = False
    global_code_check_rounds: int = 2
    enable_global_project_review: bool = True
    enable_compile_fix_loop: bool = False
    compile_fix_rounds: int = 3
    compile_timeout_seconds: int = 240
    use_llm_embeddings: bool = True
    show_feedback_stream: bool = True
    persist_feedback: bool = True

    # Chat endpoint (DeepSeek/OpenAI-compatible)
    chat_base_url: str | None = os.getenv("PROMAS_BASE_URL", "https://api.deepseek.com")
    chat_api_key: str | None = os.getenv("PROMAS_API_KEY")
    chat_model: str = os.getenv("PROMAS_CHAT_MODEL", "deepseek-chat")

    # Embedding endpoint (usually OpenAI)
    embedding_base_url: str | None = os.getenv("PROMAS_EMBED_BASE_URL")
    embedding_api_key: str | None = os.getenv("PROMAS_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
    embedding_model: str = os.getenv("PROMAS_EMBED_MODEL", "text-embedding-3-small")


# =========================
# Editable local config zone
# =========================
RUNNER_CONFIG = RunnerConfig(
    prompt="Implement a Java service that accepts JSON or YAML input and deserializes it into domain objects for further processing.",
    begin_stage="auto",
    work_dir="./promas-parallel-workspace",
    workers=1,
    system_design_workers=1,
    threat_mode="simple",
    threat_model_modular=True,
    threat_module_max_files=12,
    threat_model_workers=4,
    threat_context_max_files=32,
    reuse_existing_arch_design=True,
    enable_system_design_consistency_check=False,
    auto_apply_consistency_fixes=True,
    consistency_max_updates=20,
    enable_global_code_check=False,
    global_code_check_rounds=0,
    enable_global_project_review=False,
    enable_compile_fix_loop=False,
    compile_fix_rounds=3,
    compile_timeout_seconds=240,
    use_llm_embeddings=True,
    show_feedback_stream=True,
    persist_feedback=True,
)

# True: ignore CLI, run with RUNNER_CONFIG directly
# False: parse CLI args
USE_LOCAL_CONFIG = True


def run_with_config(cfg: RunnerConfig) -> None:
    llm = LLMClient(
        base_url=cfg.chat_base_url,
        api_key=cfg.chat_api_key,
        embedding_base_url=cfg.embedding_base_url,
        embedding_api_key=cfg.embedding_api_key,
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
    )

    def _feedback_printer(event: dict) -> None:
        et = event.get("type", "event")
        msg = event.get("message", "")
        details = event.get("details", {})
        task_id = details.get("task_id")
        file_path = details.get("file")
        stage = details.get("stage")
        worker = details.get("worker")
        step = details.get("step")
        action = details.get("action")
        wave = details.get("wave")
        module = details.get("module")
        command = details.get("command")
        target_file = details.get("target_file")
        status = details.get("status")
        reason = details.get("reason")
        edit_count = details.get("edit_count")
        issue_count = details.get("issue_count")
        needs_fix = details.get("needs_fix")
        deep_check = details.get("deep")
        completed = details.get("completed")
        total = details.get("total")
        remaining = details.get("remaining")
        extra = []
        if stage:
            extra.append(f"stage={stage}")
        if worker:
            extra.append(f"worker={worker}")
        if task_id:
            extra.append(f"task={task_id}")
        if file_path:
            extra.append(f"file={file_path}")
        if step is not None:
            extra.append(f"step={step}")
        if action:
            extra.append(f"action={action}")
        if wave is not None:
            extra.append(f"wave={wave}")
        if module:
            extra.append(f"module={module}")
        if target_file and target_file != file_path:
            extra.append(f"target={target_file}")
        if command:
            extra.append(f"command={command}")
        if status:
            extra.append(f"status={status}")
        if reason:
            reason_text = str(reason)
            extra.append(f"reason={reason_text[:120]}")
        if edit_count is not None:
            extra.append(f"edits={edit_count}")
        if issue_count is not None:
            extra.append(f"issues={issue_count}")
        if needs_fix is not None:
            extra.append(f"needs_fix={needs_fix}")
        if deep_check is not None:
            extra.append(f"deep={deep_check}")
        if completed is not None and total is not None:
            extra.append(f"progress={completed}/{total}")
        if remaining is not None:
            extra.append(f"remaining={remaining}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        print(f"[{et}] {msg}{suffix}")

    agent = ParallelMASecDev(
        work_dir=cfg.work_dir,
        code_workers=cfg.workers,
        system_design_workers=cfg.system_design_workers,
        threat_model_mode=cfg.threat_mode,
        threat_model_modular=cfg.threat_model_modular,
        threat_module_max_files=cfg.threat_module_max_files,
        threat_model_workers=cfg.threat_model_workers,
        threat_context_max_files=cfg.threat_context_max_files,
        reuse_existing_arch_design=cfg.reuse_existing_arch_design,
        enable_system_design_consistency_check=cfg.enable_system_design_consistency_check,
        auto_apply_consistency_fixes=cfg.auto_apply_consistency_fixes,
        consistency_max_updates=cfg.consistency_max_updates,
        enable_global_code_check=cfg.enable_global_code_check,
        global_code_check_rounds=cfg.global_code_check_rounds,
        enable_global_project_review=cfg.enable_global_project_review,
        enable_compile_fix_loop=cfg.enable_compile_fix_loop,
        compile_fix_rounds=cfg.compile_fix_rounds,
        compile_timeout_seconds=cfg.compile_timeout_seconds,
        use_llm_embeddings=cfg.use_llm_embeddings,
        llm_client=llm,
        feedback_callback=_feedback_printer if cfg.show_feedback_stream else None,
        persist_feedback=cfg.persist_feedback,
    )
    agent.run(cfg.prompt, begin_stage=cfg.begin_stage)


def main() -> None:
    if USE_LOCAL_CONFIG:
        run_with_config(RUNNER_CONFIG)
        return

    # Optional CLI fallback
    parser = argparse.ArgumentParser(description="PROMAS parallel orchestrator")
    parser.add_argument("prompt", help="user requirement")
    parser.add_argument(
        "--begin-stage",
        default="auto",
        choices=["auto", "prd", "system_design", "threat_model", "code", "readme"],
    )
    parser.add_argument("--work-dir", default="./promas-parallel-workspace")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--system-design-workers", type=int, default=4)
    parser.add_argument("--threat-mode", default="call_chain", choices=["simple", "call_chain"])
    parser.add_argument("--no-threat-model-modular", action="store_true", help="disable module-based threat modeling")
    parser.add_argument("--threat-module-max-files", type=int, default=12)
    parser.add_argument("--threat-model-workers", type=int, default=4)
    parser.add_argument("--threat-context-max-files", type=int, default=32)
    parser.add_argument("--no-reuse-arch-design", action="store_true", help="always redesign files in system_design")
    parser.add_argument("--no-system-design-consistency-check", action="store_true", help="disable final system design consistency review")
    parser.add_argument("--no-auto-apply-consistency-fixes", action="store_true", help="do not auto-apply consistency file_updates")
    parser.add_argument("--consistency-max-updates", type=int, default=20)
    parser.add_argument("--no-global-code-check", action="store_true", help="disable final global code check+fix pass")
    parser.add_argument("--global-code-check-rounds", type=int, default=2)
    parser.add_argument("--no-global-project-review", action="store_true", help="disable whole-project LLM review before compile")
    parser.add_argument("--no-compile-fix-loop", action="store_true", help="disable compile->fix loop")
    parser.add_argument("--compile-fix-rounds", type=int, default=3)
    parser.add_argument("--compile-timeout-seconds", type=int, default=240)
    parser.add_argument("--llm-embeddings", action="store_true", help="use model API for embeddings")
    parser.add_argument("--no-feedback-stream", action="store_true", help="disable realtime feedback prints")
    parser.add_argument("--no-persist-feedback", action="store_true", help="do not write progress.jsonl")
    parser.add_argument("--chat-base-url", default=os.getenv("PROMAS_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--chat-api-key", default=os.getenv("PROMAS_API_KEY"))
    parser.add_argument("--chat-model", default=os.getenv("PROMAS_CHAT_MODEL", "deepseek-chat"))
    parser.add_argument("--embedding-base-url", default=os.getenv("PROMAS_EMBED_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.getenv("PROMAS_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--embedding-model", default=os.getenv("PROMAS_EMBED_MODEL", "text-embedding-3-small"))
    args = parser.parse_args()

    run_with_config(
        RunnerConfig(
            prompt=args.prompt,
            begin_stage=args.begin_stage,
            work_dir=args.work_dir,
            workers=args.workers,
            system_design_workers=args.system_design_workers,
            threat_mode=args.threat_mode,
            threat_model_modular=not args.no_threat_model_modular,
            threat_module_max_files=args.threat_module_max_files,
            threat_model_workers=args.threat_model_workers,
            threat_context_max_files=args.threat_context_max_files,
            reuse_existing_arch_design=not args.no_reuse_arch_design,
            enable_system_design_consistency_check=not args.no_system_design_consistency_check,
            auto_apply_consistency_fixes=not args.no_auto_apply_consistency_fixes,
            consistency_max_updates=args.consistency_max_updates,
            enable_global_code_check=not args.no_global_code_check,
            global_code_check_rounds=args.global_code_check_rounds,
            enable_global_project_review=not args.no_global_project_review,
            enable_compile_fix_loop=not args.no_compile_fix_loop,
            compile_fix_rounds=args.compile_fix_rounds,
            compile_timeout_seconds=args.compile_timeout_seconds,
            use_llm_embeddings=args.llm_embeddings,
            show_feedback_stream=not args.no_feedback_stream,
            persist_feedback=not args.no_persist_feedback,
            chat_base_url=args.chat_base_url,
            chat_api_key=args.chat_api_key,
            chat_model=args.chat_model,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
            embedding_model=args.embedding_model,
        )
    )


if __name__ == "__main__":
    main()
