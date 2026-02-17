# PROMAS Parallel Refactor

## Core Components
- `orchestrator.py`: stage pipeline and end-to-end control.
- `scheduler.py`: dependency-aware parallel task scheduler.
- `agents.py`: `ThreatModelAgent` + reflective `CodeWorkerAgent`.
- `memory.py`: shared embedding memory pool (hash embedding by default).
- `workspace.py`: safe project file read/write utilities.
- `prompts.py`: all LLM prompts (planning, reflection, coding, security).

## Execution Flow
1. Scan existing workspace files and build `workspace_manifest.json`.
2. Build PRD.
3. Build system architecture.
4. Generate threat model (supports call-chain parallel analysis).
5. Plan file-level tasks + dependencies using both architecture and existing files.
6. Run parallel worker agents with shared memory retrieval.
7. Persist metadata, progress stream, and generate README.

## Key Guarantees
- Threat modeling stage is required before code generation.
- File-level task boundaries are enforced by scheduler.
- Agents can return `wait` when dependencies are not ready.
- Shared memory supports retrieval by embedding similarity.
- Workers can invoke a restricted read-only command set (`rg`, `ls`, `cat`, etc.) during reflection loops.
- Task planning supports `create_only`, `update_only`, and `update_or_create`.
- Direct prompt context is reserved for high-priority information (target design/current file/dependencies), while long-tail context comes from embedding retrieval.
- Runtime feedback events are streamed and persisted to `progress.jsonl`.
