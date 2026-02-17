from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass
class CodeTask:
    task_id: str
    file: str
    goal: str
    depends_on: set[str] = field(default_factory=set)
    priority: int = 100
    mode: str = "update_or_create"  # create_only | update_only | update_or_create
    source: str = "planned"
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: str | None = None
    attempts: int = 0
    wait_reason: str = ""
    last_error: str = ""


@dataclass
class WorkerResult:
    task_id: str
    worker_id: str
    status: TaskStatus
    produced_file: str | None = None
    summary: str = ""
    wait_reason: str = ""


@dataclass
class MemoryEntry:
    entry_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class ProjectState:
    user_msg: str = ""
    prd: str = ""
    arch: dict[str, Any] = field(default_factory=lambda: {"files": []})
    threat_model: str = ""
    threat_model_json: dict[str, Any] = field(default_factory=dict)
    threat_model_modules: list[dict[str, Any]] = field(default_factory=list)
    call_chain: str = ""
    readme: str = ""
    code: list[str] = field(default_factory=list)
    feedback_events: list[dict[str, Any]] = field(default_factory=list)
    workspace_manifest: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkspaceFileInfo:
    path: str
    lines: int
    chars: int
    summary: str


class TaskWaiting(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
