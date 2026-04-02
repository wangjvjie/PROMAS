"""Pydantic models for PROMAS backend."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Pipeline Stages ───────────────────────────────────────────────────────────

class Stage(str, Enum):
    AUTO = "auto"
    PRD = "prd"
    ARCHITECTURE = "architecture"
    THREAT_MODEL = "threat_model"
    CODE = "code"
    VERIFY = "verify"


# ── Architecture Models ───────────────────────────────────────────────────────

class Parameter(BaseModel):
    name: str = ""
    type: str = "any"

    @field_validator("name", "type", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return str(v) if v is not None else "any"


class FunctionSpec(BaseModel):
    name: str
    input_parameters: list[Parameter] = []
    output_parameters: list[Parameter] = []
    description: str = ""


class ClassSpec(BaseModel):
    class_name: str
    members: list[Parameter] = []
    methods: list[str] = []


class FileDesign(BaseModel):
    name: str
    path: str = "./"
    description: str = ""
    classes: list[ClassSpec] = []
    functions: list[FunctionSpec] = []
    dependencies: list[str] = []


class ArchitectureDesign(BaseModel):
    files: list[FileDesign] = []


# ── Threat Model ──────────────────────────────────────────────────────────────

class ThreatEntry(BaseModel):
    function: str
    threats: list[str] = []
    protections: list[str] = []


class ThreatModel(BaseModel):
    global_context: str = ""
    entries: list[ThreatEntry] = []
    raw_text: str = ""


# ── SSE Events ────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    AGENT_THINK = "agent_think"
    AGENT_ACT = "agent_act"
    AGENT_OBSERVE = "agent_observe"
    FILE_WRITTEN = "file_written"
    FILE_EDITED = "file_edited"
    ERROR = "error"
    WARN = "warn"
    RETRY = "retry"
    LOG = "log"
    DONE = "done"


class SSEvent(BaseModel):
    type: EventType
    stage: str = ""
    file: str = ""
    content: str = ""
    step: int = 0


# ── API Request / Response ────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model_config = {"populate_by_name": True}

    prompt: str = Field(..., min_length=1)
    begin_stage: Stage = Stage.AUTO
    threat_model_mode: str = "simple"   # "simple" | "full"
    threat_model_candidates: int = 3    # k candidates for judge (full mode)
    work_dir: str = "./workspace"
    model: str = ""
    max_steps: int = Field(25, alias="max_react_steps")
    max_retries: int = 3
    max_verify_rounds: int = 3


class ProjectStatus(BaseModel):
    stage: str = ""
    files_written: list[str] = []
    total_files: int = 0
    is_running: bool = False


class FileContent(BaseModel):
    path: str
    content: str
    language: str = ""
