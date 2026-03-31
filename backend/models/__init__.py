"""Pydantic models for API requests/responses and internal data structures."""

from __future__ import annotations
import json
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from enum import Enum


# ── Pipeline Stages ──────────────────────────────────────────────────────────

class Stage(str, Enum):
    AUTO = "auto"  # auto-detect: resume from first incomplete stage
    PRD = "prd"
    ARCHITECTURE = "architecture"
    THREAT_MODEL = "threat_model"
    CODE = "code"
    VERIFY = "verify"


# ── Architecture Models ──────────────────────────────────────────────────────

class Parameter(BaseModel):
    name: str = ""
    type: str = "any"

    @field_validator("name", "type", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        """LLMs sometimes return dicts/lists instead of strings — coerce them."""
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
    dependencies: list[str] = []  # files this file imports from


class ArchitectureDesign(BaseModel):
    files: list[FileDesign] = []


# ── Threat Model ─────────────────────────────────────────────────────────────

class ThreatEntry(BaseModel):
    function: str
    threats: list[str] = []
    protections: list[str] = []


class ThreatModel(BaseModel):
    global_context: str = ""
    entries: list[ThreatEntry] = []
    raw_text: str = ""  # keep the full markdown for the agent to query


# ── ReAct Agent Models ───────────────────────────────────────────────────────

class ToolCall(BaseModel):
    tool: str
    arguments: dict = {}

    @field_validator("arguments", mode="before")
    @classmethod
    def coerce_arguments(cls, v: Any) -> dict:
        """Ensure arguments is always a dict, even if LLM returns a string."""
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            # Try JSON parse first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"raw": v}
        if v is None:
            return {}
        return {"raw": str(v)}


class AgentStep(BaseModel):
    step: int = 0
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: str = ""


class AgentResult(BaseModel):
    file_name: str
    steps: list[AgentStep] = []
    final_code: str = ""
    success: bool = False


# ── SSE Event Models ─────────────────────────────────────────────────────────

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


# ── API Request / Response ───────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User's project description")
    begin_stage: Stage = Stage.AUTO
    threat_model_mode: str = "simple"  # "simple" (default) or "full" (paper-aligned)
    threat_model_candidates: int = 3   # k candidates for LLM-Judge (full mode only)
    work_dir: str = "./workspace"
    model: str = ""
    max_react_steps: int = 15
    context_token_budget: int = 6000
    max_retries: int = 3  # retries per stage/batch/file on failure
    max_verify_rounds: int = 3  # how many build-fix cycles in verify stage


class ProjectStatus(BaseModel):
    stage: str = ""
    files_written: list[str] = []
    total_files: int = 0
    is_running: bool = False


class FileContent(BaseModel):
    path: str
    content: str
    language: str = ""