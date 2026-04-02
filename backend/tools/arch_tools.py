"""Architecture and threat model query tools + file picker."""

from __future__ import annotations

from .base import Tool
from ..project.state import ProjectState


class ReadArchitectureTool(Tool):
    name = "read_architecture"
    description = (
        "Get the designed API spec (classes, functions, typed parameters) for a specific file.\n\n"
        "Use this when:\n"
        "- You need parameter descriptions or return type details beyond the summary\n"
        "- You want to confirm a function's exact signature before calling it\n\n"
        "Note: The full architecture summary is already in your system prompt. "
        "Only call this if you need detail that was truncated or isn't in the summary. "
        "Can be called in parallel with read_file and read_threats."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "File name or path to look up, e.g. 'auth.py' or 'src/db.ts'",
                },
            },
            "required": ["file_name"],
        }

    async def execute(self, file_name: str = "", **_) -> str:
        if not file_name:
            return "[Error] file_name is required"
        return self.state.get_file_design_text(file_name)


class ReadThreatsTool(Tool):
    name = "read_threats"
    description = (
        "Get threat model entries and required security protections for a file.\n\n"
        "Use this early — before writing a file with complex security logic. "
        "Returns relevant CWEs, attack vectors, and specific protections to implement.\n\n"
        "Can be called in parallel with read_file and read_architecture."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "File name or path, e.g. 'auth.py'",
                },
            },
            "required": ["file_name"],
        }

    async def execute(self, file_name: str = "", **_) -> str:
        if not file_name:
            return "[Error] file_name is required"
        return self._get_threats(file_name)

    def _get_threats(self, file_name: str) -> str:
        tm = self.state.threat_model
        key = self.state._resolve_key(file_name)

        relevant = [
            e for e in tm.entries
            if file_name in e.function or key in e.function
        ]
        if relevant:
            lines = []
            for e in relevant:
                lines.append(f"### {e.function}")
                for t in e.threats:
                    lines.append(f"  - {t}")
                lines.append("  Protections:")
                for p in e.protections:
                    lines.append(f"  - {p}")
            return "\n".join(lines)

        if tm.raw_text:
            chunks = tm.raw_text.split("### Function:")
            for chunk in chunks:
                if file_name in chunk or key.replace("/", ".") in chunk:
                    return f"### Function:{chunk}"

        return (
            f"[Info] No specific threats found for {file_name}. "
            "Apply general security practices: validate inputs, sanitize outputs, "
            "use parameterized queries, avoid shell injection."
        )


class PickNextFileTool(Tool):
    """Select the next file to write. Updates WriteFileTool.current_file."""

    name = "pick_next_file"
    description = (
        "Select the next file to implement. ALWAYS call this first, before any read or finish.\n\n"
        "Returns:\n"
        "- The file's detailed design (classes, functions, signatures)\n"
        "- Dependency status (which imports are already written vs pending)\n"
        "- What other files will be unblocked after writing this one\n"
        "- Focused architecture (target file + deps + dependents in full detail)\n\n"
        "Pick order: config/manifest → shared types → utilities → core modules "
        "→ entry points → barrel/index files → README\n\n"
        "Prefer files whose dependencies are all written (marked ✓). "
        "Writing a file with unwritten dependencies (marked ○) is allowed but "
        "you must check the architecture carefully since you can't read the actual code."
    )
    is_read_only = False  # mutates write_tool.current_file

    def __init__(self, state: ProjectState, write_tool):
        self.state = state
        self.write_tool = write_tool

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "File to implement, e.g. 'src/auth.py' or 'package.json'",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this file now (dependency order, unblocks others, etc.)",
                },
            },
            "required": ["file_name"],
        }

    async def execute(self, file_name: str = "", reason: str = "", **_) -> str:
        if not file_name:
            return "[Error] file_name is required"
        resolved_key, observation = self.state.pick_next_file(file_name)
        if resolved_key:
            self.write_tool.current_file = resolved_key
        return observation
