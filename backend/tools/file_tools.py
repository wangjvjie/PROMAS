"""Read-only file access tools — safe to run concurrently."""

from __future__ import annotations

from .base import Tool
from ..project.state import ProjectState


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the complete content of an already-written project file.\n\n"
        "Use this when:\n"
        "- You need to see how a dependency is implemented before importing from it\n"
        "- You need the exact function signatures in a written file\n"
        "- You're about to call edit_file and need the current content\n\n"
        "Tip: If you need to read multiple files, issue all read_file calls in one "
        "response — they run in parallel. Do not read one, wait, then read the next."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path, e.g. 'src/auth.py' or 'package.json'",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str = "", **_) -> str:
        if not path:
            return "[Error] path is required"
        return self.state.read_file(path)


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Search all written files for a symbol, pattern, or keyword.\n\n"
        "Use this when:\n"
        "- You need to find where a class or function is defined\n"
        "- You want to verify a symbol exists before importing it\n"
        "- You need to find all usages of a name across files\n\n"
        "Returns matching lines with file path and line number. "
        "Can be called in parallel with read_file and other read-only tools."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term: function name, class name, import statement, etc.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str = "", **_) -> str:
        if not query:
            return "[Error] query is required"
        return self.state.search_code(query)


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List all project files with status: written (✓) or planned (○), "
        "plus exports and dependencies.\n\n"
        "Use this for orientation — to see what's written and what remains. "
        "The file index is also shown in the user prompt, so only call this "
        "if you need the most up-to-date status mid-session."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_) -> str:
        return self.state.get_file_index()
