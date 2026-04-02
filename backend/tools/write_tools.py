"""File write and edit tools — non-read-only, run serially."""

from __future__ import annotations

import ast
import json
import logging
import os
import re

from .base import Tool
from ..project.state import ProjectState

logger = logging.getLogger("promas")


class WriteFileTool(Tool):
    """Submit the complete code for the current file (finish).

    current_file is set by PickNextFileTool (for pick→write flow) or
    directly by the stage (for generate_file with a known target).
    """

    name = "finish"
    description = (
        "Submit the COMPLETE, final source code for the current file.\n\n"
        "Rules:\n"
        "- Code must be complete and runnable — no `pass`, `TODO`, `...`, or placeholders\n"
        "- Raw source only — NO markdown fences (no ```python or ``` wrappers)\n"
        "- All imports must reference symbols that actually exist in those files\n"
        "- All security protections from the threat model must be implemented\n\n"
        "Before calling finish:\n"
        "- If you haven't read a dependency, call read_file or search_code first\n"
        "- Verify every symbol you import exists in the file you're importing from\n"
        "- Config files (JSON, YAML, TOML) must be syntactically valid\n\n"
        "If finish returns an error (validation failed), fix the issue and call again."
    )
    is_read_only = False

    def __init__(self, state: ProjectState, file_name: str = ""):
        self.state = state
        self.current_file: str = file_name

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete file content — raw source, no markdown fences",
                },
            },
            "required": ["code"],
        }

    async def execute(self, code: str = "", **kwargs) -> str:
        if not self.current_file:
            return "[Error] No file selected. Call pick_next_file first."

        # Recover from truncated JSON output
        if kwargs.get("_parse_failed"):
            raw = kwargs.get("_raw_truncated", "")
            logger.warning(f"finish() truncated JSON ({len(raw)} chars)")
            code = _salvage_code(raw)
            if not code:
                return (
                    "[Error] finish() was truncated mid-output. "
                    "Minimize comments and docstrings to reduce output size, then try again."
                )

        if not code or not code.strip():
            return "[Error] code was empty. Call finish with complete file content."

        validation = _validate_code(self.current_file, code)
        if not validation["ok"]:
            return (
                f"[Error] Validation failed: {validation['error']}. "
                "Fix the issue and call finish again."
            )

        # Distinguish create vs update (like claude-code's FileWriteTool)
        is_update = self.current_file in self.state.get_written()
        self.state.write_file(self.current_file, code)
        lines = len(code.splitlines())
        if is_update:
            return f"✅ The file {self.current_file} has been updated successfully. ({lines} lines)"
        return f"✅ File created successfully at: {self.current_file} ({lines} lines)"


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Fix a file by replacing an exact section of text.\n\n"
        "Usage:\n"
        "- You MUST call read_file first to see the current content\n"
        "- old_text must match EXACTLY once — identical whitespace and indentation\n"
        "- Use the SMALLEST old_text that uniquely identifies the location "
        "(typically 2-4 lines including surrounding context)\n"
        "- new_text replaces old_text exactly — do not change unrelated code\n\n"
        "If the edit fails with 'not found': the file changed or your whitespace "
        "is wrong. Call read_file again to get the current content."
    )
    is_read_only = False

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace — must match once, including whitespace",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, path: str = "", old_text: str = "", new_text: str = "", **_) -> str:
        if not path or old_text == "":
            return "[Error] path and old_text are required"
        return self.state.edit_file(path, old_text, new_text)


# ── Code validation ───────────────────────────────────────────────────────────

def _validate_code(file_name: str, code: str) -> dict:
    if not code or not code.strip():
        return {"ok": False, "error": "Empty code"}

    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    if ext == ".py":
        try:
            ast.parse(code)
        except SyntaxError as e:
            total = len(code.splitlines())
            truncated = e.lineno and e.lineno >= total - 2
            truncation_msgs = {
                "unexpected EOF while parsing",
                "EOF while scanning triple-quoted string literal",
                "EOF while scanning string literal",
                "expected an indented block after",
            }
            hint = " (likely truncated — minimize comments)" \
                if truncated or e.msg in truncation_msgs else ""
            return {"ok": False, "error": f"SyntaxError line {e.lineno}: {e.msg}{hint}"}

        last = code.rstrip().splitlines()[-1].rstrip() if code.strip() else ""
        if any([
            last.endswith("=") and not last.endswith("=="),
            last.endswith(","),
            last.endswith("("),
        ]):
            return {"ok": False, "error": f"Incomplete last line: '{last[-60:]}' (truncated?)"}
        return {"ok": True, "error": ""}

    if ext == ".json":
        try:
            json.loads(code)
            return {"ok": True, "error": ""}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"JSON error: {e.msg} at line {e.lineno}"}

    CODE_EXTS = {".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs"}
    if ext in CODE_EXTS:
        opens = code.count("(") + code.count("[") + code.count("{")
        closes = code.count(")") + code.count("]") + code.count("}")
        if opens - closes >= 3:
            return {"ok": False, "error": f"Bracket imbalance: {opens - closes} unclosed (truncated?)"}

    return {"ok": True, "error": ""}


def _salvage_code(raw_args: str) -> str:
    if not raw_args:
        return ""
    m = re.search(r'"code"\s*:\s*"', raw_args)
    if not m:
        return ""
    code_raw = raw_args[m.end():]
    for suffix in ['"}\n', '"}', '"', '']:
        try:
            parsed = json.loads('{"code": "' + code_raw + suffix)
            if parsed.get("code"):
                return parsed["code"]
        except json.JSONDecodeError:
            continue
    code = code_raw.replace("\\n", "\n").replace("\\t", "\t")
    code = code.replace('\\"', '"').replace("\\\\", "\\")
    if code.endswith("\\"):
        code = code[:-1]
    return code if len(code) > 50 else ""
