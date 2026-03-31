"""Tool definitions and router for the ReAct agent.

Tools are defined with OpenAI function-calling JSON schemas so the LLM
uses native tool_use instead of fragile text parsing.  This is how
Claude Code / Codex-style agents work.
"""

from __future__ import annotations

import json
from typing import Any

from .context import ContextManager


# ── OpenAI function-calling tool definitions ─────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full content of an already-written project file. "
                "Use this to see how a dependency is implemented so you can "
                "import from it correctly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. 'src/auth.py'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_architecture",
            "description": (
                "Get the detailed API design (classes, functions, typed parameters) "
                "for a specific file. NOTE: You already have all file signatures in "
                "the architecture summary — use this tool only if you need extra detail "
                "like full parameter descriptions that were truncated in the summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "File name or path to look up",
                    }
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_threats",
            "description": (
                "Get threat model entries relevant to a specific file. "
                "Use this to understand what security protections to apply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "File name or path",
                    }
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search across all already-written code for a symbol, pattern, "
                "or keyword. Returns matching lines with file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (function name, class name, import, etc.)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List all project files with their status: written or planned. "
                "Shows exports and dependencies for each file."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_next_file",
            "description": (
                "Choose the next file to implement. Call this FIRST at the start "
                "of each file-writing cycle. Returns the file's architecture design, "
                "dependency status, and what it will unblock. You should pick based on: "
                "dependencies satisfied > foundation files first > unblocking others > leaf nodes last."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "File path to implement next, e.g. 'src/auth.py'",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for choosing this file",
                    },
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Submit the final, complete code for the current file. "
                "The code must be the FULL file content, ready to run. "
                "Do NOT wrap in markdown fences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete file content (source code only, no markdown)",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ── Verify-phase tool definitions ────────────────────────────────────────

VERIFY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command in the project workspace directory. "
                "Use this to run build commands (e.g. `python -m py_compile file.py`, "
                "`npx tsc --noEmit`, `go build ./...`), linters, tests, or any "
                "verification command. Returns stdout + stderr. "
                "Commands are run with a 60-second timeout. "
                "ALWAYS start by running a build/compile command to find real errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute, e.g. 'python -m py_compile src/main.py'",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit an already-written file by replacing a specific section of text. "
                "Provide the exact old text to find (must match EXACTLY once, including "
                "whitespace and indentation) and the new text to replace it with. "
                "Use this to fix errors found by build/compile/test commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. 'src/auth.py'",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace (must match exactly once)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full content of a project file. Use this to see the code "
                "around an error before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search across all written code for a symbol or pattern. "
                "Useful for finding where a class/function is actually defined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_done",
            "description": (
                "Declare verification complete. Call this ONLY after the build/compile "
                "command runs cleanly with no errors, or after you have fixed all "
                "fixable errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary: what was checked, what was fixed, final status",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


class ToolRouter:
    """Routes agent tool calls to their implementations."""

    def __init__(self, ctx: ContextManager):
        self.ctx = ctx
        self._handlers = {
            "read_file": self._read_file,
            "read_architecture": self._read_architecture,
            "read_threats": self._read_threats,
            "search_code": self._search_code,
            "list_files": self._list_files,
            "pick_next_file": self._pick_next_file,
            "edit_file": self._edit_file,
            "run_command": self._run_command,
        }

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the observation string."""
        handler = self._handlers.get(tool_name)
        if not handler:
            available = ", ".join(self._handlers.keys())
            return f"[Error] Unknown tool: {tool_name}. Available tools: {available}, finish"
        try:
            return handler(arguments)
        except Exception as e:
            return f"[Error] Tool '{tool_name}' failed: {e}"

    def _read_file(self, args: dict) -> str:
        return self.ctx.tool_read_file(args.get("path", ""))

    def _read_architecture(self, args: dict) -> str:
        return self.ctx.tool_read_architecture(args.get("file_name", ""))

    def _read_threats(self, args: dict) -> str:
        return self.ctx.tool_read_threats(args.get("file_name", ""))

    def _search_code(self, args: dict) -> str:
        return self.ctx.tool_search_code(args.get("query", ""))

    def _list_files(self, args: dict) -> str:
        return self.ctx.tool_list_files()

    def _pick_next_file(self, args: dict) -> str:
        return self.ctx.tool_pick_next_file(args.get("file_name", ""))

    def _edit_file(self, args: dict) -> str:
        return self.ctx.tool_edit_file(
            path=args.get("path", ""),
            old_text=args.get("old_text", ""),
            new_text=args.get("new_text", ""),
        )

    def _run_command(self, args: dict) -> str:
        return self.ctx.tool_run_command(args.get("command", ""))

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI function-calling tool definitions."""
        return TOOL_DEFINITIONS