"""System tools: shell commands and verify-done signal."""

from __future__ import annotations

import subprocess

from .base import Tool
from ..project.state import ProjectState


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Execute a shell command in the project workspace directory.\n\n"
        "Use this for:\n"
        "- Installing dependencies: `npm install`, `pip install -e .`, `go mod tidy`\n"
        "- Building: `npm run build`, `tsc --noEmit`, `go build ./...`, "
        "`python -m compileall .`\n"
        "- Running tests: `pytest`, `npm test`, `go test ./...`\n"
        "- Linting: `eslint src/`, `mypy .`, `golangci-lint run`\n"
        "- Adversarial probes: curl requests, python -c scripts, etc.\n\n"
        "Tips:\n"
        "- Chain dependent commands with `&&`: `npm install && npm run build`\n"
        "- For independent checks, issue multiple run_command calls in one response "
        "(they run in parallel)\n"
        "- Timeout: 60 seconds. For long builds, split into steps.\n"
        "- Always check exit code — non-zero means failure even with no stderr"
    )
    is_read_only = False  # may install packages, create lock files

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run in the workspace directory",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str = "", **_) -> str:
        if not command or not command.strip():
            return "[Error] command is required"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.state.work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            parts = []
            if result.stdout:
                parts.append(f"[stdout]\n{result.stdout.rstrip()}")
            if result.stderr:
                parts.append(f"[stderr]\n{result.stderr.rstrip()}")
            if not parts:
                parts.append("(no output)")
            return f"exit code: {result.returncode}\n" + "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "[Error] Command timed out after 60 seconds."
        except Exception as e:
            return f"[Error] {e}"


class VerifyDoneTool(Tool):
    name = "verify_done"
    description = (
        "Signal that verification is complete.\n\n"
        "Only call this after ALL of the following:\n"
        "1. Build/compile exits with code 0\n"
        "2. Tests pass (or there are no tests)\n"
        "3. At least two adversarial probes have been run\n"
        "4. All fixable errors have been fixed\n\n"
        "Do NOT call this just because the build passes. "
        "You must have tried to break the project first."
    )
    is_read_only = True

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What was checked, what was fixed, adversarial probes run and results, "
                        "final status"
                    ),
                },
            },
            "required": ["summary"],
        }

    async def execute(self, summary: str = "", **_) -> str:
        return f"✅ verify_done: {summary}"
