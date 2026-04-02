from .base import Tool
from .file_tools import ReadFileTool, SearchCodeTool, ListFilesTool
from .arch_tools import ReadArchitectureTool, ReadThreatsTool, PickNextFileTool
from .write_tools import WriteFileTool, EditFileTool
from .system_tools import RunCommandTool, VerifyDoneTool
from .env_tools import RuntimeInfoTool, WebFetchTool, LintCheckTool


def make_code_tools(state, write_tool: WriteFileTool) -> list[Tool]:
    """Tools available during code generation."""
    return [
        ReadFileTool(state),
        ReadArchitectureTool(state),
        ReadThreatsTool(state),
        SearchCodeTool(state),
        ListFilesTool(state),
        PickNextFileTool(state, write_tool),
        RuntimeInfoTool(),         # detect installed runtimes
        WebFetchTool(),            # fetch docs/API references
        LintCheckTool(state),      # quick syntax check after writing
        write_tool,
    ]


def make_verify_tools(state) -> list[Tool]:
    """Tools available during verification."""
    return [
        RunCommandTool(state),
        EditFileTool(state),
        ReadFileTool(state),
        SearchCodeTool(state),
        RuntimeInfoTool(),         # check what's available for install
        WebFetchTool(),            # lookup fix for build errors
        VerifyDoneTool(),
    ]
