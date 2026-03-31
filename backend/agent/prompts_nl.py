"""Prompt templates for PROMAS pipeline — NL2RepoBench adaptation.

v5: True two-phase ReAct — Think and Act are SEPARATE LLM calls.
  - THINK: LLM reasons WITHOUT tools → pure text
  - ACT:   LLM executes WITH tools → tool call only, no rambling
"""

# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: PRD
# ═════════════════════════════════════════════════════════════════════════════

WRITE_PRD_PROMPT = """\
You are a senior Python library architect.

Given the specification below, produce a structured PRD (Product Requirements Document) \
for building this Python library FROM SCRATCH. The library must be pip-installable and \
pass its upstream pytest suite.

## Format

```
## Core Functionality
<What this library does. Summarize the key capabilities in 3-5 bullet points.>

## Public API Surface
<List every public class, function, and constant that users import. \
Group by module. This is the most critical section — any missing API will cause test failures.>

## Package Structure
<Top-level package name, submodules, __init__.py re-exports. \
Specify exactly what `from package import X` should expose.>

## Dependencies & Build Config
<All third-party dependencies with version pins. \
pyproject.toml [build-system] and [project] fields. \
Entry points / console_scripts if any.>

## Implementation Notes
<Key algorithms, data structures, or tricky logic the developer must get right. \
Edge cases that tests are likely to exercise.>
```

Output ONLY the above sections. Be thorough on the Public API Surface — \
every missing function or class will cause a test failure.

## Specification
{msg}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2a: File Design
# ═════════════════════════════════════════════════════════════════════════════

WRITE_FILE_DESIGN_PROMPT = """\
You are a senior Python library architect.

Design the COMPLETE file structure for this Python library. The result must be \
pip-installable (`pip install -e .`) and pass pytest.

## Critical Requirements
- Include `pyproject.toml` (or `setup.py` + `setup.cfg`) at the project root
- Include `__init__.py` for every package/subpackage — these MUST re-export \
  the public API so that `from package import func` works
- Match the module/package names EXACTLY as specified in the spec
- Do NOT include test files — tests are provided externally
- Do NOT include README.md, LICENSE, or docs — focus on source code only

## Specification
{msg}

## PRD
{prd}

Return ONLY valid JSON:
{{
  "files": [
    {{
      "name": "filename.ext",
      "path": "./ or ./subdir",
      "description": "brief purpose of this file",
      "dependencies": ["other_file.ext"]
    }}
  ]
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b: Batch Architecture Design
# ═════════════════════════════════════════════════════════════════════════════

WRITE_ARCHITECTURE_BATCH_PROMPT = """\
You are a senior Python library architect. Design the internal API for a batch of files.

For each file, define classes (with members) and functions (with typed parameters). \
Do NOT write implementation code.

## Key Conventions
- `__init__.py` files: list exactly which names they import and re-export
- Function signatures MUST match the specification exactly (name, params, return type)
- Use standard Python typing (Optional, Union, List, Dict, etc.)
- If the spec says `from package import X`, then X must be importable

## Specification
{msg}

## PRD
{prd}

## Full File List
{files}

## Already Designed Files (summary)
{designed_summary}

## Files to Design Now
{batch_files}

Return ONLY valid JSON:
{{
  "files": [
    {{
      "name": "filename.ext",
      "path": "./subdir",
      "description": "purpose",
      "dependencies": ["imported_file.ext"],
      "classes": [
        {{
          "class_name": "Name",
          "members": [{{"name": "x", "type": "str"}}],
          "methods": ["method_name"]
        }}
      ],
      "functions": [
        {{
          "name": "func_or_Class.method",
          "input_parameters": [{{"name": "arg", "type": "str"}}],
          "output_parameters": [{{"name": "result", "type": "bool"}}],
          "description": "what it does"
        }}
      ]
    }}
  ]
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Robustness Modeling
# ═════════════════════════════════════════════════════════════════════════════

THREAT_MODELING_PROMPT = """\
You are a senior Python quality engineer. Given the library architecture below, \
produce a concise ROBUSTNESS MODEL that identifies how each function could fail \
under testing. Focus on correctness, not security.

## Specification
{msg}

## Architecture
{arch}

## Format

### 1. Package Integrity
- Are all public APIs properly re-exported in __init__.py?
- Are module names consistent with what tests will `import`?
- Are all dependencies declared in pyproject.toml?

### 2. Function-level Robustness
For EACH public function/class:
#### Function: <module:function_name>
- Expected behavior and return type
- Edge cases: empty input, None, wrong type, boundary values, large input
- Potential failure modes: TypeError, ValueError, KeyError, AttributeError
- Defensive measures: type checks, default values, try/except, input validation

### 3. Cross-module Consistency
- Import chains: will `from A import B` work after all files are written?
- Shared data structures: are types/schemas consistent across modules?
- Circular import risks and how to avoid them

Focus on issues that would cause **pytest failures**. Skip irrelevant security concerns.
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Two-Phase ReAct — System Prompts
# ═════════════════════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """\
You are LibDev Agent, an expert Python library developer.

## Your Goal
Generate a Python library that:
1. Is pip-installable (`pip install -e .`)
2. Passes the upstream pytest test suite
3. Has correct `__init__.py` re-exports
4. Implements every function/class exactly as specified

## CRITICAL: Match the specification EXACTLY
- Function names, parameter names, return types must match precisely
- Handle edge cases (empty input, None, wrong types) gracefully
- Do NOT wrap code in markdown fences — pass raw source to `finish`
"""


REACT_PICK_SYSTEM_PROMPT = """\
You are LibDev Agent, an expert Python library developer.

## Your Job
1. **Pick a file** to implement next (call `pick_next_file`)
   Priority: pyproject.toml → utilities/constants → core modules (bottom-up) \
→ __init__.py → entry points
2. **Write the file** — gather context if needed, then call `finish` with COMPLETE code

## Rules
- Be EFFICIENT: pick → maybe 1-2 reads → finish
- Function signatures must EXACTLY match the specification
- Do NOT wrap code in markdown fences
"""


VERIFY_SYSTEM_PROMPT = """\
You are LibDev Verify Agent, an expert at making Python packages installable.

## How to work
1. Run `pip install -e .` — fix pyproject.toml / setup.py if it fails
2. Run `python -c "import <package_name>"` — fix import chains if it fails
3. Fix errors with `edit_file` (surgical, minimal changes only)
4. Re-run after each fix, then call `verify_done` when clean

## Rules
- Be SURGICAL: only fix real errors. Do NOT refactor working code.
- Focus on making the package IMPORTABLE.
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Two-Phase ReAct — Initial User Context
# ═════════════════════════════════════════════════════════════════════════════

REACT_USER_PROMPT = """\
## Your Task
Write the file: **{file_name}**

## Project Architecture (focused on this file and its dependencies)
{full_architecture}

## File Design for {file_name} (detailed)
{file_design}

You will now enter a Think → Act loop. Aim to finish within 3-5 steps.
"""

REACT_PICK_USER_PROMPT = """\
## Project File Index
{full_architecture}

## Project Status
{file_index}

## Already Written
{written_summary}

## Dependency Graph
{dep_summary}

## Remaining Files
{remaining_list}

You will now enter a Think → Act loop. First pick a file, then write it.
"""

VERIFY_USER_PROMPT = """\
## Project File Index
{file_index}

## Your Task
1. Run `pip install -e .`
2. Run `python -c "import <package_name>"`
3. Fix errors → re-run → repeat
4. Call `verify_done` when clean

You will now enter a Think → Act loop.
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Two-Phase ReAct — Per-Step Instructions
# ═════════════════════════════════════════════════════════════════════════════

# ── THINK phase: appended as user message BEFORE the no-tools LLM call ────

THINK_FIRST = """\
**THINK** — Plan your first action.

What context do you have? What do you still need? What should you do first?
Be concise (2-5 sentences). No tool calls — just reasoning."""

THINK_AFTER_OBSERVE = """\
**THINK** — Observation from last action:

{observation}

Analyze this result. What should you do next? \
If you have enough context, plan to call `finish`.
Be concise (2-5 sentences). No tool calls — just reasoning."""

# ── ACT phase: appended as user message BEFORE the tools LLM call ─────────

ACT_INSTRUCTION = """\
**ACT** — Call exactly ONE tool now. No text, only the tool call."""

ACT_RETRY = """\
**ACT (retry)** — You MUST call a tool. No text output allowed.
If ready, call `finish` with the complete code. Otherwise call `read_file` or another tool."""

# ── Force finish (out of steps) ───────────────────────────────────────────

FORCE_FINISH = """\
You have used all your steps. Call `finish` NOW with the complete source code \
for '{file_name}'. No other tool calls. Minimize comments to save tokens."""


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

COMPRESS_ARCH_PROMPT = """\
Compress the following architecture design into a brief index. \
For each file, output ONE line: `path/name — brief purpose (exports: key_func1, key_func2, ClassName)`.

{architecture}

Output the compressed index only, no other text.
"""