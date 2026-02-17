WRITE_PRD_PROMPT = """
You are a project manager, the user will give an instruction for coding, your job is to understand the requirements and come up with the implementation approach.
You must include practical, deployable security requirements.

Return markdown with exactly:
## User Story
## System-Level Security Design
## Implementation approach

[User Messages]
{msg}
"""


WRITE_FILE_DESIGN_PROMPT = """
You are a senior system architect.
Design the full file structure for a runnable project. Include all required config files.
Do not write code.

## User Message
{msg}

## User Story
{prd}

Return JSON only:
{{
  "files": [
    {{
      "name": "string(only filename eg. main.py)",
      "path": "./ or ./subdir",
      "description": "string"
    }}
  ]
}}
"""


WRITE_SYSTEM_DESIGN_PROMPT = """
You are a senior system architect.
Design classes/functions for one file. Do not write implementation code.

## User Message
{msg}

## User Story
{prd}

## All file structures
{files}

## History System Design
{api}

## The file you are currently designing is
{file_name}

Return JSON only:
{{
  "name": "string(only filename eg. main.py)",
  "path": "./ or ./subdir",
  "classes": [
    {{
      "class_name": "string",
      "members": [{{"name": "string", "type": "string"}}]
    }}
  ],
  "functions": [
    {{
      "name": "class.func or func",
      "input_parameters": [{{"name": "string", "type": "string"}}],
      "output_parameters": [{{"name": "string", "type": "string"}}]
    }}
  ]
}}
"""


THREAT_MODELING_PROMPT = """
Given the architecture below, produce a concrete threat model only.
Do not output code.

## 1. Global Security Context
- Overall purpose
- Main trust boundaries
- Key assets and privileges

## 2. Function-level Threats
For EACH function, include:
- Role
- Untrusted inputs
- Security-relevant operations
- Potential threats
- Recommended protections

## 3. Attacker Model
- Capabilities
- Goals

## User Message
{msg}

## System Design
{api}
"""


GET_CALL_CHAIN_PROMPT = """
Extract user-entry interfaces and their call chains from system design.
Return JSON only:
{{
  "interfaces": [
    {{
      "funcname": "file.py:func",
      "call-chain": ["file.py:callee"]
    }}
  ]
}}

## System Design
{api}
"""


THREAT_MODELING_PER_CHAIN_PROMPT = """
Given the architecture and one call chain, produce function-level threat analysis for each function in the chain.
Do not output code.

## User Message
{msg}

## User Story
{prd}

## Architecture Design
{api}

## Target Call Chain
{interface}
"""


COMBINE_THREAT_MODELS_PROMPT = """
Combine threat model fragments into one final threat model with sections:
1. Global Security Context
2. Function-level Threats
3. Attacker Model

## User Message
{msg}

## System Design
{api}

## Threat Model Fragments
{threat_models}
"""


THREAT_MODEL_TO_JSON_PROMPT = """
Convert the following threat model text into a strict JSON structure for retrieval.
Return JSON only.

Input Threat Model:
{threat_model}

Output JSON schema:
{{
  "global_context": {{
    "overall_purpose": "string",
    "trust_boundaries": ["string"],
    "assets_and_privileges": ["string"]
  }},
  "functions": [
    {{
      "function": "path/file.py:func_or_class.func",
      "file": "path/file.py",
      "role": "string",
      "untrusted_inputs": ["string"],
      "security_operations": ["string"],
      "threats": ["string"],
      "protections": ["string"]
    }}
  ],
  "attacker_model": {{
    "capabilities": ["string"],
    "goals": ["string"]
  }}
}}
"""


TASK_PLANNING_PROMPT = """
You are a principal engineer planning parallel implementation work.
Given architecture + existing workspace files, split code work into tasks with explicit dependencies.
Each task maps to exactly one file.
Decide whether each task should create or update existing code.

Return JSON only:
{{
  "tasks": [
    {{
      "task_id": "T1",
      "file": "path/file.py",
      "goal": "what to build in this file",
      "depends_on": ["T0"],
      "priority": 1,
      "mode": "create_only|update_only|update_or_create",
      "source": "arch|existing|hybrid"
    }}
  ]
}}

## User Message
{msg}

## System Design
{api}

## Threat Model
{threat}

## Existing Workspace Manifest
{workspace_manifest}

Rules:
- Cover all architecture files.
- If a file already exists and should be modified, use mode=update_only or update_or_create.
- Keep dependency graph acyclic.
- 你需要合理的将任务进行排序，因为不同文件之间可能存在依赖关系，必须先完成被依赖文件的任务才能开始依赖文件的任务。因此，你需要根据系统设计中的文件依赖关系来规划任务的优先级和依赖项。
"""


WORKER_REFLECTION_PROMPT = """
You are worker {worker_id}. Your current task:
- task_id: {task_id}
- file: {file_path}
- goal: {goal}
- dependencies: {dependencies}
- completed files: {completed_files}

Architecture
{api}

Relevant memory snippets:
{memory_snippets}

Known file context snippets:
{file_snippets}

当前已经完成的文件列表: 
{code_tree}

Choose one next action and return JSON only:
{{
  "action": "read_file|write_file|wait|finish",
  "args": {{
    "query": "optional",
    "file": "required if action is write_file, e.g. path/file.py",
    "command": "optional, e.g. rg --files src",
    "reason": "optional"
  }}
}}

Rules:
- 当你写代码之前，你需要先阅读相关文件来获取上下文信息，确保你的代码与现有代码库保持一致。如果你缺乏必要的上下文信息，选择read_file并说明你需要阅读哪个文件,注意你不能读当前任务的目标文件，因为这是你要写的文件。
- If dependency artifacts are missing, choose wait.
- If enough context exists and file is small/new, choose write_file.
- After write/edit, decide whether to continue editing or finish.
- Do not output anything except JSON.
"""


FILE_IMPLEMENT_PROMPT = """
You are an experienced engineer writing one secure file.
You must satisfy system design and threat model.
Output code only, no markdown.

## User Message
{msg}

## System Design
{api}

## Threat Model
{threat}

## Target Task
- task_id: {task_id}
- file: {file_path}
- goal: {goal}

## Direct Required Context (high priority, must follow)
{direct_context}

## Dependency code context
{dependency_context}


Hard constraints:
- Do not import/reference project-local classes that are absent from System Design (`files` / `all_files_index`) and dependency context.
- Do not invent new local packages/types (e.g. `com.xxx.*`) unless you implement them in the current target file.
- If a required local type is unavailable, use a minimal local fallback (small inner DTO, Map, or primitive type) instead.
"""


FILE_EDIT_PROMPT = """
You are an experienced engineer editing an existing file with minimal changes.
Return JSON only. Do NOT return full file content.

## User Message
{msg}

## System Design
{api}

## Threat Model
{threat}

## Target Task
- task_id: {task_id}
- file: {file_path}
- goal: {goal}

## Direct Required Context
{direct_context}

## Dependency code context
{dependency_context}

## Retrieved memory context
{memory_context}

Hard constraints:
- Do not add imports/references to project-local classes that are absent from System Design (`files` / `all_files_index`) and dependency context.
- Keep edits compile-oriented: prefer removing invalid local imports/usages over introducing new missing dependencies.
- If a required local type is unavailable, use a minimal local fallback (small inner DTO, Map, or primitive type).

## Current file content with line numbers
{current_file_numbered}

Return JSON only:
{{
  "edits": [
    {{
      "start_line": 10,
      "end_line": 14,
      "replacement": "new code block"
    }}
  ],
  "summary": "what was changed and why"
}}

Line edit rules:
- 1-based line numbers, end_line is inclusive.
- Replace lines [start_line, end_line] with replacement.
- Insert: use start_line = end_line + 1 (e.g. append with start=N+1,end=N).
- Delete: replacement can be empty string.
- Keep edits minimal and deterministic.
"""


FILE_EDIT_REPAIR_PROMPT = """
Your previous edit response could not be safely applied.
Regenerate a corrected edit plan against the latest file content below.
Return JSON only.

## Previous apply error
{error}

## Previous invalid edit response
{previous_response}

## Current file content with line numbers
{current_file_numbered}

Return JSON only:
{{
  "edits": [
    {{
      "start_line": 10,
      "end_line": 14,
      "replacement": "new code block"
    }}
  ],
  "summary": "corrected edits"
}}

Rules:
- Use the current line numbers only.
- Do not include overlapping ranges.
- Keep edits minimal and deterministic.
"""


FILE_CHECK_PROMPT = """
You are a senior reviewer checking one file after implementation.
Focus on syntax correctness, unresolved local imports, and cross-file consistency.
Return JSON only.

## User Message
{msg}

## System Design (relevant subset)
{api}

## Threat Model (relevant subset)
{threat}

## Target Task
- task_id: {task_id}
- file: {file_path}
- goal: {goal}

## Dependency code context
{dependency_context}

## Current target file with line numbers
{current_file_numbered}

Return JSON only:
{{
  "needs_fix": true,
  "issues": [
    {{
      "severity": "high|medium|low",
      "line": 12,
      "message": "problem detail"
    }}
  ],
  "fix_strategy": "short suggestion"
}}
"""


SYSTEM_DESIGN_CONSISTENCY_PROMPT = """
You are a principal architect doing a final consistency check for a multi-file system design.
Analyze cross-file contracts, naming, dependencies, and missing interfaces.

## User Message
{msg}

## User Story
{prd}

## Current System Design
{api}

Return JSON only:
{{
  "summary": "short summary",
  "consistency_score": 1,
  "issues": [
    {{
      "severity": "high|medium|low",
      "file": "path/file.ext",
      "description": "problem"
    }}
  ],
  "file_updates": [
    {{
      "file": "path/file.ext",
      "replace_design": {{
        "name": "file.ext",
        "path": "./path",
        "classes": [],
        "functions": []
      }},
      "reason": "why this replacement fixes consistency"
    }}
  ]
}}

Rules:
- If no changes are needed, return empty file_updates.
- Keep updates minimal, only for inconsistent files.
"""


WRITE_README_PROMPT = """
Write a concise README.md from this codebase snapshot.
Include: Overview, Requirements, Installation, Run.
Do not wrap entire output in code fences.

## Project code
{code}
"""


GLOBAL_PROJECT_REVIEW_PROMPT = """
You are a principal engineer doing a whole-project review before compilation.
Read the full workspace context and output a minimal, compile-oriented fix plan.
Return JSON only.

## User Message
{msg}

## System Design
{api}

## Threat Model
{threat}

## Workspace Manifest
{workspace_manifest}

## File Snapshots
{file_snapshots}

Return JSON only:
{{
  "summary": "short summary",
  "priority_fixes": [
    {{
      "file": "path/file.ext",
      "mode": "update_only|update_or_create",
      "goal": "what to change for compile/runtime consistency",
      "reason": "why this file is high priority"
    }}
  ]
}}

Rules:
- Focus on compile/runtime correctness over refactor style.
- Keep file list small and high-confidence.
- Use update_or_create when a missing local class/interface is needed.
- Do not suggest external package installation commands.
"""


COMPILE_ERROR_TASKS_PROMPT = """
You are fixing build/compile failures for an already-generated project.
Given compiler output, produce concrete file-level repair tasks.
Return JSON only.

## User Message
{msg}

## Compiler Output
{compile_output}

## Workspace Manifest
{workspace_manifest}

## File Snapshots
{file_snapshots}

Return JSON only:
{{
  "summary": "short diagnosis",
  "tasks": [
    {{
      "file": "path/file.ext",
      "mode": "update_only|update_or_create",
      "goal": "minimal fix for compile error(s)",
      "reason": "brief reason"
    }}
  ]
}}

Rules:
- Prefer minimal edits.
- If a type/package is missing locally, either create it (update_or_create) or remove invalid usage.
- Keep task count focused on compiler-reported root causes.
"""
