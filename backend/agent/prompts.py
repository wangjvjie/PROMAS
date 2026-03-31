"""Prompt templates for PROMAS pipeline — language-agnostic, general-purpose project generation.

v5: True two-phase ReAct — Think and Act are SEPARATE LLM calls.
  - THINK: LLM reasons WITHOUT tools → pure text
  - ACT:   LLM executes WITH tools → tool call only, no rambling
"""

# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: PRD
# ═════════════════════════════════════════════════════════════════════════════

WRITE_PRD_PROMPT = """\
You are a project manager. The user will give an instruction for coding. \
Your job is to understand the requirements and come up with the implementation approach.

1. Understand the user's underlying needs and refine them into a clear User Story.
2. Consider security at both macro and deployment levels — include measures that \
   can be implemented programmatically (e.g. CAPTCHA, rate limiting, anti-scraping, \
   input sanitisation, auth guards).
3. List ALL configuration files needed so the project runs with a single command. \
   Include files conditionally based on the chosen stack \
   (e.g. "if using TypeScript", "if using Tailwind").

Respond in EXACTLY this format, no extra text:

## User Story
<Restate the user's goal as a user story. Clarify implicit needs.>

## System-Level Security Design
<Macro-level and deployment-level security measures relevant to this project.>

## Implementation Approach
<Bullet-point breakdown of the technical approach: language, framework, \
key libraries, and major implementation steps.>

## Config Files
<List minimal running config/manifest file the project needs, with a one-line note \
on what it covers. Mark optional ones with "(if using X)".>

[User Message]
{msg}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2a: File Design
# ═════════════════════════════════════════════════════════════════════════════

WRITE_FILE_DESIGN_PROMPT = """\
You are a senior software architect.

Design the COMPLETE file structure for this project so that it is fully runnable \
from a clean checkout.

## Critical Requirements
- Include minimal running config/manifest files needed to install dependencies and start the project. \
  Examples: package.json, tsconfig.json, vite.config.ts, index.html, index.php...
- Include a README.md at the project root with install, run, and usage instructions.
- Include every source file implied by the spec. \
  For TypeScript/JS include barrel index files; \
  for Python include __init__.py; for Go include package-level files; etc.
- Match module/package names EXACTLY as specified.
- Do NOT omit any file just because it seems boilerplate — \

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
      "dependencies": ["other_file.ext"]
    }}
  ]
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b: Batch Architecture Design
# ═════════════════════════════════════════════════════════════════════════════

WRITE_ARCHITECTURE_BATCH_PROMPT = """\
You are a senior software architect. Design the internal API for a batch of files.

For each source file, define exported symbols (classes with members, functions with \
typed signatures). For config/manifest files, specify the key fields and values. \
Do NOT write implementation code.

## Key Conventions
- Barrel / index files: list exactly which names they re-export.
- Function signatures MUST match the specification exactly \
  (name, parameter names, types, return type).
- Use idiomatic types for the target language \
  (TypeScript generics, Go interfaces, Python typing, Rust traits, etc.).
- If the spec says `import X from 'pkg'`, then X must be exported \
  from the correct file.

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
          "members": [{{"name": "x", "type": "string"}}],
          "methods": ["methodName"]
        }}
      ],
      "functions": [
        {{
          "name": "funcOrClass.method",
          "input_parameters": [{{"name": "arg", "type": "string"}}],
          "output_parameters": [{{"name": "result", "type": "boolean"}}],
          "description": "what it does"
        }}
      ],
      "config_fields": {{
        "field": "value (for manifest/config files only)"
      }}
    }}
  ]
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Threat Modeling — Full mode (paper-aligned multi-step)
#
# Paper pipeline:
#   Step 1: Extract E_in (entry interfaces) + IC(e) (invocation chains) from Arch
#   Step 2: Generate SC (Global Security Context) + AM (Attacker Model)
#   Step 3: Per-chain function-level threat analysis → merge into global FT
#   Step 4: Generate k candidates
#   Step 5: LLM-Judge scoring → select best
# ═════════════════════════════════════════════════════════════════════════════

# Step 1: Extract entry interfaces and invocation chains
TM_EXTRACT_CHAINS_PROMPT = """\
You are a senior security architect. Given the project architecture below, \
extract ALL user-facing entry interfaces and their invocation chains.

## Architecture
{arch}

Return ONLY valid JSON:
{{
  "entry_interfaces": [
    {{
      "interface": "POST /api/login",
      "entry_function": "auth.py:handle_login",
      "invocation_chain": [
        "auth.py:handle_login",
        "auth.py:verify_credentials",
        "db.py:query_user",
        "auth.py:create_session"
      ]
    }}
  ]
}}

Rules:
- Include ALL HTTP endpoints, CLI commands, API routes, WebSocket handlers, etc.
- Each chain should list functions in call order: entry → downstream callees
- Use the format "file:function_name" for each function
- Trace calls across files using the dependency and function information
- Include utility/helper functions that are transitively called
"""

# Step 2: Generate Global Security Context + Attacker Model
TM_GENERATE_SC_AM_PROMPT = """\
You are a senior security engineer. Given the project specification, PRD, \
and architecture, produce the Global Security Context and Attacker Model.

## Specification
{msg}

## PRD
{prd}

## Architecture
{arch}

## Entry Interfaces
{entry_interfaces}

Respond in EXACTLY this format:

### Global Security Context (SC)
- **Trust boundaries**: where does trusted ↔ untrusted data cross? \
  (e.g., HTTP request boundary, DB query boundary, file system boundary)
- **Key assets**: what data/resources must be protected? \
  (e.g., user credentials, session tokens, uploaded files, DB records)
- **Privilege relationships**: what roles/permissions exist and how is \
  access control enforced?

### Attacker Model (AM)
- **Attacker profile**: unauthenticated remote user, authenticated malicious user, etc.
- **Capabilities**: can craft arbitrary HTTP requests, inject payloads, \
  manipulate file uploads, etc.
- **Objectives**: data exfiltration, privilege escalation, denial of service, \
  code execution, etc.
"""

# Step 3: Per-chain function-level threat analysis
TM_ANALYZE_CHAIN_PROMPT = """\
You are a senior security engineer. Analyze the following invocation chain \
for security threats at the function level.

## Global Security Context
{sc_am}

## Entry Interface: {interface}
## Invocation Chain
{chain}

## Architecture (relevant functions)
{arch_excerpt}

## Existing Threats (from previously analyzed chains)
{existing_threats}

For EACH function in this chain, identify:

Return ONLY valid JSON:
{{
  "function_threats": [
    {{
      "function": "file:function_name",
      "threats": [
        {{
          "cwe": "CWE-79",
          "name": "Cross-Site Scripting",
          "attack_vector": "Attacker injects script via username field",
          "protection": "HTML-escape all user input before rendering"
        }}
      ]
    }}
  ]
}}

Rules:
- Be SPECIFIC: reference actual parameter names and data flows from the architecture
- Focus on CWE-aligned threats (CWE-79, CWE-89, CWE-22, CWE-352, CWE-78, etc.)
- If a function already has threats from previous chains, ADD new threats only \
  (do not repeat existing ones)
- Skip functions with no plausible security-relevant attack surface
"""

# Step 5: LLM-Judge scoring prompt
TM_JUDGE_PROMPT = """\
You are a security expert evaluating threat model quality. \
Score the following threat model candidate on three dimensions.

## Project Specification
{msg}

## Architecture Summary
{arch_summary}

## Threat Model Candidate
{candidate}

Score each dimension from 1 (poor) to 10 (excellent):

Return ONLY valid JSON:
{{
  "relevance": <1-10>,
  "impact": <1-10>,
  "exploitability": <1-10>,
  "reasoning": "brief justification"
}}

Scoring criteria:
- **Relevance** (α): Are the identified threats actually applicable to this project? \
  Are there false positives or missed real threats?
- **Impact** (β): Do the identified threats cover high-impact vulnerabilities? \
  Would exploitation cause significant damage?
- **Exploitability** (γ): Are the identified attack vectors realistic and feasible? \
  Are the protections specific enough to be implemented?
"""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Threat Modeling — Simple mode (single-pass, default)
# ═════════════════════════════════════════════════════════════════════════════

THREAT_MODELING_SIMPLE_PROMPT = """\
You are a senior security engineer. Given the project architecture below, \
produce a concise THREAT MODEL focusing on the most critical security risks.

## Specification
{msg}

## PRD (Requirement Analysis)
{prd}

## Architecture
{arch}

## Required Output Format

### 1. Global Security Context
- Trust boundaries (where untrusted input enters the system)
- Key assets to protect
- Authentication/authorization model

### 2. Function-level Threats
Note: The potential threats should be as comprehensive as possible, covering all possible threats.
### Function: <path/file:function_name>
- Role in the system:
- Untrusted inputs / external sources:
- Security-relevant operations (e.g., file I/O, DB, network, eval/exec, template rendering):
- Potential threats:
  - T1: <short name> — <1–2 line description>
  - T2: ...
- Recommended protections (high-level):

### 3. Project Integrity
- Module consistency and import chains
- Dependency completeness
- Install → build → start correctness
"""

# Backward compatibility alias
THREAT_MODELING_PROMPT = THREAT_MODELING_SIMPLE_PROMPT


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Two-Phase ReAct — System Prompts
# ═════════════════════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """\
You are DevAgent, a security-aware full-stack software developer.

## Your Goal
Generate a project that:
1. Can be installed and started from a clean checkout \
   (e.g. `npm install && npm run dev`, `go run .`, `pip install -e . && python -m app`)
2. Implements every interface/endpoint/component exactly as specified
3. Includes all config files (package.json, tsconfig, .env.example, README, etc.)
4. Handles edge cases (empty input, null, wrong types) gracefully
5. Implements all security protections from the threat model

## CRITICAL: Cross-File Consistency
- Before calling `finish`, verify that EVERY function/class you import from \
  another file actually EXISTS in that file's architecture design.
- If you are unsure what a dependency file exports, call `read_file` or \
  `read_architecture` to check BEFORE writing code that depends on it.
- Symbol names, parameter names, parameter ORDER, and return/response types \
  must match the architecture EXACTLY — across ALL files.
- Do NOT invent functions that are not in the architecture. \
  Do NOT assume a dependency exports something without checking.

## CRITICAL: Match the specification EXACTLY
- Implement ALL security mitigations specified in the threat model for this file
- Do NOT wrap code in markdown fences — pass raw source to `finish`

## Always-Visible Context
The following PRD, Architecture, and Threat Model are your ground truth.

### User Requirement
{user_message}

### PRD (Requirement Analysis)
{prd}

### Threat Model
{threat_model}
"""


REACT_PICK_SYSTEM_PROMPT = """\
You are DevAgent, a security-aware full-stack software developer.

## Your Job
1. **Pick a file** to implement next (call `pick_next_file`)
   Priority: manifest/config files → shared utilities/types → core modules \
   (bottom-up dependency order) → entry points → barrel/index files → README
2. **Write the file** — gather context if needed, then call `finish` with COMPLETE content

## Rules
- Be EFFICIENT: pick → maybe 1-2 reads → finish
- Symbol names and signatures must EXACTLY match the architecture
- Implement ALL security protections specified in the threat model
- Do NOT wrap code or config in markdown fences
- Config files (JSON, YAML) must be valid and complete
- README must contain working install + run instructions

## CRITICAL: Cross-File Consistency
- Before calling `finish`, verify that EVERY function/class you call from \
  another file actually EXISTS in that file's architecture design below.
- If unsure, call `read_file` to see the actual code of a dependency.
- Do NOT invent functions. Do NOT assume exports without checking.
- When a dependency file is already written (marked ✓), `read_file` it to see \
  exact function signatures before importing from it.

## Always-Visible Context
The following PRD, Architecture, and Threat Model are your ground truth. \
Every file you write MUST be consistent with this specification.

### User Requirement
{user_message}

### PRD (Requirement Analysis)
{prd}

### Full Architecture
{full_architecture}

### Threat Model
{threat_model}
"""


VERIFY_SYSTEM_PROMPT = """\
You are DevAgent Verify, an expert at making projects install and run cleanly.

## How to work
1. Run the install command (e.g. `npm install`, `go mod tidy`, `pip install -e .`) \
   — fix manifest files if it fails
2. Run the build command if applicable (e.g. `npm run build`, `tsc`) \
   — fix config files if it fails
3. Run a smoke-start (e.g. `npm run dev -- --dry-run` or import the entry module) \
   — fix entry points and import chains if it fails
4. Fix errors with `edit_file` (surgical, minimal changes only)
5. Re-run after each fix, then call `verify_done` when clean

## Rules
- Be SURGICAL: only fix real errors. Do NOT refactor working code.
- Focus on making the project INSTALLABLE and STARTABLE.
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
1. Run the install command
2. Run the build command (if applicable)
3. Run a smoke-start or import check
4. Fix errors → re-run → repeat
5. Call `verify_done` when clean

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
If ready, call `finish` with the complete file content. \
Otherwise call `read_file` or another tool."""

# ── Force finish (out of steps) ───────────────────────────────────────────

FORCE_FINISH = """\
You have used all your steps. Call `finish` NOW with the complete content \
for '{file_name}'. No other tool calls. Minimize comments to save tokens."""


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

COMPRESS_ARCH_PROMPT = """\
Compress the following architecture design into a brief index. \
For each file, output ONE line: \
`path/name — brief purpose (exports: key_symbol1, key_symbol2, TypeName)`.

{architecture}

Output the compressed index only, no other text.
"""