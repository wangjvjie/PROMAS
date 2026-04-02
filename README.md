# PROMAS — Security-First Code Generation Agent

PROMAS is a code generation agent that produces secure, production-grade project code from a natural language description. It follows a five-stage pipeline — PRD, Architecture Design, Threat Modeling, Code Generation, and Verification — to ensure that security is embedded into every generated file, not bolted on as an afterthought.

## How It Works

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: PRD                                            │
│   Requirement analysis → user story, security design,   │
│   implementation approach, config file list              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Architecture Design                            │
│   File list → dependency-ordered per-file API design    │
│   → cross-file validation → LLM reconciliation          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Threat Modeling                                │
│   Simple mode: single-pass function-level threats       │
│   Full mode: entry chains → SC/AM → per-chain analysis  │
│              → k candidates → LLM-judge scoring         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Code Generation                                │
│   Agent writes files in dependency order with full      │
│   architecture + threat model as ground truth.          │
│   Persistent conversation — no redundant file reads.    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Adversarial Verification                       │
│   Install → build → test → adversarial probes           │
│   (bad inputs, auth bypass, boundary values)            │
│   Surgical fixes via edit → re-verify until clean       │
└─────────────────────────────────────────────────────────┘
```

## Architecture Design

The architecture stage avoids the "one giant JSON" failure mode by designing files individually in dependency order:

1. **File list generation** — LLM produces the complete file structure with dependencies
2. **Topological sort** — Files grouped into dependency layers (layer 0 = no deps, layer 1 = depends on layer 0, etc.)
3. **Per-file API design with layer-level concurrency** — Files in the same layer are designed concurrently via `asyncio.gather`. Each file sees the full API signatures of all previously designed files, not a compressed summary
4. **Cross-file validation** — Eight programmatic checks run without LLM cost:
   - Dependency targets exist
   - Source files have exports (config/views/templates exempted)
   - Type references match across files (if A uses `User`, and `User` is in B, A must depend on B)
   - Naming style near-miss detection (`getUser` vs `get_user`)
   - Circular dependency detection
   - Orphan file detection (multi-language aware)
   - Duplicate symbol detection
   - Function signature consistency (param count mismatch between caller and callee)
5. **LLM reconciliation** — If validation finds issues, one LLM call patches the affected files

This produces small, reliable JSON per file (~200-500 tokens) instead of one massive batch that truncates.

## Threat Modeling

Two modes, selectable per request:

### Simple Mode (default)
Single-pass LLM call producing:
- Global security context (trust boundaries, key assets, auth model)
- Function-level threats with CWE IDs, attack vectors, and specific protections
- Project integrity checks (import chains, install/build/start correctness)

### Full Mode (paper-aligned)
Multi-step pipeline based on the threat modeling research methodology:
1. **Extract entry interfaces** — All HTTP endpoints, CLI commands, WebSocket handlers and their invocation chains
2. **Generate SC + AM** — Global Security Context and Attacker Model
3. **Per-chain function-level analysis** — Each invocation chain analyzed for threats, merged into a global function-threat map with iterative deduplication
4. **k candidate generation** — Multiple threat model variants via temperature variation
5. **LLM-judge scoring** — Each candidate scored on relevance, impact, and exploitability. Best candidate selected by weighted score

The resulting threat model is injected into the code generation system prompt, so every file is written with its specific security requirements.

## Code Generation Agent

The code generation agent uses a persistent conversation (one message history for all files) with concurrent tool execution:

- **Persistent conversation** — Writing file B has full context of file A's content. No re-reading.
- **Concurrent read-only tools** — `read_file`, `search_code`, `read_architecture`, `read_threats`, `detect_env`, `web_fetch`, `lint_check` run in parallel via `asyncio.gather`
- **Write tools run serially** — `pick_next_file`, `finish`, `edit_file` execute one at a time
- **Context compaction** — When approaching the token limit, old messages are compressed while preserving file reads and write results
- **Environment detection** — Auto-detects installed runtimes (node, python, go, php, etc.) and injects versions into the system prompt
- **Code validation** — Python AST parsing, JSON validation, bracket balance checking before accepting generated code

## Verification

The verify stage auto-detects the project's tech stack and injects language-specific commands:

| Stack | Install | Build | Lint | Test |
|-------|---------|-------|------|------|
| Node/React/Next.js | `npm install` | `npm run build` | `eslint`, `tsc --noEmit` | `npm test` |
| Python | `pip install -r requirements.txt` | `python -m compileall .` | `py_compile` | `pytest` |
| PHP/Laravel | `composer install` | `php artisan config:cache` | `php -l` | `phpunit` |
| Go | `go mod tidy` | `go build ./...` | `go vet` | `go test ./...` |
| Rust | — | `cargo build` | `cargo clippy` | `cargo test` |
| Docker | — | `docker build .` | — | — |

After the happy path passes, the agent runs adversarial probes: bad inputs, auth bypass, malformed data, boundary values. Each check requires an actual command and its output — reading code alone doesn't count.

## Quick Start

```bash
# Install
cd backend
pip install -r requirements.txt

# Configure
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible provider
export OPENAI_MODEL="gpt-4o"

# Run
python -m backend.main
```

Open **http://localhost:8000**.

## Project Structure

```
backend/
├── main.py                        # FastAPI + SSE streaming
├── llm/client.py                  # LLM client (timeout, retry, json_mode)
├── project/
│   ├── state.py                   # File index, dependency graph, persistence
│   └── prompt.py                  # Chainable prompt builder
├── tools/
│   ├── base.py                    # Tool ABC (name, is_read_only, execute)
│   ├── file_tools.py              # read_file, search_code, list_files
│   ├── arch_tools.py              # read_architecture, read_threats, pick_next_file
│   ├── write_tools.py             # finish (write file), edit_file
│   ├── system_tools.py            # run_command, verify_done
│   └── env_tools.py               # detect_env, web_fetch, lint_check
├── engine/
│   ├── agent.py                   # Persistent-conversation agent loop
│   └── context_window.py          # Token tracking + message compaction
├── pipeline/
│   ├── orchestrator.py            # 5-stage pipeline
│   └── stages/
│       ├── prd.py
│       ├── architecture.py        # Toposort + per-file design + validation
│       ├── threat_model.py        # Simple + full (paper-aligned) modes
│       ├── code_gen.py            # Persistent agent, env detection
│       └── verify.py              # Stack-aware adversarial verification
├── prompts.py                     # All prompt templates
└── models/__init__.py             # Pydantic schemas
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key for the LLM provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `OPENAI_MODEL` | `gpt-4o` | Model name |
| `MODEL_CONTEXT_LIMIT` | `128000` | Model's context window size (tokens) |
| `LLM_TIMEOUT` | `180` | Timeout per LLM call (seconds) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Start pipeline, returns SSE stream |
| `GET` | `/api/stream?from_index=N` | Reconnect to running SSE stream |
| `GET` | `/api/state` | Full session state (for page reload) |
| `GET` | `/api/status` | Quick status check |
| `GET` | `/api/files` | List generated files |
| `GET` | `/api/files/{path}` | Get file content |
| `POST` | `/api/stop` | Cancel running pipeline |

## License

MIT
