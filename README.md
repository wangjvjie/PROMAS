# 🛡️ PROMAS Agent — ReAct Coding Agent with Security-First Design

A ReAct-based coding agent that generates **secure, production-grade project code** with an interactive web UI. The agent autonomously decides what context to read, which files to reference, and how to maintain cross-file consistency — even for large projects.

## ✨ Key Features

- **TAF Paradigm** — The agent thinks → acts → observes in a loop, choosing its own tools (read files, search code, inspect architecture) before writing each file
- **Smart Context Retrieval** — Instead of dumping the entire codebase into the prompt, the agent selectively reads only the files it needs via dependency-aware retrieval
- **Compressed Architecture Index** — Full API designs are stored per-file; the agent sees a lightweight index (~2K tokens) and drills into details on demand
- **Security-First Pipeline** — PRD → Architecture → Threat Model → Code, with threat analysis woven into every generated file
- **Streaming Web UI** — Watch the agent think, plan, and write code in real-time via Server-Sent Events
- **Resumable Pipeline** — Crash mid-generation? Resume from any stage

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Web Frontend                    │
│  (TAF SPA — Chat + File Tree + Code Viewer)   │
└────────────────────┬────────────────────────────┘
                     │ SSE / REST
┌────────────────────▼────────────────────────────┐
│               FastAPI Backend                    │
│  ┌───────────────────────────────────────────┐  │
│  │            ReAct Agent Core                │  │
│  │  ┌─────┐  ┌──────┐  ┌─────────────────┐  │  │
│  │  │Think│→ │ Act  │→ │    Observe      │  │  │
│  │  └─────┘  └──┬───┘  └─────────────────┘  │  │
│  │              │                             │  │
│  │     ┌────────▼────────────┐               │  │
│  │     │    Tool Router      │               │  │
│  │     ├────────────────────-┤               │  │
│  │     │ read_file           │               │  │
│  │     │ list_files          │               │  │
│  │     │ read_architecture   │               │  │
│  │     │ search_code         │               │  │
│  │     │ write_file          │               │  │
│  │     │ run_command         │               │  │
│  │     └─────────────────────┘               │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │        Context Manager                    │  │
│  │  • File dependency graph                  │  │
│  │  • Per-file architecture summaries        │  │
│  │  • Compressed index (~2K tokens)          │  │
│  │  • Token budget allocation                │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
# Or for other providers:
export OPENAI_BASE_URL="https://your-provider/v1"
export OPENAI_MODEL="gpt-4o"
```

### 3. Run

```bash
# Start backend + frontend together
python -m backend.main

# Or use the convenience script
chmod +x run.sh && ./run.sh
```

Then open **http://localhost:8000** in your browser.

## 🧠 How the TAF Loop Works

Traditional approach (your old code):
```
prompt = system_design(ALL files) + threat_model + ALL written code + "write file X"
→ Context explodes for large projects
```

ReAct approach (this project):
```
Agent receives: compressed_index + "write file X"
→ Agent THINKS: "File X imports from auth.py and db.py, let me read those"
→ Agent ACTS:   read_file("auth.py"), read_architecture("db.py")  
→ Agent OBSERVES: [file contents]
→ Agent THINKS: "Now I have enough context, and the threat model mentions SQL injection for db.py"
→ Agent ACTS:   read_threat("db.py")
→ Agent OBSERVES: [threat details]
→ Agent ACTS:   write_file("X.py", code)
```

This keeps context under control while maintaining cross-file consistency.

## 📁 Project Structure

```
promas/
├── backend/
│   ├── main.py                 # FastAPI app + SSE streaming
│   ├── agent/
│   │   ├── core.py             # ReAct agent loop
│   │   ├── tools.py            # Agent tools (read/write/search)
│   │   ├── context.py          # Smart context manager
│   │   ├── prompts.py          # All prompt templates  
│   │   └── pipeline.py         # Stage orchestrator
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── requirements.txt
├── frontend/
│   └── index.html              # React SPA (single file, no build step)
├── README.md
└── run.sh
```

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `OPENAI_MODEL` | `gpt-4o` | Model to use |
| `WORK_DIR` | `./workspace` | Output directory |
| `MAX_REACT_STEPS` | `15` | Max ReAct iterations per file |
| `CONTEXT_TOKEN_BUDGET` | `6000` | Token budget for context retrieval |

## 📄 License

MIT
