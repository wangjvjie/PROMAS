"""FastAPI application — serves the API and the frontend.

Endpoints:
  GET  /                    → Frontend SPA
  POST /api/generate        → Start generation pipeline (returns SSE stream)
  GET  /api/stream          → Reconnect to live SSE event stream (for page reload)
  GET  /api/state           → Get full restorable session state
  GET  /api/status          → Get current project status
  GET  /api/files           → List generated files
  GET  /api/files/{path}    → Get file content
  POST /api/stop            → Stop current generation
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import GenerateRequest, ProjectStatus, FileContent, SSEvent, EventType
from .agent.pipeline import Pipeline

app = FastAPI(title="SecDev Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────────────────────────

_current_pipeline: Pipeline | None = None
_is_running = False
_cancel_event = asyncio.Event()
_event_log: list[dict] = []            # accumulated events for current/last session
_session_prompt: str = ""              # prompt for current/last session
_completed_stages: set[str] = set()    # stages that finished successfully
_current_stage: str = ""
_pipeline_task: asyncio.Task | None = None

# Notify mechanism for SSE subscribers
_event_version = 0
_event_notify: asyncio.Condition | None = None  # lazily created

DEFAULT_WORK_DIR = "./workspace"


async def _get_notify() -> asyncio.Condition:
    """Lazily create the asyncio.Condition (must be called inside an event loop)."""
    global _event_notify
    if _event_notify is None:
        _event_notify = asyncio.Condition()
    return _event_notify


async def _push_event(event: SSEvent, work_dir: str = DEFAULT_WORK_DIR):
    """Append an event to the log and wake all SSE subscribers."""
    global _event_version, _current_stage, _completed_stages, _event_log

    evt_dict = event.model_dump()
    _event_log.append(evt_dict)

    # Track stage progress
    if event.type == EventType.STAGE_START:
        _current_stage = event.stage
    elif event.type == EventType.STAGE_END:
        _completed_stages.add(event.stage)
        _current_stage = ""
    elif event.type == EventType.DONE:
        _current_stage = ""

    # Periodic save
    should_save = (
        len(_event_log) % 5 == 0
        or event.type in (EventType.STAGE_END, EventType.FILE_WRITTEN, EventType.DONE)
    )
    if should_save:
        _save_session(work_dir)

    # Notify all waiting SSE subscribers
    cond = await _get_notify()
    _event_version += 1
    async with cond:
        cond.notify_all()


# ── Session Persistence ──────────────────────────────────────────────────────

def _session_path(work_dir: str = DEFAULT_WORK_DIR) -> str:
    return os.path.join(work_dir, "session.json")


def _save_session(work_dir: str = DEFAULT_WORK_DIR):
    """Persist session state (events, prompt, stages) to disk."""
    os.makedirs(work_dir, exist_ok=True)
    data = {
        "prompt": _session_prompt,
        "events": _event_log,
        "completed_stages": sorted(_completed_stages),
        "current_stage": _current_stage if _is_running else "",
    }
    try:
        with open(_session_path(work_dir), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _load_session(work_dir: str = DEFAULT_WORK_DIR) -> dict | None:
    """Load session state from disk. Returns None if not found."""
    path = _session_path(work_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Pipeline Background Runner ───────────────────────────────────────────────

async def _run_pipeline_bg(pipeline: Pipeline, work_dir: str):
    """Run the pipeline as a background task, fully decoupled from SSE streams.

    If every SSE client disconnects (e.g. browser refresh), the pipeline
    keeps running here. Clients reconnect via GET /api/stream.
    """
    global _is_running, _current_stage
    try:
        async for event in pipeline.run():
            if _cancel_event.is_set():
                cancel_evt = SSEvent(type=EventType.ERROR, content="Cancelled by user")
                await _push_event(cancel_evt, work_dir)
                break
            await _push_event(event, work_dir)

    except Exception as e:
        err_evt = SSEvent(type=EventType.ERROR, content=str(e))
        await _push_event(err_evt, work_dir)
    finally:
        _is_running = False
        _current_stage = ""
        _save_session(work_dir)


async def _sse_subscriber(start_index: int = 0):
    """Generator that yields SSE data from the shared event log.

    Multiple SSE clients can subscribe concurrently. Each tracks its own
    read cursor (idx). The pipeline pushes events via _push_event(); this
    generator picks them up and yields them.
    """
    idx = start_index
    cond = await _get_notify()

    while True:
        # Drain all new events
        while idx < len(_event_log):
            evt_dict = _event_log[idx]
            data = json.dumps(evt_dict, ensure_ascii=False)
            yield f"data: {data}\n\n"
            idx += 1

        # If pipeline is finished and we've sent everything, exit
        if not _is_running and idx >= len(_event_log):
            break

        # Wait for new events or send a keepalive
        try:
            async with cond:
                await asyncio.wait_for(cond.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"


# ── Frontend ─────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Start the generation pipeline in the background. Returns an SSE stream."""
    global _current_pipeline, _is_running, _cancel_event, _pipeline_task
    global _event_log, _session_prompt, _completed_stages, _current_stage

    if _is_running:
        raise HTTPException(409, "A generation is already in progress")

    _cancel_event.clear()
    _is_running = True
    _event_log = []
    _session_prompt = request.prompt
    _completed_stages = set()
    _current_stage = ""

    pipeline = Pipeline(request)
    _current_pipeline = pipeline

    # Save initial session
    _save_session(request.work_dir)

    # Launch pipeline in a background task — survives SSE disconnections
    _pipeline_task = asyncio.create_task(_run_pipeline_bg(pipeline, request.work_dir))

    # Return an SSE stream starting from event 0
    return StreamingResponse(
        _sse_subscriber(start_index=0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stream")
async def stream_events(from_index: int = Query(0, ge=0)):
    """Reconnect to the live SSE event stream.

    After a page refresh, the frontend:
      1. Calls GET /api/state  → restores all past events, files, stages
      2. Connects here with from_index = state.event_count
         → receives ONLY new events going forward, no duplicates

    The pipeline is NOT affected by this reconnection.
    """
    if not _is_running and not _event_log:
        raise HTTPException(404, "No active session")

    return StreamingResponse(
        _sse_subscriber(start_index=from_index),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/stop")
async def stop_generation():
    global _is_running, _cancel_event
    if _is_running:
        _cancel_event.set()
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.get("/api/state")
async def get_state():
    """Return full session state for frontend restoration after page reload."""
    global _current_pipeline, _is_running
    global _event_log, _session_prompt, _completed_stages, _current_stage

    # In-memory session (most up-to-date)
    if _event_log or _current_pipeline:
        files = {}
        if _current_pipeline:
            files = _current_pipeline.ctx.get_written_files()

        return {
            "has_session": True,
            "prompt": _session_prompt,
            "events": _event_log,
            "files": {path: content for path, content in files.items()},
            "completed_stages": sorted(_completed_stages),
            "current_stage": _current_stage if _is_running else "",
            "is_running": _is_running,
            "event_count": len(_event_log),
        }

    # Try loading from disk (server restarted but workspace persists)
    saved = _load_session()
    if saved:
        files = {}
        work_dir = DEFAULT_WORK_DIR
        if os.path.isdir(work_dir):
            _try_restore_pipeline(work_dir, saved.get("prompt", ""))
            if _current_pipeline:
                files = _current_pipeline.ctx.get_written_files()

        events = saved.get("events", [])
        return {
            "has_session": True,
            "prompt": saved.get("prompt", ""),
            "events": events,
            "files": {path: content for path, content in files.items()},
            "completed_stages": saved.get("completed_stages", []),
            "current_stage": "",
            "is_running": False,
            "event_count": len(events),
        }

    return {
        "has_session": False,
        "prompt": "",
        "events": [],
        "files": {},
        "completed_stages": [],
        "current_stage": "",
        "is_running": False,
        "event_count": 0,
    }


def _try_restore_pipeline(work_dir: str, prompt: str):
    """Restore a Pipeline from saved workspace state (for file serving)."""
    global _current_pipeline, _session_prompt, _event_log, _completed_stages

    if _current_pipeline:
        return

    try:
        config = GenerateRequest(prompt=prompt or "restored session", work_dir=work_dir)
        pipeline = Pipeline(config)
        pipeline.ctx.load_state()
        if pipeline.ctx.architecture.files or pipeline.ctx.prd:
            _current_pipeline = pipeline
            _session_prompt = prompt

            saved = _load_session(work_dir)
            if saved:
                _event_log = saved.get("events", [])
                _completed_stages = set(saved.get("completed_stages", []))
    except Exception:
        pass


@app.get("/api/status")
async def get_status():
    global _current_pipeline, _is_running
    if not _current_pipeline:
        return ProjectStatus()

    ctx = _current_pipeline.ctx
    return ProjectStatus(
        stage=_current_pipeline.config.begin_stage.value,
        files_written=list(ctx.get_written_files().keys()),
        total_files=len(ctx.architecture.files),
        is_running=_is_running,
    )


@app.get("/api/files")
async def list_files():
    global _current_pipeline
    if not _current_pipeline:
        return {"files": []}

    ctx = _current_pipeline.ctx
    files = []
    for path, content in ctx.get_written_files().items():
        lang = _detect_language(path)
        files.append({"path": path, "language": lang, "size": len(content)})
    return {"files": files}


@app.get("/api/files/{path:path}")
async def get_file(path: str):
    global _current_pipeline
    if not _current_pipeline:
        raise HTTPException(404, "No project loaded")

    ctx = _current_pipeline.ctx
    written = ctx.get_written_files()
    if path not in written:
        raise HTTPException(404, f"File not found: {path}")

    return FileContent(
        path=path,
        content=written[path],
        language=_detect_language(path),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_sse(event: SSEvent) -> str:
    data = event.model_dump_json()
    return f"data: {data}\n\n"


def _detect_language(path: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html", ".css": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
        ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
        ".php": "php", ".rb": "ruby", ".rs": "rust", ".sh": "bash",
        ".sql": "sql", ".toml": "toml",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return "text"


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n🛡️  SecDev Agent running at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
