"""FastAPI application.

Endpoints:
  POST /api/generate   → Start pipeline, returns SSE stream
  GET  /api/stream     → Reconnect to running SSE stream
  GET  /api/state      → Full session state (for page reload restore)
  GET  /api/status     → Quick status check
  GET  /api/files      → List generated files
  GET  /api/files/{p}  → Get file content
  POST /api/stop       → Cancel running pipeline
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .models import EventType, FileContent, GenerateRequest, ProjectStatus, SSEvent
from .pipeline import Pipeline

app = FastAPI(title="PROMAS", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global session state ──────────────────────────────────────────────────────

_pipeline: Pipeline | None = None
_is_running = False
_cancel = asyncio.Event()
_event_log: list[dict] = []
_session_prompt: str = ""
_completed_stages: set[str] = set()
_current_stage: str = ""
_pipeline_task: asyncio.Task | None = None

_notify: asyncio.Condition | None = None
_event_version = 0

DEFAULT_WORK_DIR = "./workspace"


async def _get_notify() -> asyncio.Condition:
    global _notify
    if _notify is None:
        _notify = asyncio.Condition()
    return _notify


async def _push(event: SSEvent, work_dir: str = DEFAULT_WORK_DIR):
    global _event_version, _current_stage, _completed_stages, _event_log

    _event_log.append(event.model_dump())

    if event.type == EventType.STAGE_START:
        _current_stage = event.stage
    elif event.type == EventType.STAGE_END:
        _completed_stages.add(event.stage)
        _current_stage = ""
    elif event.type == EventType.DONE:
        _current_stage = ""

    if (
        len(_event_log) % 5 == 0
        or event.type in (EventType.STAGE_END, EventType.FILE_WRITTEN, EventType.DONE)
    ):
        _save_session(work_dir)

    cond = await _get_notify()
    _event_version += 1
    async with cond:
        cond.notify_all()


# ── Session persistence ───────────────────────────────────────────────────────

def _session_path(work_dir: str) -> str:
    return os.path.join(work_dir, "session.json")


def _save_session(work_dir: str):
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


def _load_session(work_dir: str) -> dict | None:
    path = _session_path(work_dir)
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


# ── Pipeline runner ───────────────────────────────────────────────────────────

async def _run_pipeline(pipeline: Pipeline, work_dir: str):
    global _is_running, _current_stage
    try:
        async for event in pipeline.run():
            if _cancel.is_set():
                await _push(SSEvent(type=EventType.ERROR, content="Cancelled by user"), work_dir)
                break
            await _push(event, work_dir)
    except Exception as e:
        await _push(SSEvent(type=EventType.ERROR, content=str(e)), work_dir)
    finally:
        _is_running = False
        _current_stage = ""
        _save_session(work_dir)


async def _sse_stream(start_index: int = 0):
    idx = start_index
    cond = await _get_notify()
    while True:
        while idx < len(_event_log):
            yield f"data: {json.dumps(_event_log[idx], ensure_ascii=False)}\n\n"
            idx += 1
        if not _is_running and idx >= len(_event_log):
            break
        try:
            async with cond:
                await asyncio.wait_for(cond.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"


# ── Frontend ──────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    global _pipeline, _is_running, _cancel, _pipeline_task
    global _event_log, _session_prompt, _completed_stages, _current_stage

    if _is_running:
        raise HTTPException(409, "A generation is already in progress")

    _cancel.clear()
    _is_running = True
    _event_log = []
    _session_prompt = request.prompt
    _completed_stages = set()
    _current_stage = ""

    _pipeline = Pipeline(request)
    _save_session(request.work_dir)
    _pipeline_task = asyncio.create_task(_run_pipeline(_pipeline, request.work_dir))

    return StreamingResponse(
        _sse_stream(0),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stream")
async def stream_events(from_index: int = Query(0, ge=0)):
    if not _is_running and not _event_log:
        raise HTTPException(404, "No active session")
    return StreamingResponse(
        _sse_stream(from_index),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stop")
async def stop():
    if _is_running:
        _cancel.set()
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.get("/api/state")
async def get_state():
    if _event_log or _pipeline:
        files = _pipeline.state.get_written() if _pipeline else {}
        return {
            "has_session": True,
            "prompt": _session_prompt,
            "events": _event_log,
            "files": dict(files),
            "completed_stages": sorted(_completed_stages),
            "current_stage": _current_stage if _is_running else "",
            "is_running": _is_running,
            "event_count": len(_event_log),
        }

    saved = _load_session(DEFAULT_WORK_DIR)
    if saved:
        _try_restore(DEFAULT_WORK_DIR, saved.get("prompt", ""))
        files = _pipeline.state.get_written() if _pipeline else {}
        events = saved.get("events", [])
        return {
            "has_session": True,
            "prompt": saved.get("prompt", ""),
            "events": events,
            "files": dict(files),
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


def _try_restore(work_dir: str, prompt: str):
    global _pipeline, _session_prompt, _event_log, _completed_stages
    if _pipeline:
        return
    try:
        from .models import GenerateRequest as GR
        cfg = GR(prompt=prompt or "restored", work_dir=work_dir)
        p = Pipeline(cfg)
        p.state.load()
        if p.state.architecture.files or p.state.prd:
            _pipeline = p
            _session_prompt = prompt
            saved = _load_session(work_dir)
            if saved:
                _event_log = saved.get("events", [])
                _completed_stages = set(saved.get("completed_stages", []))
    except Exception:
        pass


@app.get("/api/status")
async def status():
    if not _pipeline:
        return ProjectStatus()
    s = _pipeline.state
    return ProjectStatus(
        stage=_current_stage,
        files_written=list(s.get_written()),
        total_files=len(s.architecture.files),
        is_running=_is_running,
    )


@app.get("/api/files")
async def list_files():
    if not _pipeline:
        return {"files": []}
    return {
        "files": [
            {"path": p, "language": _lang(p), "size": len(c)}
            for p, c in _pipeline.state.get_written().items()
        ]
    }


@app.get("/api/files/{path:path}")
async def get_file(path: str):
    if not _pipeline:
        raise HTTPException(404, "No project loaded")
    written = _pipeline.state.get_written()
    if path not in written:
        raise HTTPException(404, f"File not found: {path}")
    return FileContent(path=path, content=written[path], language=_lang(path))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lang(path: str) -> str:
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


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\nPROMAS running at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
