from __future__ import annotations

import ast
import json
import os
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*?\})", re.DOTALL)
CODE_FENCE_BLOCK_RE = re.compile(r"```[^\n`]*\n([\s\S]*?)```", re.MULTILINE)


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model response; cannot extract JSON object")

    raw = text.strip()

    # Fast path: whole response is valid JSON object.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Common case: response wrapped in ```json ... ```
    for blk in JSON_BLOCK_RE.findall(raw):
        blk = blk.strip()
        if not blk:
            continue
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Robust scan: decode from each '{' occurrence, ignoring trailing noise.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # Fallback: try minimal regex candidates.
    for m in JSON_OBJECT_RE.finditer(raw):
        candidate = m.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                lit = ast.literal_eval(candidate)
                if isinstance(lit, dict):
                    return lit
            except Exception:
                continue

    # Final fallback: Python-literal dict style in whole response.
    try:
        lit = ast.literal_eval(raw)
        if isinstance(lit, dict):
            return lit
    except Exception:
        pass

    preview = raw[:400].replace("\n", "\\n")
    raise ValueError(f"No valid JSON object found in model response. Preview: {preview}")


def safe_rel_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    cleaned = cleaned.lstrip("/")
    cleaned = cleaned.replace("./", "", 1) if cleaned.startswith("./") else cleaned
    normalized = os.path.normpath(cleaned).replace("\\", "/")
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"Unsafe relative path: {path}")
    return normalized


def parse_file_block(raw: str) -> tuple[str | None, str]:
    """Try to parse legacy <file> block; fallback to raw text as code."""
    blocks = re.findall(r"<file>(.*?)</file>", raw, flags=re.DOTALL)
    if not blocks:
        return None, strip_markdown_code_fence(raw.strip())

    block = blocks[0].strip()
    lines = block.splitlines()
    in_code = False
    file_path = None
    code_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## File"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                file_path = lines[j].strip()
        if stripped.startswith("## Code"):
            in_code = True
            continue
        if in_code:
            code_lines.append(line)

    code = "\n".join(code_lines).lstrip("\n")
    return file_path, strip_markdown_code_fence(code)


def strip_markdown_code_fence(text: str) -> str:
    """
    Remove outer markdown code fences when model outputs fenced code.
    Keep content unchanged if it is not clearly a fenced-code response.
    """
    s = text.strip()
    if not s:
        return s

    full = re.fullmatch(r"```[^\n`]*\n([\s\S]*?)\n?```", s)
    if full:
        return full.group(1)

    blocks = CODE_FENCE_BLOCK_RE.findall(s)
    if not blocks:
        return s

    outside = CODE_FENCE_BLOCK_RE.sub("", s).strip()
    if not outside:
        return "\n\n".join(block.rstrip("\n") for block in blocks).strip("\n")

    if len(blocks) == 1:
        return blocks[0].strip("\n")

    return s


def dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def with_line_numbers(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return "1: "
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{i:>{width}}: {line}" for i, line in enumerate(lines, start=1))


def apply_line_edits(original_text: str, edits: list[dict[str, Any]]) -> tuple[str, int]:
    """
    Apply 1-based inclusive range edits to a text blob.
    Returns (new_text, applied_edit_count).
    """
    lines = original_text.splitlines()
    had_trailing_newline = original_text.endswith("\n")
    total = len(lines)

    normalized: list[tuple[int, int, str]] = []
    for raw in edits:
        start = int(raw.get("start_line"))
        end = int(raw.get("end_line"))
        replacement = str(raw.get("replacement", ""))

        if start < 1:
            raise ValueError(f"Invalid start_line: {start}")
        if end < 0:
            raise ValueError(f"Invalid end_line: {end}")
        if start > total + 1:
            raise ValueError(f"start_line out of range: {start} > {total + 1}")
        if end > total:
            raise ValueError(f"end_line out of range: {end} > {total}")
        if not (start <= end or start == end + 1):
            raise ValueError(f"Invalid range: start_line={start}, end_line={end}")

        normalized.append((start, end, replacement))

    # Validate non-overlap on original line ranges to avoid ambiguous/stale edits.
    intervals = sorted(
        [(s, e) for s, e, _ in normalized if s <= e],
        key=lambda x: (x[0], x[1]),
    )
    prev_start = -1
    prev_end = -1
    for s, e in intervals:
        if prev_end >= s:
            raise ValueError(
                f"Overlapping edit ranges detected: [{prev_start},{prev_end}] and [{s},{e}]"
            )
        prev_start, prev_end = s, e

    # Apply bottom-up so earlier indexes do not shift.
    normalized.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for start, end, replacement in normalized:
        repl_lines = replacement.splitlines()
        s_idx = start - 1
        e_idx = end  # inclusive -> exclusive
        lines[s_idx:e_idx] = repl_lines

    out = "\n".join(lines)
    if had_trailing_newline and out and not out.endswith("\n"):
        out += "\n"
    return out, len(normalized)
