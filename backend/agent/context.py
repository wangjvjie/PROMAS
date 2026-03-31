"""Smart context manager for the ReAct agent.

Solves the core problem: instead of dumping the entire architecture + all code into every prompt,
this module maintains a compressed index and lets the agent retrieve context on-demand.

Key ideas:
  1. Compressed file index — ~2K tokens for the full project overview
  2. Per-file architecture stored separately — agent reads on demand (~200-500 tokens each)
  3. Dependency graph — knows which files import which
  4. Token budget — agent's tool calls are bounded to stay within context limits
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from ..models import ArchitectureDesign, FileDesign, ThreatModel


@dataclass
class ContextManager:
    """Manages project context with token-aware retrieval."""

    work_dir: str = "./workspace"
    architecture: ArchitectureDesign = field(default_factory=ArchitectureDesign)
    threat_model: ThreatModel = field(default_factory=ThreatModel)
    prd: str = ""

    # Written file cache: path -> content
    _written_files: dict[str, str] = field(default_factory=dict)

    # Compressed index (rebuilt when architecture changes)
    _file_index: str = ""

    # Dependency graph: file_key -> set of file_keys it depends on
    _dep_graph: dict[str, set[str]] = field(default_factory=dict)

    # ── Index Management ─────────────────────────────────────────────────

    def build_index(self) -> str:
        """Build a compressed file index from the architecture.
        
        Output format (one line per file):
          path/name — description (exports: func1, func2, ClassName)
        
        This typically stays under 2K tokens even for 50+ files.
        """
        lines = []
        for f in self.architecture.files:
            key = self._file_key(f)
            exports = []
            for cls in f.classes:
                exports.append(cls.class_name)
            for fn in f.functions:
                exports.append(fn.name)
            
            status = "✓" if key in self._written_files else "○"
            export_str = ", ".join(exports)
            
            desc = f.description if f.description else "no description"
            deps = f.dependencies
            dep_str = f" [imports: {', '.join(deps)}]" if deps else ""
            
            lines.append(f"  {status} {key} — {desc} (exports: {export_str}){dep_str}")

        self._file_index = "Project Files:\n" + "\n".join(lines)
        return self._file_index

    def get_file_index(self) -> str:
        """Get the compressed file index."""
        if not self._file_index:
            self.build_index()
        return self._file_index

    def get_full_architecture_summary(self) -> str:
        """Build a compact but COMPLETE architecture summary showing every file's
        classes (with members) and functions (with typed signatures).

        This gives the agent a global view of all APIs so it can write code that
        is consistent across files without needing to call read_architecture for
        each dependency.  Typically 4-8K tokens for a 30-50 file project.
        """
        if not self.architecture.files:
            return "(no architecture available)"

        sections = []
        for f in self.architecture.files:
            sections.append(self._format_file_summary(f))

        return "\n\n".join(sections)

    def get_focused_architecture(self, file_name: str, max_total_chars: int = 0) -> str:
        """Build a FOCUSED architecture summary for generating a specific file.

        Shows FULL API details for the target file, its dependencies, AND its
        dependents so the agent can write code that is consistent in both
        directions (what it imports AND what imports it).
        
        No truncation — modern models have enough context (128K+).
        """
        if not self.architecture.files:
            return "(no architecture available)"

        target_key = self._resolve_file_key(file_name)
        deps = set(self.get_dependencies(file_name))
        dependents = set(self.get_dependents(file_name))

        # Categorize files
        target_section = ""
        dep_sections = []
        dependent_sections = []
        other_lines = []

        for f in self.architecture.files:
            key = self._file_key(f)
            if key == target_key:
                target_section = self._format_file_summary(f)
            elif key in deps:
                dep_sections.append(self._format_file_summary(f))
            elif key in dependents:
                # Full detail for dependents too — they need to match our exports
                dependent_sections.append(self._format_file_summary(f))
            else:
                other_lines.append(self._format_file_index_line(f))

        parts = []
        parts.append(f"## Target File\n{target_section}")

        if dep_sections:
            parts.append(f"## Dependencies (full API)\n" + "\n\n".join(dep_sections))

        if dependent_sections:
            parts.append(f"## Dependents (full API — they import from this file)\n" + "\n\n".join(dependent_sections))

        if other_lines:
            parts.append(f"## Other Files (index only)\n" + "\n".join(other_lines))

        return "\n\n".join(parts)

    def _format_file_summary(self, f: FileDesign) -> str:
        """Format a single file's architecture as a complete summary block.
        
        No truncation — every class member, method, and function signature
        is shown in full so the agent can write consistent cross-file code.
        """
        key = self._file_key(f)
        status = "✓" if key in self._written_files else "○"
        lines = [f"### {status} {key}"]
        if f.description:
            lines.append(f"  # {f.description}")
        if f.dependencies:
            lines.append(f"  imports: {', '.join(f.dependencies)}")

        for cls in f.classes:
            members = ", ".join(f"{m.name}: {m.type}" for m in cls.members)
            lines.append(f"  class {cls.class_name}({members})")
            for method in cls.methods:
                lines.append(f"    .{method}()")

        for fn in f.functions:
            params = ", ".join(f"{p.name}: {p.type}" for p in fn.input_parameters)
            returns = ", ".join(f"{p.type}" for p in fn.output_parameters) or "None"
            desc = f"  # {fn.description}" if fn.description else ""
            lines.append(f"  def {fn.name}({params}) -> {returns}{desc}")

        return "\n".join(lines)

    def _format_file_index_line(self, f: FileDesign) -> str:
        """One-line index entry for a file — shows ALL exports."""
        key = self._file_key(f)
        status = "✓" if key in self._written_files else "○"
        exports = [c.class_name for c in f.classes] + [fn.name for fn in f.functions]
        export_str = ", ".join(exports)
        desc = f.description or ""
        return f"  {status} {key} — {desc} (exports: {export_str})"

    # ── Dependency Graph ─────────────────────────────────────────────────

    def build_dependency_graph(self):
        """Build a dependency graph from architecture file declarations."""
        self._dep_graph.clear()
        file_keys = {self._file_key(f) for f in self.architecture.files}

        for f in self.architecture.files:
            key = self._file_key(f)
            deps = set()
            for dep_name in f.dependencies:
                # Try to resolve dependency name to a file key
                for candidate in file_keys:
                    if candidate.endswith(dep_name) or dep_name in candidate:
                        deps.add(candidate)
                        break
            self._dep_graph[key] = deps

    def get_dependencies(self, file_name: str) -> list[str]:
        """Get direct dependencies for a file."""
        key = self._resolve_file_key(file_name)
        return sorted(self._dep_graph.get(key, set()))

    def get_dependents(self, file_name: str) -> list[str]:
        """Get files that depend on the given file."""
        key = self._resolve_file_key(file_name)
        result = []
        for fk, deps in self._dep_graph.items():
            if key in deps:
                result.append(fk)
        return sorted(result)

    def get_dependency_summary(self) -> str:
        """Get a human-readable dependency graph summary for LLM consumption."""
        if not self._dep_graph:
            return "(no dependency info available)"
        lines = []
        for fk in sorted(self._dep_graph.keys()):
            deps = self._dep_graph[fk]
            if deps:
                dep_status = []
                for d in sorted(deps):
                    status = "✓" if d in self._written_files else "○"
                    dep_status.append(f"{status} {d}")
                lines.append(f"  {fk} → needs: {', '.join(dep_status)}")
            else:
                lines.append(f"  {fk} → no dependencies (root)")
        return "\n".join(lines)

    def get_remaining_files(self) -> list[str]:
        """Get file keys that haven't been written yet."""
        written = set(self._written_files.keys())
        return [self._file_key(f) for f in self.architecture.files if self._file_key(f) not in written]

    def tool_pick_next_file(self, file_name: str) -> str:
        """Validate the agent's file pick and return rich context for it."""
        key = self._resolve_file_key(file_name)
        remaining = self.get_remaining_files()

        if key not in remaining:
            if key in self._written_files:
                return (f"[Error] '{key}' is already written. Pick an unwritten file.\n"
                        f"Remaining: {', '.join(remaining)}")
            # Might be a bad name — list what's available
            return (f"[Error] '{file_name}' not found in project.\n"
                    f"Remaining files: {', '.join(remaining)}")

        # Return architecture design + dependency status
        fd = self.get_file_design(file_name)
        design = self._format_file_design(fd) if fd else "(no design found)"

        deps = self.get_dependencies(file_name)
        dep_lines = []
        all_deps_ready = True
        for d in deps:
            if d in self._written_files:
                dep_lines.append(f"  ✓ {d} (written)")
            else:
                dep_lines.append(f"  ○ {d} (NOT yet written)")
                all_deps_ready = False

        dependents = self.get_dependents(file_name)
        unblocks = [d for d in dependents if d not in self._written_files]

        lines = [
            f"✅ Selected: {key}",
            f"Dependencies ready: {'all satisfied' if all_deps_ready else 'some missing (proceed with care)'}",
        ]
        if dep_lines:
            lines.append("Dependencies:\n" + "\n".join(dep_lines))
        if unblocks:
            lines.append(f"Writing this will unblock: {', '.join(unblocks)}")
        lines.append(f"\n{design}")

        return "\n".join(lines)

    # ── Tool Implementations (called by the ReAct agent) ─────────────────

    def tool_read_file(self, path: str) -> str:
        """Read an already-written file's FULL content — no truncation.
        
        Modern models (DeepSeek V3 128K, GPT-4o 128K, Claude 200K) have
        plenty of context.  Truncating dependency files was the #1 cause
        of cross-file inconsistency (agent can't see some exports → writes
        code that references non-existent functions).
        """
        key = self._resolve_file_key(path)

        # Check in-memory cache first
        if key in self._written_files:
            return self._written_files[key]

        # Try reading from disk
        full_path = os.path.join(self.work_dir, key)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._written_files[key] = content
            return content

        return f"[Error] File not found: {path}"

    def tool_read_architecture(self, file_name: str) -> str:
        """Get the detailed architecture design for a specific file."""
        key = self._resolve_file_key(file_name)
        for f in self.architecture.files:
            if self._file_key(f) == key or f.name == file_name:
                return self._format_file_design(f)
        return f"[Error] No architecture found for: {file_name}"

    def tool_read_threats(self, file_name: str) -> str:
        """Get threat model entries relevant to a file."""
        key = self._resolve_file_key(file_name)
        
        # Search threat model entries for mentions of this file
        relevant = []
        for entry in self.threat_model.entries:
            if file_name in entry.function or key in entry.function:
                relevant.append(entry)

        if relevant:
            lines = []
            for e in relevant:
                lines.append(f"### {e.function}")
                for t in e.threats:
                    lines.append(f"  - {t}")
                lines.append("  Protections:")
                for p in e.protections:
                    lines.append(f"  - {p}")
            return "\n".join(lines)

        # Fallback: return a chunk of the raw threat model mentioning this file
        if self.threat_model.raw_text:
            chunks = self.threat_model.raw_text.split("### Function:")
            for chunk in chunks:
                if file_name in chunk or key.replace("/", ".") in chunk:
                    return f"### Function:{chunk}"

        return f"[Info] No specific threats found for {file_name}. Use general security practices."

    def tool_search_code(self, query: str) -> str:
        """Search across all written files for a pattern/symbol."""
        results = []
        query_lower = query.lower()

        for path, content in self._written_files.items():
            matches = []
            for i, line in enumerate(content.splitlines(), 1):
                if query_lower in line.lower():
                    matches.append(f"  L{i}: {line.strip()}")
            if matches:
                results.append(f"── {path} ──\n" + "\n".join(matches[:15]))
                if len(matches) > 15:
                    results.append(f"  ... and {len(matches) - 15} more matches")

        if not results:
            return f"[Info] No matches found for '{query}'"
        return "\n".join(results)  # no cap — show all file results

    def tool_list_files(self) -> str:
        """List all project files with their status."""
        return self.get_file_index()

    def tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edit an already-written file by replacing old_text with new_text.
        
        The old_text must match exactly once in the file.
        """
        key = self._resolve_file_key(path)

        if key not in self._written_files:
            return f"[Error] File not found: '{path}'. Use read_file or list_files to check."

        content = self._written_files[key]

        count = content.count(old_text)
        if count == 0:
            return (
                f"[Error] old_text not found in '{key}'. "
                f"Make sure you copy the exact text including whitespace and indentation. "
                f"Use read_file('{key}') to see the current content."
            )
        if count > 1:
            return (
                f"[Error] old_text appears {count} times in '{key}'. "
                f"Include more surrounding context to make the match unique."
            )

        new_content = content.replace(old_text, new_text, 1)
        self._written_files[key] = new_content

        full_path = os.path.join(self.work_dir, key)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        self.build_index()

        old_lines = old_text.count("\n") + 1
        new_lines = new_text.count("\n") + 1
        return (
            f"✅ Edited '{key}': replaced {old_lines} line(s) with {new_lines} line(s). "
            f"File now has {len(new_content.splitlines())} lines total."
        )

    def tool_run_command(self, command: str) -> str:
        """Execute a shell command in the workspace and return output."""
        import subprocess

        if not command or not command.strip():
            return "[Error] Empty command."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"[stdout]\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")
            if not output_parts:
                output_parts.append("(no output)")

            status = f"exit code: {result.returncode}"
            full = f"{status}\n" + "\n".join(output_parts)

            return full

        except subprocess.TimeoutExpired:
            return "[Error] Command timed out after 60 seconds."
        except Exception as e:
            return f"[Error] Failed to run command: {e}"

    # ── File Write ───────────────────────────────────────────────────────

    def register_written_file(self, path: str, content: str):
        """Register a newly written file in the context."""
        key = self._normalize_path(path)
        self._written_files[key] = content
        # Rebuild index to update status markers
        self.build_index()

    def get_written_files(self) -> dict[str, str]:
        """Get all written file contents."""
        return dict(self._written_files)

    # ── Architecture Helpers ─────────────────────────────────────────────

    def get_file_design(self, file_name: str) -> Optional[FileDesign]:
        """Get the FileDesign object for a given file."""
        key = self._resolve_file_key(file_name)
        for f in self.architecture.files:
            if self._file_key(f) == key or f.name == file_name:
                return f
        return None

    def get_design_for_prompt(self, file_name: str) -> str:
        """Get a formatted file design suitable for the agent prompt."""
        fd = self.get_file_design(file_name)
        if fd:
            return self._format_file_design(fd)
        return "No design available."

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _file_key(f: FileDesign) -> str:
        path = f.path.strip().rstrip("/")
        if path in ("", ".", "./"):
            return f.name
        path = path.lstrip("./")
        return f"{path}/{f.name}"

    def _resolve_file_key(self, name: str) -> str:
        """Resolve a file name/path to its canonical key."""
        name = self._normalize_path(name)
        # Exact match
        keys = {self._file_key(f) for f in self.architecture.files}
        if name in keys:
            return name
        # Partial match
        for k in keys:
            if k.endswith(name) or name.endswith(k):
                return k
        return name

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = path.strip().replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        return path.lstrip("/")

    @staticmethod
    def _truncate(text: str, max_chars: int = 6000) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:]

    @staticmethod
    def _format_file_design(f: FileDesign) -> str:
        lines = [f"## {f.name}", f"Path: {f.path}", f"Description: {f.description}"]
        if f.dependencies:
            lines.append(f"Dependencies: {', '.join(f.dependencies)}")
        if f.classes:
            lines.append("\nClasses:")
            for cls in f.classes:
                members = ", ".join(f"{m.name}: {m.type}" for m in cls.members)
                lines.append(f"  class {cls.class_name}({members})")
                for method in cls.methods:
                    lines.append(f"    - {method}")
        if f.functions:
            lines.append("\nFunctions:")
            for fn in f.functions:
                params = ", ".join(f"{p.name}: {p.type}" for p in fn.input_parameters)
                returns = ", ".join(f"{p.name}: {p.type}" for p in fn.output_parameters)
                lines.append(f"  {fn.name}({params}) -> {returns}")
                if fn.description:
                    lines.append(f"    {fn.description}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self):
        """Save architecture, threat model, PRD, and written-file manifest to disk."""
        os.makedirs(self.work_dir, exist_ok=True)

        with open(os.path.join(self.work_dir, "prd.txt"), "w") as f:
            f.write(self.prd)

        with open(os.path.join(self.work_dir, "architecture.json"), "w") as f:
            f.write(self.architecture.model_dump_json(indent=2))

        with open(os.path.join(self.work_dir, "threat_model.md"), "w") as f:
            f.write(self.threat_model.raw_text)

        # Persist the list of written file keys so resume can find them reliably
        self.save_manifest()

    def save_manifest(self):
        """Save only the written-file manifest (lightweight, safe to call often)."""
        os.makedirs(self.work_dir, exist_ok=True)
        manifest_path = os.path.join(self.work_dir, ".written_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(sorted(self._written_files.keys()), f, indent=2)

    def load_state(self) -> bool:
        """Load existing state from disk. Returns True if any state was loaded."""
        loaded = False

        prd_path = os.path.join(self.work_dir, "prd.txt")
        if os.path.exists(prd_path):
            with open(prd_path, "r") as f:
                self.prd = f.read()
            loaded = True

        arch_path = os.path.join(self.work_dir, "architecture.json")
        if os.path.exists(arch_path):
            with open(arch_path, "r") as f:
                self.architecture = ArchitectureDesign.model_validate_json(f.read())
            self.build_index()
            self.build_dependency_graph()
            loaded = True

        threat_path = os.path.join(self.work_dir, "threat_model.md")
        if os.path.exists(threat_path):
            with open(threat_path, "r") as f:
                self.threat_model = ThreatModel(raw_text=f.read())
            loaded = True

        # Load already-written code files: manifest first, then scan as fallback
        self._load_written_files()

        return loaded

    def _load_written_files(self):
        """Load written files using manifest (reliable) + disk scan (fallback).
        
        Strategy:
          1. If a manifest exists, load every file listed in it from disk.
          2. Then do a broad disk scan to catch any files the manifest missed
             (e.g. written after the last save_state but before crash).
          3. Cross-reference with architecture file keys for extra safety.
        """
        # Step 1: Load from manifest (most reliable source of truth)
        manifest_path = os.path.join(self.work_dir, ".written_manifest.json")
        manifest_keys: set[str] = set()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest_keys = set(json.load(f))
                for key in manifest_keys:
                    full_path = os.path.join(self.work_dir, key)
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                self._written_files[key] = f.read()
                        except (UnicodeDecodeError, OSError):
                            pass
            except (json.JSONDecodeError, TypeError):
                pass

        # Step 2: Broad disk scan to catch files written after last save_state
        self._scan_written_files()

    def _scan_written_files(self):
        """Scan workspace for existing code files.
        
        Uses a blacklist approach (skip known binary/meta files) instead of
        a whitelist, so we never miss files with uncommon extensions like
        .md, .txt, .sql, .vue, .svelte, .dart, .proto, .graphql, .env,
        or extensionless files like Dockerfile, Makefile, .gitignore, etc.
        """
        # Directories to skip entirely
        EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build",
                        ".venv", "venv", "env", ".tox", ".mypy_cache",
                        ".pytest_cache", ".eggs", "*.egg-info"}
        # Files to skip (binary, images, or pipeline meta files we manage ourselves)
        EXCLUDE_FILES = {".written_manifest.json", "architecture.json",
                         "prd.txt", "threat_model.md"}
        BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
                             ".webp", ".svg", ".mp3", ".mp4", ".wav", ".avi",
                             ".mov", ".zip", ".tar", ".gz", ".bz2", ".7z",
                             ".rar", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
                             ".ppt", ".pptx", ".exe", ".dll", ".so", ".dylib",
                             ".o", ".a", ".pyc", ".pyo", ".class", ".wasm",
                             ".ttf", ".otf", ".woff", ".woff2", ".eot",
                             ".sqlite", ".db", ".lock"}

        if not os.path.isdir(self.work_dir):
            return

        # Build set of expected file keys from architecture for cross-reference
        arch_keys = {self._file_key(f) for f in self.architecture.files} if self.architecture.files else set()

        for root, dirs, files in os.walk(self.work_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                if fname in EXCLUDE_FILES:
                    continue
                # Skip binary files by extension
                _, ext = os.path.splitext(fname)
                if ext.lower() in BINARY_EXTENSIONS:
                    continue

                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.work_dir)

                # Already loaded (e.g. from manifest) — skip
                if rel in self._written_files:
                    continue

                # Only load if it matches an architecture key, or if there's
                # no architecture yet (early stage recovery)
                if arch_keys and rel not in arch_keys:
                    # Try fuzzy match: the scan path might differ slightly
                    if not any(rel.endswith(k) or k.endswith(rel) for k in arch_keys):
                        continue

                try:
                    with open(full, "r", encoding="utf-8") as f:
                        self._written_files[rel] = f.read()
                except (UnicodeDecodeError, OSError):
                    pass  # Binary file or permission issue — skip