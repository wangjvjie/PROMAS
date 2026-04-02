"""ProjectState — project data, file index, dependency graph, persistence.

Responsibilities (only these):
  - Store PRD, architecture, threat model, written files
  - Build and serve file index + dependency graph
  - Read/write files to disk
  - Save/load state from workspace

No tool logic lives here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..models import ArchitectureDesign, FileDesign, ThreatModel


@dataclass
class ProjectState:
    work_dir: str = "./workspace"
    prd: str = ""
    architecture: ArchitectureDesign = field(default_factory=ArchitectureDesign)
    threat_model: ThreatModel = field(default_factory=ThreatModel)

    _written: dict[str, str] = field(default_factory=dict)   # rel_path → content
    _file_index: str = ""
    _dep_graph: dict[str, set[str]] = field(default_factory=dict)

    # ── File Index ────────────────────────────────────────────────────────────

    def build_index(self) -> str:
        lines = []
        for f in self.architecture.files:
            key = self._file_key(f)
            status = "✓" if key in self._written else "○"
            exports = [c.class_name for c in f.classes] + [fn.name for fn in f.functions]
            export_str = ", ".join(exports) if exports else "(none)"
            desc = f.description or "no description"
            deps = f.dependencies
            dep_str = f" [imports: {', '.join(deps)}]" if deps else ""
            lines.append(f"  {status} {key} — {desc} (exports: {export_str}){dep_str}")
        self._file_index = "Project Files:\n" + "\n".join(lines)
        return self._file_index

    def get_file_index(self) -> str:
        if not self._file_index:
            self.build_index()
        return self._file_index

    # ── Architecture Views ────────────────────────────────────────────────────

    def get_full_arch(self) -> str:
        """Full architecture: every file's classes and function signatures."""
        if not self.architecture.files:
            return "(no architecture)"
        return "\n\n".join(self._fmt_file(f) for f in self.architecture.files)

    def get_focused_arch(self, file_name: str) -> str:
        """Architecture focused on one file: target + deps + dependents in full,
        everything else as index lines."""
        if not self.architecture.files:
            return "(no architecture)"

        target_key = self._resolve_key(file_name)
        deps = set(self.get_dependencies(file_name))
        dependents = set(self.get_dependents(file_name))

        target_section = ""
        dep_sections: list[str] = []
        dependent_sections: list[str] = []
        other_lines: list[str] = []

        for f in self.architecture.files:
            key = self._file_key(f)
            if key == target_key:
                target_section = self._fmt_file(f)
            elif key in deps:
                dep_sections.append(self._fmt_file(f))
            elif key in dependents:
                dependent_sections.append(self._fmt_file(f))
            else:
                other_lines.append(self._fmt_index_line(f))

        parts = [f"## Target File\n{target_section}"]
        if dep_sections:
            parts.append("## Dependencies (full API)\n" + "\n\n".join(dep_sections))
        if dependent_sections:
            parts.append("## Dependents (full API)\n" + "\n\n".join(dependent_sections))
        if other_lines:
            parts.append("## Other Files (index only)\n" + "\n".join(other_lines))
        return "\n\n".join(parts)

    def _fmt_file(self, f: FileDesign) -> str:
        key = self._file_key(f)
        status = "✓" if key in self._written else "○"
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
            returns = ", ".join(p.type for p in fn.output_parameters) or "None"
            desc = f"  # {fn.description}" if fn.description else ""
            lines.append(f"  def {fn.name}({params}) -> {returns}{desc}")
        return "\n".join(lines)

    def _fmt_index_line(self, f: FileDesign) -> str:
        key = self._file_key(f)
        status = "✓" if key in self._written else "○"
        exports = [c.class_name for c in f.classes] + [fn.name for fn in f.functions]
        return f"  {status} {key} — {f.description} (exports: {', '.join(exports)})"

    def get_file_design_text(self, file_name: str) -> str:
        fd = self._get_file_design(file_name)
        if not fd:
            return f"(no design for {file_name})"
        key = self._file_key(fd)
        lines = [f"## {fd.name}", f"Path: {fd.path}", f"Description: {fd.description}"]
        if fd.dependencies:
            lines.append(f"Dependencies: {', '.join(fd.dependencies)}")
        if fd.classes:
            lines.append("\nClasses:")
            for cls in fd.classes:
                members = ", ".join(f"{m.name}: {m.type}" for m in cls.members)
                lines.append(f"  class {cls.class_name}({members})")
                for method in cls.methods:
                    lines.append(f"    - {method}")
        if fd.functions:
            lines.append("\nFunctions:")
            for fn in fd.functions:
                params = ", ".join(f"{p.name}: {p.type}" for p in fn.input_parameters)
                returns = ", ".join(f"{p.name}: {p.type}" for p in fn.output_parameters)
                lines.append(f"  {fn.name}({params}) -> {returns}")
                if fn.description:
                    lines.append(f"    {fn.description}")
        return "\n".join(lines)

    # ── Dependency Graph ──────────────────────────────────────────────────────

    def build_dep_graph(self):
        self._dep_graph.clear()
        file_keys = {self._file_key(f) for f in self.architecture.files}
        for f in self.architecture.files:
            key = self._file_key(f)
            deps: set[str] = set()
            for dep_name in f.dependencies:
                for candidate in file_keys:
                    if candidate.endswith(dep_name) or dep_name in candidate:
                        deps.add(candidate)
                        break
            self._dep_graph[key] = deps

    def get_dependencies(self, file_name: str) -> list[str]:
        key = self._resolve_key(file_name)
        return sorted(self._dep_graph.get(key, set()))

    def get_dependents(self, file_name: str) -> list[str]:
        key = self._resolve_key(file_name)
        return sorted(fk for fk, deps in self._dep_graph.items() if key in deps)

    def get_dep_summary(self) -> str:
        if not self._dep_graph:
            return "(no dependency info)"
        lines = []
        for fk in sorted(self._dep_graph):
            deps = self._dep_graph[fk]
            if deps:
                dep_status = [
                    f"{'✓' if d in self._written else '○'} {d}" for d in sorted(deps)
                ]
                lines.append(f"  {fk} → needs: {', '.join(dep_status)}")
            else:
                lines.append(f"  {fk} → no dependencies (root)")
        return "\n".join(lines)

    def get_remaining(self) -> list[str]:
        written = set(self._written)
        return [self._file_key(f) for f in self.architecture.files
                if self._file_key(f) not in written]

    # ── File I/O ──────────────────────────────────────────────────────────────

    def read_file(self, path: str) -> str:
        """Read file and return with line numbers (cat -n style, like claude-code)."""
        key = self._resolve_key(path)
        content: str | None = None

        if key in self._written:
            content = self._written[key]
        else:
            full = os.path.join(self.work_dir, key)
            if os.path.exists(full):
                try:
                    content = open(full, encoding="utf-8").read()
                    self._written[key] = content
                except Exception as e:
                    return f"[Error] Cannot read {path}: {e}"

        if content is None:
            return f"[Error] File not found: {path}"

        # Format with line numbers (like FileReadTool in claude-code)
        lines = content.splitlines()
        width = len(str(len(lines)))
        numbered = [f"{i:>{width}}\t{line}" for i, line in enumerate(lines, 1)]
        return "\n".join(numbered)

    def write_file(self, path: str, content: str):
        # Workspace boundary check BEFORE normalization — use raw path to detect traversal
        workspace_real = os.path.realpath(self.work_dir)
        raw_full = os.path.realpath(os.path.join(self.work_dir, path))
        if not raw_full.startswith(workspace_real + os.sep) and raw_full != workspace_real:
            raise ValueError(
                f"Path traversal blocked: '{path}' resolves outside workspace"
            )
        key = self._normalize(path)
        full = os.path.join(self.work_dir, key)
        self._written[key] = content
        os.makedirs(os.path.dirname(full) if os.path.dirname(full) else ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        self.build_index()

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edit file and return a unified diff (like claude-code's FileEditTool)."""
        key = self._resolve_key(path)
        if key not in self._written:
            return f"[Error] File not found: '{path}'. Use list_files to check available files."
        content = self._written[key]
        count = content.count(old_text)
        if count == 0:
            return (
                f"[Error] old_text not found in '{key}'. "
                "Copy exact text including whitespace. Use read_file to see current content."
            )
        if count > 1:
            return (
                f"[Error] old_text matches {count} times in '{key}'. "
                "Include more surrounding context to make it unique."
            )
        new_content = content.replace(old_text, new_text, 1)
        self.write_file(key, new_content)

        # Generate a unified diff (like claude-code's structuredPatch)
        diff = _unified_diff(key, content, new_content)
        return f"The file {key} has been updated successfully.\n\n{diff}"

    def search_code(self, query: str) -> str:
        """Search files — returns ripgrep-style output (path:line:content)."""
        results: list[str] = []
        query_lower = query.lower()
        total_matches = 0
        for path, content in sorted(self._written.items()):
            file_matches: list[str] = []
            for i, line in enumerate(content.splitlines(), 1):
                if query_lower in line.lower():
                    # ripgrep style: file:line:content
                    file_matches.append(f"{path}:{i}:{line.rstrip()}")
            if file_matches:
                total_matches += len(file_matches)
                results.extend(file_matches[:20])
                if len(file_matches) > 20:
                    results.append(f"  ... {len(file_matches) - 20} more matches in {path}")
        if not results:
            return f"No matches found for '{query}'"
        header = f"Found {total_matches} match(es) across {len(set(r.split(':')[0] for r in results if ':' in r))} file(s)"
        return header + "\n" + "\n".join(results)

    def pick_next_file(self, file_name: str) -> tuple[str, str]:
        """Validate a file pick. Returns (resolved_key, observation_text)."""
        key = self._resolve_key(file_name)
        remaining = self.get_remaining()

        if key not in remaining:
            if key in self._written:
                return "", f"[Error] '{key}' is already written. Remaining: {', '.join(remaining)}"
            return "", f"[Error] '{file_name}' not found. Remaining: {', '.join(remaining)}"

        fd = self._get_file_design(file_name)
        design = self.get_file_design_text(file_name)
        deps = self.get_dependencies(file_name)
        dep_lines = [
            f"  {'✓' if d in self._written else '○'} {d}"
            for d in deps
        ]
        dependents = self.get_dependents(file_name)
        unblocks = [d for d in dependents if d not in self._written]
        all_deps_ready = all(d in self._written for d in deps)

        lines = [
            f"✅ Selected: {key}",
            f"Dependencies: {'all satisfied' if all_deps_ready else 'some missing — check before importing'}",
        ]
        if dep_lines:
            lines.append("\n".join(dep_lines))
        if unblocks:
            lines.append(f"Writing this unblocks: {', '.join(unblocks)}")
        lines.append(f"\n{design}")
        lines.append(f"\n## Focused Architecture\n{self.get_focused_arch(key)}")

        return key, "\n".join(lines)

    def get_written(self) -> dict[str, str]:
        return dict(self._written)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self.work_dir, exist_ok=True)
        _write(os.path.join(self.work_dir, "prd.txt"), self.prd)
        _write(os.path.join(self.work_dir, "architecture.json"),
               self.architecture.model_dump_json(indent=2))
        _write(os.path.join(self.work_dir, "threat_model.md"), self.threat_model.raw_text)
        self.save_manifest()

    def save_manifest(self):
        os.makedirs(self.work_dir, exist_ok=True)
        path = os.path.join(self.work_dir, ".written_manifest.json")
        with open(path, "w") as f:
            json.dump(sorted(self._written), f, indent=2)

    def load(self) -> bool:
        loaded = False
        prd_path = os.path.join(self.work_dir, "prd.txt")
        if os.path.exists(prd_path):
            self.prd = open(prd_path).read()
            loaded = True

        arch_path = os.path.join(self.work_dir, "architecture.json")
        if os.path.exists(arch_path):
            self.architecture = ArchitectureDesign.model_validate_json(open(arch_path).read())
            self.build_index()
            self.build_dep_graph()
            loaded = True

        threat_path = os.path.join(self.work_dir, "threat_model.md")
        if os.path.exists(threat_path):
            self.threat_model = ThreatModel(raw_text=open(threat_path).read())
            loaded = True

        self._load_written_files()
        return loaded

    def _load_written_files(self):
        # Load from manifest first (reliable), then scan for anything missed
        manifest_path = os.path.join(self.work_dir, ".written_manifest.json")
        if os.path.exists(manifest_path):
            try:
                keys = json.load(open(manifest_path))
                for key in keys:
                    full = os.path.join(self.work_dir, key)
                    if os.path.exists(full):
                        try:
                            self._written[key] = open(full, encoding="utf-8").read()
                        except (UnicodeDecodeError, OSError):
                            pass
            except (json.JSONDecodeError, TypeError):
                pass
        self._scan_workspace()

    def _scan_workspace(self):
        SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build",
                     ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}
        SKIP_FILES = {".written_manifest.json", "architecture.json",
                      "prd.txt", "threat_model.md", "session.json"}
        BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                       ".mp3", ".mp4", ".wav", ".zip", ".tar", ".gz", ".pdf",
                       ".doc", ".docx", ".exe", ".dll", ".so", ".pyc", ".wasm",
                       ".sqlite", ".db", ".lock", ".ttf", ".woff", ".woff2"}

        if not os.path.isdir(self.work_dir):
            return

        arch_keys = {self._file_key(f) for f in self.architecture.files} if self.architecture.files else set()

        for root, dirs, files in os.walk(self.work_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if fname in SKIP_FILES:
                    continue
                _, ext = os.path.splitext(fname)
                if ext.lower() in BINARY_EXTS:
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.work_dir)
                if rel in self._written:
                    continue
                if arch_keys and rel not in arch_keys:
                    if not any(rel.endswith(k) or k.endswith(rel) for k in arch_keys):
                        continue
                try:
                    self._written[rel] = open(full, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _file_key(f: FileDesign) -> str:
        path = f.path.strip().rstrip("/")
        # Collapse all leading ./ sequences
        while path.startswith("./"):
            path = path[2:]
        path = path.lstrip(".").lstrip("/")
        if not path:
            return f.name
        return f"{path}/{f.name}"

    def _resolve_key(self, name: str) -> str:
        name = self._normalize(name)
        keys = {self._file_key(f) for f in self.architecture.files}
        if name in keys:
            return name
        for k in keys:
            if k.endswith(name) or name.endswith(k):
                return k
        return name

    @staticmethod
    def _normalize(path: str) -> str:
        """Normalize a relative path. Collapses ./ prefixes, blocks path traversal."""
        path = path.strip().replace("\\", "/")
        # Collapse all leading ./
        while path.startswith("./"):
            path = path[2:]
        # Block path traversal
        if ".." in path:
            path = path.replace("..", "").replace("//", "/")
        return path.lstrip("/")

    def _get_file_design(self, file_name: str) -> Optional[FileDesign]:
        key = self._resolve_key(file_name)
        for f in self.architecture.files:
            if self._file_key(f) == key or f.name == file_name:
                return f
        return None


def _write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _unified_diff(file_path: str, old: str, new: str, context: int = 3) -> str:
    """Generate a compact unified diff (like claude-code's structuredPatch)."""
    import difflib
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=context,
    )
    return "".join(diff).rstrip()
