from __future__ import annotations

from pathlib import Path

from .utils import safe_rel_path


class Workspace:
    # 只把“真正的代码/脚本/构建文件”放这里；默认不含 json/jsonl
    CODE_EXT = {
        ".py", ".pyi",
        ".c", ".h", ".cc", ".cpp", ".hpp",
        ".go",
        ".rs",
        ".java", ".kt",
        ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
        ".vue",
        ".php",
        ".rb",
        ".cs",
        ".swift",
        ".scala",
        ".pl",
        ".lua",
        ".sh", ".bash", ".zsh",
        ".ps1",
        ".sql",
        ".html", ".css",
        ".md", ".markdown", ".mdx",  # 你如果不想把文档算“代码文件”，可删
        ".yml", ".yaml", ".toml", ".ini", ".cfg",  # 你如果不想带配置文件，可删
        ".mk",
    }

    # 默认排除的扩展名（你提到的 json/jsonl）
    DEFAULT_EXCLUDE_EXT = {".json", ".jsonl"}

    # 项目运行过程生成的 meta 文件，不应该被喂给 LLM 当“代码上下文”
    META_FILES = {
        "prd.txt",
        "arch.txt",
        "arch.json",
        "arch.progress.json",
        "arch.consistency.json",
        "threat_model.txt",
        "threat_model.json",
        "threat_model_modules.json",
        "progress.jsonl",
        "workspace_manifest.json",
    }

    # 扫描时跳过这些目录（防 node_modules / .git 巨量文件）
    SKIP_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: str) -> Path:
        safe = safe_rel_path(rel_path)
        abs_path = (self.root / safe).resolve()
        if not str(abs_path).startswith(str(self.root)):
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return abs_path

    def exists(self, rel_path: str) -> bool:
        return self.resolve(rel_path).exists()

    def read_text(self, rel_path: str) -> str:
        path = self.resolve(rel_path)
        return path.read_text(encoding="utf-8")

    def write_text(self, rel_path: str, content: str) -> str:
        path = self.resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def append_text(self, rel_path: str, content: str) -> str:
        path = self.resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def os_path(self) -> str:
        return str(self.root)

    def _should_skip_path(self, p: Path) -> bool:
        # 跳过在 SKIP_DIRS 下的任何文件
        try:
            rel = p.relative_to(self.root)
        except Exception:
            return True
        if any(part in self.SKIP_DIRS for part in rel.parts):
            return True
        return False

    def list_code_files(self) -> list[str]:
        """
        保留原接口：返回“代码文件列表”（不排除 json/jsonl）。
        如果你希望原接口也排除 json/jsonl，可以把 exclude_exts 改默认即可。
        """
        files: list[str] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if self._should_skip_path(p):
                continue

            ext = p.suffix.lower()
            name = p.name

            # 特殊文件：Makefile / Dockerfile 也算代码文件
            special = name in {"Makefile", "Dockerfile"}
            if not special and ext not in self.CODE_EXT and ext not in {".json"}:
                continue

            rel = p.relative_to(self.root).as_posix()
            if rel in self.META_FILES:
                continue

            files.append(rel)

        files.sort()
        return files

    def list_readable_code_files(
        self,
        *,
        exclude_exts: set[str] | None = None,
        max_files: int = 4000,
    ) -> list[str]:
        """
        你要的“只给他代码文件，去掉 json/jsonl”：
        - 默认排除 .json / .jsonl
        - 仍然会跳过 META_FILES
        """
        if exclude_exts is None:
            exclude_exts = set(self.DEFAULT_EXCLUDE_EXT)

        out: list[str] = []
        for p in self.root.rglob("*"):
            if len(out) >= max_files:
                break
            if not p.is_file():
                continue
            if self._should_skip_path(p):
                continue

            ext = p.suffix.lower()
            name = p.name

            if ext in exclude_exts:
                continue

            special = name in {"Makefile", "Dockerfile"}
            if not special and ext not in self.CODE_EXT:
                continue

            rel = p.relative_to(self.root).as_posix()
            if rel in self.META_FILES:
                continue

            out.append(rel)

        out.sort()
        return out

    def render_code_tree(
        self,
        *,
        exclude_exts: set[str] | None = None,
        max_files: int = 4000,
    ) -> str:
        """
        把“可读代码文件列表”渲染成树状结构，适合直接塞进 prompt。
        """
        files = self.list_readable_code_files(exclude_exts=exclude_exts, max_files=max_files)
        if not files:
            return "(no code files found)"

        tree: dict = {}
        for f in files:
            node = tree
            parts = f.split("/")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node.setdefault("__files__", []).append(parts[-1])

        def render(node: dict, prefix: str = "") -> list[str]:
            lines: list[str] = []
            dirs = sorted([k for k in node.keys() if k != "__files__"])
            local_files = sorted(node.get("__files__", []))

            # 先输出文件，再输出目录（你也可以反过来）
            for i, fn in enumerate(local_files):
                is_last = (i == len(local_files) - 1) and (not dirs)
                branch = "└── " if is_last else "├── "
                lines.append(prefix + branch + fn)

            for j, d in enumerate(dirs):
                is_last_dir = (j == len(dirs) - 1)
                branch = "└── " if is_last_dir else "├── "
                lines.append(prefix + branch + d + "/")
                child_prefix = prefix + ("    " if is_last_dir else "│   ")
                lines.extend(render(node[d], child_prefix))

            return lines

        return "\n".join(render(tree))

    def load_code_blocks(
        self,
        *,
        exclude_exts: set[str] | None = None,
        max_files: int = 4000,
    ) -> list[str]:
        """
        原来的 load_code_blocks 会把 json/md/yaml 都塞进去。
        这里改成默认走 list_readable_code_files（会去掉 json/jsonl），更符合你现在诉求。
        """
        blocks: list[str] = []
        for rel in self.list_readable_code_files(exclude_exts=exclude_exts, max_files=max_files):
            try:
                content = self.read_text(rel)
            except Exception:
                continue
            blocks.append(
                "\n".join(
                    [
                        "<file>",
                        "## File",
                        rel,
                        "",
                        "## Code",
                        content,
                        "</file>",
                    ]
                )
            )
        return blocks

    def load_state_file(self, file_name: str) -> str:
        path = self.resolve(file_name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_state_file(self, file_name: str, content: str) -> None:
        self.write_text(file_name, content)

    def scan_workspace_manifest(self, *, max_chars_per_file: int = 1200) -> list[dict]:
        """
        修复你原来那个 `return manifest ,` 的问题，并默认不扫 json/jsonl。
        """
        manifest: list[dict] = []
        for rel in self.list_readable_code_files():
            try:
                text = self.read_text(rel)
            except Exception:
                continue
            lines = text.splitlines()
            head = "\n".join(lines[:20])[:max_chars_per_file]
            manifest.append(
                {
                    "path": rel,
                    "lines": len(lines),
                    "chars": len(text),
                    "summary": head,
                }
            )
        return manifest
