"""Environment detection, web fetch, lint, and tech stack analysis tools."""

from __future__ import annotations

import asyncio
import os
import subprocess
import logging
from typing import Any

from .base import Tool
from ..project.state import ProjectState

logger = logging.getLogger("promas")


class RuntimeInfoTool(Tool):
    """Detect installed runtimes, package managers, and their versions."""

    name = "detect_env"
    description = (
        "Detect installed programming runtimes, package managers, and their versions.\n\n"
        "Use this at the START of code generation to know what's available.\n"
        "Returns: which runtimes are installed, their versions, and available package managers.\n\n"
        "Example output:\n"
        "  node: v20.11.0\n"
        "  npm: 10.2.4\n"
        "  python3: 3.11.6\n"
        "  pip: 23.3.1\n"
        "  go: not installed\n\n"
        "This is a read-only tool — safe to call in parallel with others."
    )
    is_read_only = True

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_) -> str:
        return await detect_environment()


class WebFetchTool(Tool):
    """Fetch a URL and return its text content."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return its text content (HTML converted to readable text).\n\n"
        "Use this when you need to:\n"
        "- Look up API documentation for a library\n"
        "- Check the latest version of a package on npm/pypi\n"
        "- Read a reference page for correct usage\n\n"
        "Returns the page text, truncated to ~8000 chars to save context.\n"
        "Read-only — safe to call in parallel with other read tools."
    )
    is_read_only = True

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch, e.g. 'https://docs.python.org/3/library/asyncio.html'",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str = "", **_) -> str:
        if not url:
            return "[Error] url is required"
        return await _fetch_url(url)


class LintCheckTool(Tool):
    """Run a quick syntax/lint check on a file."""

    name = "lint_check"
    description = (
        "Run a language-appropriate syntax check on a written file.\n\n"
        "Auto-detects language from file extension:\n"
        "- .py → python -m py_compile\n"
        "- .js/.ts/.jsx/.tsx → node --check (JS only) or npx tsc --noEmit\n"
        "- .json → json.tool validation\n"
        "- .go → go vet\n\n"
        "Use this after writing a file to catch syntax errors early,\n"
        "before the full verify stage.\n\n"
        "Read-only — does not modify files."
    )
    is_read_only = True

    def __init__(self, state: ProjectState):
        self.state = state

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the file to check",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str = "", **_) -> str:
        if not path:
            return "[Error] path is required"
        return await _lint_file(path, self.state.work_dir)


# ── Implementation ────────────────────────────────────────────────────────────

# Runtime version commands: (display_name, command, args)
_RUNTIME_CHECKS = [
    ("node", "node", ["--version"]),
    ("npm", "npm", ["--version"]),
    ("npx", "npx", ["--version"]),
    ("python3", "python3", ["--version"]),
    ("python", "python", ["--version"]),
    ("pip", "pip", ["--version"]),
    ("pip3", "pip3", ["--version"]),
    ("go", "go", ["version"]),
    ("cargo", "cargo", ["--version"]),
    ("rustc", "rustc", ["--version"]),
    ("java", "java", ["-version"]),
    ("mvn", "mvn", ["--version"]),
    ("php", "php", ["--version"]),
    ("composer", "composer", ["--version"]),
    ("ruby", "ruby", ["--version"]),
    ("gem", "gem", ["--version"]),
    ("dotnet", "dotnet", ["--version"]),
    ("docker", "docker", ["--version"]),
    ("git", "git", ["--version"]),
]


async def detect_environment() -> str:
    """Run version checks concurrently and return a summary."""

    async def _check(name: str, cmd: str, args: list[str]) -> tuple[str, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            output = (stdout or stderr or b"").decode().strip().splitlines()
            version = output[0] if output else "unknown"
            # Clean up verbose output (java -version prints multiple lines)
            version = version.replace(f"{name} version ", "").replace(f"{name} ", "").strip()
            return name, version
        except FileNotFoundError:
            return name, "not installed"
        except asyncio.TimeoutError:
            return name, "timeout"
        except Exception:
            return name, "not installed"

    results = await asyncio.gather(*[
        _check(name, cmd, args) for name, cmd, args in _RUNTIME_CHECKS
    ])

    # Deduplicate python/python3
    seen: dict[str, str] = {}
    for name, version in results:
        if version != "not installed":
            seen[name] = version

    lines = []
    for name, cmd, args in _RUNTIME_CHECKS:
        version = seen.get(name, "not installed")
        if version == "not installed" and name in ("python", "pip"):
            # Skip python/pip if python3/pip3 is available
            if seen.get(f"{name}3"):
                continue
        lines.append(f"  {name}: {version}")

    return "Detected environment:\n" + "\n".join(lines)


async def _fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch URL content. Uses curl + simple HTML→text conversion."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sL", "--max-time", "10",
            "-H", "User-Agent: Mozilla/5.0 (PROMAS Agent)",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            return f"[Error] Failed to fetch {url}: {stderr.decode()[:200]}"

        content = stdout.decode("utf-8", errors="replace")

        # Simple HTML → text conversion
        if "<html" in content.lower() or "<body" in content.lower():
            content = _html_to_text(content)

        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (truncated, {len(content)} total chars)"

        return content

    except asyncio.TimeoutError:
        return f"[Error] Timeout fetching {url}"
    except FileNotFoundError:
        return "[Error] curl not installed"
    except Exception as e:
        return f"[Error] {e}"


def _html_to_text(html: str) -> str:
    """Minimal HTML→text. Strips tags, decodes common entities."""
    import re
    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
    # Convert <br>, <p>, <div>, <li> to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.I)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_LINT_COMMANDS: dict[str, list[str]] = {
    ".py": ["python3", "-m", "py_compile"],
    ".js": ["node", "--check"],
    ".mjs": ["node", "--check"],
    ".json": ["python3", "-m", "json.tool"],
    ".go": ["go", "vet"],
    ".php": ["php", "-l"],
    ".rb": ["ruby", "-c"],
    ".yaml": ["python3", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"],
    ".yml": ["python3", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"],
    ".xml": ["python3", "-c", "import xml.etree.ElementTree as ET,sys; ET.parse(sys.argv[1])"],
}


async def _lint_file(path: str, work_dir: str) -> str:
    """Run appropriate lint command for a file."""
    import os
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    full_path = os.path.join(work_dir, path)
    if not os.path.exists(full_path):
        return f"[Error] File not found: {path}"

    cmd_template = _LINT_COMMANDS.get(ext)
    if not cmd_template:
        return f"[Info] No lint command configured for {ext} files"

    if ext == ".json":
        # json.tool reads from stdin or file arg
        cmd = cmd_template + [full_path]
    elif ext == ".py":
        cmd = cmd_template + [full_path]
    elif ext == ".js":
        cmd = cmd_template + [full_path]
    elif ext == ".go":
        cmd = cmd_template + [os.path.dirname(full_path) or "."]
    else:
        cmd = cmd_template + [full_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = (stdout.decode() + stderr.decode()).strip()

        if proc.returncode == 0:
            return f"✓ {path}: syntax OK"
        return f"✗ {path}: lint errors\n{output}"

    except FileNotFoundError:
        return f"[Info] Lint tool not available for {ext} (command not found)"
    except asyncio.TimeoutError:
        return f"[Error] Lint check timed out for {path}"
    except Exception as e:
        return f"[Error] {e}"


# ── Tech stack detection ──────────────────────────────────────────────────────

# Extension → stack tag(s)
_EXT_TO_STACK: dict[str, list[str]] = {
    ".py": ["python"],
    ".js": ["javascript", "node"],
    ".mjs": ["javascript", "node"],
    ".ts": ["typescript", "node"],
    ".tsx": ["typescript", "react", "node"],
    ".jsx": ["javascript", "react", "node"],
    ".vue": ["vue", "node"],
    ".svelte": ["svelte", "node"],
    ".php": ["php"],
    ".go": ["go"],
    ".rs": ["rust"],
    ".java": ["java"],
    ".kt": ["kotlin", "java"],
    ".rb": ["ruby"],
    ".cs": ["csharp", "dotnet"],
    ".swift": ["swift"],
    ".dart": ["dart", "flutter"],
    ".c": ["c"],
    ".cpp": ["cpp"],
    ".h": ["c"],
    ".hpp": ["cpp"],
}

# Filename → stack tag(s)
_FILE_TO_STACK: dict[str, list[str]] = {
    "package.json": ["node"],
    "tsconfig.json": ["typescript", "node"],
    "next.config.js": ["nextjs", "react", "node"],
    "next.config.ts": ["nextjs", "react", "node"],
    "next.config.mjs": ["nextjs", "react", "node"],
    "vite.config.ts": ["vite", "node"],
    "vite.config.js": ["vite", "node"],
    "webpack.config.js": ["webpack", "node"],
    "tailwind.config.js": ["tailwind", "node"],
    "tailwind.config.ts": ["tailwind", "node"],
    "composer.json": ["php", "composer"],
    "artisan": ["php", "laravel"],
    "requirements.txt": ["python", "pip"],
    "pyproject.toml": ["python"],
    "setup.py": ["python"],
    "Pipfile": ["python", "pipenv"],
    "go.mod": ["go"],
    "Cargo.toml": ["rust"],
    "pom.xml": ["java", "maven"],
    "build.gradle": ["java", "gradle"],
    "Gemfile": ["ruby"],
    "Dockerfile": ["docker"],
    "docker-compose.yml": ["docker"],
    ".env": ["dotenv"],
    ".env.example": ["dotenv"],
}

# Stack → verify instructions
_STACK_VERIFY: dict[str, dict[str, str]] = {
    "node": {
        "install": "npm install",
        "build": "npm run build (if build script exists in package.json)",
        "lint": "npx eslint . --ext .js,.ts,.jsx,.tsx (if eslint configured)",
        "typecheck": "npx tsc --noEmit (if tsconfig.json exists)",
        "smoke": "node -e \"require('./dist/index.js')\" or node -e \"require('./src/index.js')\"",
        "test": "npm test (if test script exists in package.json)",
    },
    "typescript": {
        "typecheck": "npx tsc --noEmit",
    },
    "react": {
        "build": "npm run build",
        "smoke": "Check that the dev server starts: npm run dev (kill after 5s)",
    },
    "nextjs": {
        "build": "npm run build",
        "smoke": "npm run dev &; sleep 3; curl -s http://localhost:3000; kill %1",
    },
    "python": {
        "install": "pip install -e . (if setup.py/pyproject.toml) or pip install -r requirements.txt",
        "lint": "python -m py_compile <each .py file>",
        "typecheck": "mypy . (if mypy installed)",
        "smoke": "python -c \"import <main_module>\"",
        "test": "pytest (if tests exist)",
    },
    "php": {
        "install": "composer install (if composer.json exists)",
        "lint": "php -l <each .php file>",
        "smoke": "php -r \"require 'index.php';\" or php -S localhost:8080 (kill after 5s)",
        "test": "php vendor/bin/phpunit (if phpunit configured)",
        "probe_xss": "curl with <script>alert(1)</script> in input fields",
        "probe_sqli": "curl with ' OR 1=1 -- in input fields",
        "probe_path": "curl with ../../etc/passwd in file parameters",
    },
    "laravel": {
        "install": "composer install && php artisan key:generate",
        "build": "php artisan config:cache",
        "smoke": "php artisan serve &; sleep 3; curl -s http://localhost:8000; kill %1",
        "test": "php artisan test",
    },
    "go": {
        "install": "go mod tidy",
        "build": "go build ./...",
        "lint": "go vet ./...",
        "test": "go test ./...",
        "smoke": "go run . (kill after 5s if it's a server)",
    },
    "rust": {
        "build": "cargo build",
        "lint": "cargo clippy (if clippy installed)",
        "test": "cargo test",
    },
    "java": {
        "build": "mvn compile or gradle build",
        "test": "mvn test or gradle test",
    },
    "ruby": {
        "install": "bundle install",
        "lint": "ruby -c <each .rb file>",
        "test": "bundle exec rspec or bundle exec rake test",
    },
    "csharp": {
        "build": "dotnet build",
        "test": "dotnet test",
    },
    "docker": {
        "build": "docker build -t test-app .",
        "smoke": "docker run --rm -d -p 8080:8080 test-app; sleep 3; curl http://localhost:8080; docker stop ...",
    },
}


def detect_stack(files: list) -> dict[str, Any]:
    """Detect the project's tech stack from architecture file list.

    Returns:
        {
            "tags": {"node", "typescript", "react", ...},
            "primary": "typescript",  # most common
            "verify_instructions": "... language-specific commands ..."
        }
    """
    tags: dict[str, int] = {}  # tag → count

    for f in files:
        name = f.name if hasattr(f, "name") else str(f)
        _, ext = os.path.splitext(name)
        ext = ext.lower()

        # Check filename match first (more specific)
        if name in _FILE_TO_STACK:
            for tag in _FILE_TO_STACK[name]:
                tags[tag] = tags.get(tag, 0) + 2  # weight filenames higher

        # Then extension match
        if ext in _EXT_TO_STACK:
            for tag in _EXT_TO_STACK[ext]:
                tags[tag] = tags.get(tag, 0) + 1

    if not tags:
        return {
            "tags": set(),
            "primary": "unknown",
            "verify_instructions": "",
        }

    # Primary = most common tag
    primary = max(tags, key=tags.get)

    # Build verify instructions from all matched stacks
    instructions_parts: list[str] = []
    seen_cmds: set[str] = set()

    for tag in sorted(tags, key=tags.get, reverse=True):
        if tag not in _STACK_VERIFY:
            continue
        cmds = _STACK_VERIFY[tag]
        for step, cmd in cmds.items():
            key = f"{step}:{cmd}"
            if key not in seen_cmds:
                instructions_parts.append(f"- **{step}**: `{cmd}`")
                seen_cmds.add(key)

    verify_text = ""
    if instructions_parts:
        tag_list = ", ".join(sorted(tags, key=tags.get, reverse=True)[:5])
        verify_text = (
            f"Detected tech stack: **{tag_list}**\n\n"
            f"Recommended verify commands:\n" + "\n".join(instructions_parts)
        )

    return {
        "tags": set(tags),
        "primary": primary,
        "verify_instructions": verify_text,
    }
