"""Architecture stage — file list design + per-file API design with dependency-ordered concurrency.

Improvements over old batch approach:
  1. Files designed ONE at a time (small JSON ~200-500 tokens, never truncated)
  2. Topological sort by dependencies → design in dependency order
  3. Files in the same dependency layer run concurrently (asyncio.gather)
  4. Each file sees FULL API of all previously designed files (no compressed summary)
  5. json_mode=True forces valid JSON output where supported
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from typing import AsyncGenerator

from ...models import (
    SSEvent, EventType, ArchitectureDesign, FileDesign,
    ClassSpec, FunctionSpec, Parameter,
)
from ...project.state import ProjectState
from ...llm.client import LLMClient
from ...prompts import (
    WRITE_FILE_DESIGN_PROMPT,
    WRITE_SINGLE_FILE_ARCH_PROMPT,
)

logger = logging.getLogger("promas")


async def run_architecture(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
    max_retries: int = 3,
) -> AsyncGenerator[SSEvent, None]:

    # ── Step 1: File list ─────────────────────────────────────────────────────
    yield SSEvent(type=EventType.LOG, stage="architecture",
                  content="Designing file structure...")

    p = WRITE_FILE_DESIGN_PROMPT.format(msg=prompt, prd=state.prd)
    response = await llm.chat([{"role": "user", "content": p}], json_mode=True)
    files_data = _extract_json(response)
    raw_files = files_data.get("files", [])

    file_designs: list[FileDesign] = []
    for f in raw_files:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        path = _norm_path(f.get("path", "./"))
        deps = f.get("dependencies", [])
        if not isinstance(deps, list):
            deps = [str(deps)] if deps else []
        file_designs.append(FileDesign(
            name=str(f["name"]),
            path=path,
            description=str(f.get("description", "")),
            dependencies=[str(d) for d in deps],
        ))

    if not file_designs:
        raise ValueError("File design returned 0 files")

    yield SSEvent(type=EventType.LOG, stage="architecture",
                  content=f"Designed {len(file_designs)} files")

    # ── Step 2: Topological sort into layers ──────────────────────────────────
    layers = _toposort_layers(file_designs)
    layer_summary = ", ".join(f"L{i}({len(layer)})" for i, layer in enumerate(layers))
    yield SSEvent(type=EventType.LOG, stage="architecture",
                  content=f"Dependency layers: {layer_summary}")

    # ── Step 3: Design each layer (concurrent within layer) ───────────────────
    all_files_json = json.dumps(
        {"files": [
            {"name": f.name, "path": f.path, "description": f.description,
             "dependencies": f.dependencies}
            for f in file_designs
        ]}, indent=2, ensure_ascii=False)

    designed: list[FileDesign] = []
    failed_count = 0

    for layer_idx, layer_files in enumerate(layers):
        layer_names = [f.name for f in layer_files]
        yield SSEvent(type=EventType.LOG, stage="architecture",
                      content=f"Layer {layer_idx}: {', '.join(layer_names)}")

        # Build full context of everything designed so far
        designed_context = _format_designed_context(designed)

        # Design all files in this layer concurrently
        tasks = [
            _design_one_file(
                target=f,
                prompt=prompt,
                prd=state.prd,
                all_files_json=all_files_json,
                designed_context=designed_context,
                llm=llm,
                max_retries=max_retries,
            )
            for f in layer_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for f, result in zip(layer_files, results):
            if isinstance(result, Exception):
                yield SSEvent(type=EventType.WARN, stage="architecture",
                              content=f"  {f.name} failed: {result} — using stub")
                designed.append(FileDesign(
                    name=f.name, path=f.path,
                    description=f.description or "(design failed)",
                    dependencies=f.dependencies,
                ))
                failed_count += 1
            else:
                designed.append(result)

    if not designed:
        raise ValueError("Architecture produced 0 files")

    # ── Step 4: Cross-file validation + LLM reconciliation ────────────────────
    yield SSEvent(type=EventType.LOG, stage="architecture",
                  content="Validating cross-file consistency...")

    issues = _validate_architecture(designed)

    if issues:
        issue_summary = "\n".join(f"  - {iss}" for iss in issues[:20])
        yield SSEvent(type=EventType.WARN, stage="architecture",
                      content=f"Found {len(issues)} issue(s):\n{issue_summary}")

        yield SSEvent(type=EventType.LOG, stage="architecture",
                      content="Reconciling issues via LLM...")

        designed = await _reconcile_architecture(
            designed, issues, prompt, state.prd, all_files_json, llm
        )
        yield SSEvent(type=EventType.LOG, stage="architecture",
                      content="Reconciliation complete")

        # Re-validate after fix
        remaining_issues = _validate_architecture(designed)
        if remaining_issues:
            yield SSEvent(type=EventType.WARN, stage="architecture",
                          content=f"{len(remaining_issues)} issue(s) remain after reconciliation "
                                  f"(will be caught during code gen)")
    else:
        yield SSEvent(type=EventType.LOG, stage="architecture",
                      content="Validation passed — no cross-file inconsistencies")

    state.architecture = ArchitectureDesign(files=designed)
    state.build_index()
    state.build_dep_graph()

    summary = f"Architecture complete: {len(designed)} files designed"
    if failed_count:
        summary += f" ({failed_count} used stubs)"
    yield SSEvent(type=EventType.LOG, stage="architecture", content=summary)


# ── Per-file design ───────────────────────────────────────────────────────────

async def _design_one_file(
    target: FileDesign,
    prompt: str,
    prd: str,
    all_files_json: str,
    designed_context: str,
    llm: LLMClient,
    max_retries: int,
) -> FileDesign:
    """Design a single file's API. Returns FileDesign with classes/functions filled in."""

    target_json = json.dumps({
        "name": target.name,
        "path": target.path,
        "description": target.description,
        "dependencies": target.dependencies,
    }, indent=2, ensure_ascii=False)

    p = WRITE_SINGLE_FILE_ARCH_PROMPT.format(
        msg=prompt,
        prd=prd,
        all_files=all_files_json,
        designed_context=designed_context,
        target_file=target_json,
    )

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm.chat(
                [{"role": "user", "content": p}],
                max_tokens=4096,
                json_mode=True,
            )
            data = _extract_json(response)
            return _parse_file_design(data, target)
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries:
                raise ValueError(f"{target.name}: {last_error}") from e


# ── Validation + Reconciliation ────────────────────────────────────────────────

def _validate_architecture(files: list[FileDesign]) -> list[str]:
    """Language-agnostic cross-file consistency checks.

    Checks:
      1. Dependency targets exist
      2. Dependencies have exports (not empty source files)
      3. Cross-file type/symbol references — if A's function params reference
         a type defined in B, A must declare B as a dependency
      4. Near-miss symbol detection — catch naming style mismatches (getUser vs get_user)
      5. Circular dependency detection
      6. Orphan file detection (multi-language entry/config awareness)
      7. Duplicate symbol names within a file
      8. Signature consistency — same function name in different files should have
         compatible signatures if one depends on the other
    """
    issues: list[str] = []

    def _key(f: FileDesign) -> str:
        p = f.path.strip().rstrip("/")
        if p in ("", ".", "./"):
            return f.name
        return f"{p.lstrip('.').lstrip('/')}/{f.name}"

    by_key: dict[str, FileDesign] = {_key(f): f for f in files}
    all_keys = set(by_key)

    # ── Build global symbol registry ──────────────────────────────────────────
    # key → set of exported symbol names
    exports: dict[str, set[str]] = {}
    # symbol_name → list of (key, kind) where it's defined
    global_symbols: dict[str, list[tuple[str, str]]] = {}

    for f in files:
        k = _key(f)
        syms: set[str] = set()
        for cls in f.classes:
            syms.add(cls.class_name)
            global_symbols.setdefault(cls.class_name, []).append((k, "class"))
        for fn in f.functions:
            syms.add(fn.name)
            global_symbols.setdefault(fn.name, []).append((k, "function"))
        exports[k] = syms

    # ── Build dependency graph ────────────────────────────────────────────────
    dep_graph: dict[str, set[str]] = {}
    for f in files:
        k = _key(f)
        deps: set[str] = set()
        for dep_name in f.dependencies:
            resolved = _resolve_dep(dep_name, all_keys)
            if resolved:
                deps.add(resolved)
        dep_graph[k] = deps

    # ── Collect all type references per file (from function params + members) ─
    type_refs: dict[str, set[str]] = {}  # key → set of type names referenced
    for f in files:
        k = _key(f)
        refs: set[str] = set()
        for fn in f.functions:
            for p in fn.input_parameters + fn.output_parameters:
                refs |= _extract_type_names(p.type)
        for cls in f.classes:
            for m in cls.members:
                refs |= _extract_type_names(m.type)
        type_refs[k] = refs

    # ══ CHECK 1: Dependency targets exist ═════════════════════════════════════
    for f in files:
        k = _key(f)
        for dep_name in f.dependencies:
            if not _resolve_dep(dep_name, all_keys):
                issues.append(f"{k}: depends on '{dep_name}' which doesn't exist")

    # ══ CHECK 2: Dependencies have exports ════════════════════════════════════
    # Only flag if:
    #   1. The dependency is a source code file (not config/asset/style/template)
    #   2. The dependency doesn't match common non-export patterns (config, views, etc.)
    #   3. The DEPENDENT file actually references types that could come from this dep
    for f in files:
        k = _key(f)
        my_refs = type_refs.get(k, set())
        for dep_name in f.dependencies:
            dep_key = _resolve_dep(dep_name, all_keys)
            if not dep_key:
                continue
            dep_file = by_key.get(dep_key)
            if not dep_file:
                continue
            _, dep_ext = os.path.splitext(dep_file.name)
            # Non-source files never need exports
            if dep_ext.lower() not in _SOURCE_CODE_EXTS:
                continue
            # Special files never need exports
            if dep_file.name.lower() in _SPECIAL_FILES:
                continue
            # Common non-export patterns (config files, views, templates, etc.)
            if _is_non_export_file(dep_file):
                continue
            # Only flag if the dependency has no exports AND the depender
            # doesn't reference any types from it (i.e., it's purely a side-effect import)
            dep_syms = exports.get(dep_key, set())
            if not dep_syms:
                # Check: does the dependent file reference types that ONLY this dep could provide?
                # If no typed references at all, it's likely a side-effect import → OK
                has_typed_ref_to_dep = False
                for ref in my_refs:
                    if ref.lower() not in _BUILTIN_TYPES and ref not in exports.get(k, set()):
                        # This type ref is unresolved — could have come from this dep
                        has_typed_ref_to_dep = True
                        break
                if has_typed_ref_to_dep:
                    issues.append(
                        f"{k}: depends on '{dep_key}' and uses external types, "
                        f"but '{dep_key}' has no exported symbols — add class/function exports to its design"
                    )

    # ══ CHECK 3: Cross-file type references ═══════════════════════════════════
    # If A references type "User" and "User" is defined in B, A should depend on B
    for f in files:
        k = _key(f)
        my_exports = exports.get(k, set())
        my_deps = dep_graph.get(k, set())
        for type_name in type_refs.get(k, set()):
            # Skip primitive/builtin types
            if type_name.lower() in _BUILTIN_TYPES:
                continue
            # Skip types defined in this file
            if type_name in my_exports:
                continue
            # Find where this type is defined
            definitions = global_symbols.get(type_name, [])
            for def_key, def_kind in definitions:
                if def_key == k:
                    continue
                if def_key not in my_deps:
                    issues.append(
                        f"{k}: references {def_kind} '{type_name}' (defined in {def_key}) "
                        f"but doesn't list it as a dependency"
                    )

    # ══ CHECK 4: Near-miss symbol detection ═══════════════════════════════════
    # Catch naming mismatches like getUser vs get_user across dependent files
    for f in files:
        k = _key(f)
        my_refs = type_refs.get(k, set())
        for dep_key in dep_graph.get(k, set()):
            dep_exports = exports.get(dep_key, set())
            for ref in my_refs:
                if ref.lower() in _BUILTIN_TYPES or ref in exports.get(k, set()):
                    continue
                if ref in dep_exports:
                    continue  # exact match, good
                # Check for near-misses (same name, different case/style)
                for exp in dep_exports:
                    if _is_near_miss(ref, exp):
                        issues.append(
                            f"{k}: references '{ref}' — did you mean '{exp}' from {dep_key}? "
                            f"(naming style mismatch)"
                        )

    # ══ CHECK 5: Circular dependencies ════════════════════════════════════════
    cycles = _find_cycles(dep_graph)
    for cycle in cycles:
        issues.append(f"Circular dependency: {' → '.join(cycle)}")

    # ══ CHECK 6: Orphan detection (multi-language) ════════════════════════════
    all_depended_on: set[str] = set()
    for deps in dep_graph.values():
        all_depended_on |= deps

    for f in files:
        k = _key(f)
        base = f.name.rsplit(".", 1)[0].lower() if "." in f.name else f.name.lower()
        _, ext = os.path.splitext(f.name)

        is_entry = any(p in base for p in _ENTRY_PATTERNS)
        is_config = ext.lower() in _CONFIG_EXTS
        is_special = f.name.lower() in _SPECIAL_FILES
        is_test = any(p in base for p in ("test", "spec", "_test"))
        is_non_export = _is_non_export_file(f)  # views, migrations, config, etc.

        if (k not in all_depended_on and not is_entry and not is_config
                and not is_special and not is_test and not is_non_export
                and not f.dependencies):
            issues.append(f"{k}: orphan �� nothing depends on it and it depends on nothing")

    # ══ CHECK 7: Duplicate symbols within file ════════════════════════════════
    for f in files:
        k = _key(f)
        names = [fn.name for fn in f.functions] + [cls.class_name for cls in f.classes]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                issues.append(f"{k}: duplicate symbol '{name}'")
            seen.add(name)

    # ══ CHECK 8: Signature consistency ════════════════════════════════════════
    # If function foo() appears in both A and B (B depends on A), param counts should match
    for f in files:
        k = _key(f)
        for dep_key in dep_graph.get(k, set()):
            dep_file = by_key.get(dep_key)
            if not dep_file:
                continue
            my_fns = {fn.name: fn for fn in f.functions}
            dep_fns = {fn.name: fn for fn in dep_file.functions}
            for fn_name in set(my_fns) & set(dep_fns):
                my_fn = my_fns[fn_name]
                dep_fn = dep_fns[fn_name]
                if len(my_fn.input_parameters) != len(dep_fn.input_parameters):
                    issues.append(
                        f"Signature mismatch: {k}:{fn_name} has {len(my_fn.input_parameters)} params "
                        f"but {dep_key}:{fn_name} has {len(dep_fn.input_parameters)} params"
                    )

    return issues


# ── Validation helpers ────────────────────────────────────────────────────────

# Multi-language entry point patterns (base name without extension)
_ENTRY_PATTERNS = {
    # General
    "index", "main", "app", "server", "cli", "program", "entry", "page", "route",
    # Python
    "__init__", "__main__", "wsgi", "asgi", "manage",
    # PHP
    "artisan", "bootstrap",
    # Ruby
    "rakefile", "config",
    # Go (main.go detected by "main")
    # Java
    "application",
    # README, docs
    "readme", "license", "changelog",
}

# Source code extensions — ONLY these need class/function exports.
# Everything else (config, assets, styles, templates, docs) is exempt.
_SOURCE_CODE_EXTS = {
    # Server-side
    ".py", ".go", ".java", ".kt", ".scala", ".rs", ".rb", ".cs",
    ".c", ".cpp", ".h", ".hpp", ".swift", ".dart", ".ex", ".exs",
    # Client-side / full-stack
    ".js", ".mjs", ".ts", ".jsx", ".tsx",
    ".vue", ".svelte",
    # PHP — note: PHP config files (config.php) won't have exports
    # and that's OK because we also check _is_php_config() below
    ".php",
}

# Non-source extensions — never need exports (used for dependency check exemption)
_CONFIG_EXTS = {
    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".env", ".ini", ".properties",
    ".conf", ".cfg", ".xml",
    # Docs / text
    ".md", ".txt", ".rst", ".adoc",
    # Styles / assets
    ".css", ".scss", ".sass", ".less", ".styl",
    # Templates / markup
    ".html", ".htm", ".twig", ".blade.php", ".ejs", ".hbs", ".pug", ".jade",
    ".tpl", ".mustache", ".njk",
    # Database
    ".sql", ".prisma",
    # Shell / scripts
    ".sh", ".bash", ".bat", ".cmd", ".ps1",
    # Lock files
    ".lock",
    # Misc config
    ".config", ".gitignore", ".dockerignore", ".editorconfig",
    ".prettierrc", ".eslintrc", ".babelrc", ".npmrc",
}

# Special filenames — always valid (not orphans, don't need exports)
_SPECIAL_FILES = {
    # Docker
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".dockerignore",
    # Build
    "makefile", "rakefile", "gemfile", "procfile", "justfile",
    # Node
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "tsconfig.json", "jsconfig.json",
    "vite.config.ts", "vite.config.js", "vite.config.mjs",
    "webpack.config.js", "webpack.config.ts",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "nuxt.config.ts", "nuxt.config.js",
    "tailwind.config.js", "tailwind.config.ts",
    "postcss.config.js", "postcss.config.cjs",
    "eslint.config.js", ".eslintrc.js", ".eslintrc.json",
    "prettier.config.js", ".prettierrc",
    "babel.config.js", ".babelrc",
    # PHP
    "composer.json", "composer.lock",
    ".htaccess", "php.ini",
    # Python
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "pipfile", "pipfile.lock", "poetry.lock",
    "manage.py", "wsgi.py", "asgi.py",
    # Go
    "go.mod", "go.sum",
    # Rust
    "cargo.toml", "cargo.lock",
    # Java
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    # Ruby
    "gemfile.lock", "config.ru",
    # .NET
    "nuget.config", "global.json",
    # Env
    ".env", ".env.example", ".env.local", ".env.development", ".env.production",
    # Git
    ".gitignore", ".gitattributes",
    # Docs
    "readme.md", "readme.txt", "license", "license.md", "license.txt",
    "changelog.md", "contributing.md",
}

# Builtin/primitive types to ignore in cross-reference checks
_BUILTIN_TYPES = {
    # Common across languages
    "string", "str", "number", "int", "integer", "float", "double", "bool",
    "boolean", "void", "null", "none", "undefined", "any", "object",
    "array", "list", "dict", "map", "set", "tuple", "byte", "bytes",
    "char", "long", "short", "bigint",
    # TypeScript/JavaScript
    "promise", "date", "regexp", "error", "buffer", "json",
    "request", "response", "record", "partial", "readonly", "omit", "pick",
    "react", "reactnode", "jsxelement", "htmlelement", "event",
    # Python
    "optional", "union", "type", "callable", "generator", "iterator",
    "asyncgenerator", "coroutine",
    # Go
    "error", "context", "io", "reader", "writer",
    # PHP
    "mixed", "resource", "callable", "iterable", "self", "static", "parent",
    # Java
    "serializable", "comparable", "runnable", "throwable", "exception",
}


def _extract_type_names(type_str: str) -> set[str]:
    """Extract referenced type names from a type annotation string.

    Handles: 'User', 'List[User]', 'dict[str, User]', 'User | None',
    'Promise<User>', 'Optional[User]', 'Array<Post>', etc.
    """
    import re
    if not type_str:
        return set()
    # Split on delimiters: [], <>, |, ,, space, ()
    tokens = re.split(r'[\[\]<>|,\s\(\)]+', type_str)
    names = set()
    for t in tokens:
        t = t.strip().rstrip("?")  # remove optional marker
        if t and t[0].isupper() and t.lower() not in _BUILTIN_TYPES:
            names.add(t)
    return names


def _is_non_export_file(f: FileDesign) -> bool:
    """Check if a file is expected to have no class/function exports.

    Common patterns across languages:
    - Config files: config.php, settings.py, constants.go, etc.
    - View/template files: in views/, templates/, layouts/, partials/ dirs
    - Entry scripts: migration files, seed files, scripts
    - Static files that happen to have a source extension
    """
    name_lower = f.name.lower()
    path_lower = f.path.lower() if f.path else ""
    base = name_lower.rsplit(".", 1)[0] if "." in name_lower else name_lower

    # Config-like filenames (any language)
    CONFIG_NAMES = {
        "config", "configuration", "settings", "constants", "env",
        "bootstrap", "init", "setup", "seed", "migrate", "migration",
    }
    if base in CONFIG_NAMES:
        return True

    # View/template directories
    VIEW_DIRS = {"views", "templates", "layouts", "partials", "pages",
                 "components", "includes", "snippets", "resources"}
    path_parts = set(path_lower.replace("\\", "/").split("/"))
    if path_parts & VIEW_DIRS:
        return True

    # Migrations, seeds, fixtures
    INFRA_DIRS = {"migrations", "seeds", "seeders", "fixtures", "factories", "database"}
    if path_parts & INFRA_DIRS:
        return True

    # PHP-specific: blade templates
    if name_lower.endswith(".blade.php"):
        return True

    return False


def _is_near_miss(a: str, b: str) -> bool:
    """Check if two symbol names are likely the same thing with different naming style.

    Examples: getUser vs get_user, UserModel vs userModel, fetchData vs FetchData
    """
    if a == b:
        return False
    # Normalize: lowercase + remove underscores/hyphens
    norm_a = a.lower().replace("_", "").replace("-", "")
    norm_b = b.lower().replace("_", "").replace("-", "")
    if norm_a == norm_b:
        return True
    # Check if one is camelCase and the other is snake_case of the same name
    # e.g., getUser → get_user → getuser matches
    return False


def _resolve_dep(dep_name: str, all_keys: set[str]) -> str | None:
    """Resolve a dependency name to a file key."""
    if dep_name in all_keys:
        return dep_name
    for k in all_keys:
        if k.endswith(dep_name) or dep_name in k:
            return k
    return None


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find all cycles in a directed graph using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    path: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str):
        color[node] = GRAY
        path.append(node)
        for nbr in graph.get(node, set()):
            if nbr not in color:
                continue
            if color[nbr] == GRAY:
                # Found cycle: extract it from path
                idx = path.index(nbr)
                cycle = path[idx:] + [nbr]
                cycles.append(cycle)
            elif color[nbr] == WHITE:
                dfs(nbr)
        path.pop()
        color[node] = BLACK

    for node in graph:
        if color[node] == WHITE:
            dfs(node)

    return cycles


RECONCILE_PROMPT = """\
You are a senior software architect fixing cross-file inconsistency issues.

## Issues Found
{issues}

## Full Architecture (current state)
{architecture}

## Original Specification
{msg}

## PRD
{prd}

For each file that has issues, return the CORRECTED version. Only include files
that need changes — do not return unchanged files.

Return ONLY valid JSON:
{{
  "fixes": [
    {{
      "name": "filename.ext",
      "path": "./subdir",
      "description": "updated purpose",
      "dependencies": ["corrected_deps"],
      "classes": [...],
      "functions": [...]
    }}
  ]
}}

Rules:
- Fix the specific issues listed above
- Keep everything else unchanged
- If file A imports X from B, make sure B exports X
- If a dependency doesn't exist, either remove it or add the missing file
- Resolve circular dependencies by restructuring imports
- Function signatures must be consistent between caller and callee
"""


async def _reconcile_architecture(
    designed: list[FileDesign],
    issues: list[str],
    prompt: str,
    prd: str,
    all_files_json: str,
    llm: LLMClient,
) -> list[FileDesign]:
    """Use LLM to fix cross-file inconsistencies."""
    arch_context = _format_designed_context(designed)
    issues_text = "\n".join(f"- {iss}" for iss in issues)

    p = RECONCILE_PROMPT.format(
        issues=issues_text,
        architecture=arch_context,
        msg=prompt,
        prd=prd,
    )

    try:
        response = await llm.chat(
            [{"role": "user", "content": p}],
            max_tokens=8192,
            json_mode=True,
        )
        data = _extract_json(response)
        fixes = data.get("fixes", [])

        if not fixes:
            return designed

        # Build a lookup for quick patching
        def _key(f):
            p = f.path.strip().rstrip("/")
            if p in ("", ".", "./"):
                return f.name
            return f"{p.lstrip('.').lstrip('/')}/{f.name}"

        designed_by_key = {_key(f): f for f in designed}
        patched_count = 0

        for fix_raw in fixes:
            try:
                fixed = _parse_file_design(fix_raw, designed[0:0])  # empty fallback
                fk = _key(fixed)
                # Find the original to patch
                for orig_key in list(designed_by_key):
                    if orig_key == fk or orig_key.endswith(fixed.name) or fixed.name in orig_key:
                        designed_by_key[orig_key] = fixed
                        patched_count += 1
                        break
            except Exception:
                continue

        if patched_count > 0:
            logger.info(f"Reconciliation patched {patched_count} file(s)")
            return list(designed_by_key.values())

    except Exception as e:
        logger.warning(f"Reconciliation LLM call failed: {e}")

    return designed


# ── Topological sort ──────────────────────────────────────────────────────────

def _toposort_layers(files: list[FileDesign]) -> list[list[FileDesign]]:
    """Sort files into dependency layers. Layer 0 has no deps, layer 1 depends
    only on layer 0, etc. Files within a layer can be designed concurrently."""

    # Build name → FileDesign lookup
    by_name: dict[str, FileDesign] = {}
    for f in files:
        key = f"{f.path.strip('./ ')}/{f.name}" if f.path not in ("", ".", "./") else f.name
        key = key.lstrip("/")
        by_name[key] = f
        by_name[f.name] = f  # also by bare name

    # Build adjacency: file_key → set of dependency keys
    def _key(f: FileDesign) -> str:
        p = f.path.strip().rstrip("/")
        if p in ("", ".", "./"):
            return f.name
        return f"{p.lstrip('.').lstrip('/')}/{f.name}"

    file_keys = [_key(f) for f in files]
    key_set = set(file_keys)
    key_to_file = {_key(f): f for f in files}

    deps_map: dict[str, set[str]] = {}
    for f in files:
        k = _key(f)
        resolved_deps: set[str] = set()
        for dep_name in f.dependencies:
            for candidate in key_set:
                if candidate.endswith(dep_name) or dep_name in candidate:
                    resolved_deps.add(candidate)
                    break
        deps_map[k] = resolved_deps

    # Kahn's algorithm for layered toposort
    in_degree = {k: len(deps_map[k]) for k in file_keys}
    remaining = set(file_keys)
    layers: list[list[FileDesign]] = []

    while remaining:
        # Find all nodes with in_degree 0 (or deps outside the set)
        layer_keys = [
            k for k in remaining
            if all(d not in remaining for d in deps_map[k])
        ]
        if not layer_keys:
            # Cycle detected — dump everything remaining into one layer
            layer_keys = sorted(remaining)

        layers.append([key_to_file[k] for k in layer_keys])
        remaining -= set(layer_keys)

    return layers


# ── Context formatting ────────────────────────────────────────────────────────

def _format_designed_context(designed: list[FileDesign]) -> str:
    """Format all designed files as full API signatures for context."""
    if not designed:
        return "(none yet — this is the first layer)"

    parts = []
    for f in designed:
        lines = [f"### {f.path}/{f.name}"]
        if f.description:
            lines.append(f"  # {f.description}")
        for cls in f.classes:
            members = ", ".join(f"{m.name}: {m.type}" for m in cls.members)
            lines.append(f"  class {cls.class_name}({members})")
            for method in cls.methods:
                lines.append(f"    .{method}")
        for fn in f.functions:
            params = ", ".join(f"{p.name}: {p.type}" for p in fn.input_parameters)
            returns = ", ".join(p.type for p in fn.output_parameters) or "None"
            lines.append(f"  def {fn.name}({params}) -> {returns}")
            if fn.description:
                lines.append(f"    # {fn.description}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_file_design(data: dict, fallback: FileDesign) -> FileDesign:
    """Parse a single file's design from JSON. Aggressive type coercion."""
    name = str(data.get("name", fallback.name)).strip()
    if not name:
        name = fallback.name
    path = _norm_path(data.get("path", fallback.path))

    classes = []
    for c in data.get("classes", []):
        if not isinstance(c, dict):
            continue
        try:
            classes.append(ClassSpec(
                class_name=str(c.get("class_name", "Unknown")),
                members=_parse_params(c.get("members", [])),
                methods=[str(m) for m in (c.get("methods", []) or [])],
            ))
        except Exception:
            classes.append(ClassSpec(class_name=str(c.get("class_name", "Unknown"))))

    functions = []
    for f in data.get("functions", []):
        if not isinstance(f, dict):
            continue
        try:
            functions.append(FunctionSpec(
                name=str(f.get("name", "unknown")),
                input_parameters=_parse_params(f.get("input_parameters", [])),
                output_parameters=_parse_params(f.get("output_parameters", [])),
                description=str(f.get("description", "")),
            ))
        except Exception:
            functions.append(FunctionSpec(name=str(f.get("name", "unknown"))))

    deps = data.get("dependencies", fallback.dependencies)
    if not isinstance(deps, list):
        deps = [str(deps)] if deps else []
    deps = [str(d) for d in deps] or fallback.dependencies

    return FileDesign(
        name=name, path=path,
        description=str(data.get("description", fallback.description)),
        classes=classes, functions=functions, dependencies=deps,
    )


def _parse_params(raw) -> list[Parameter]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            Parameter(name=str(k),
                      type=str(v) if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for k, v in raw.items()
        ]
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(Parameter(name=item, type="any"))
        elif isinstance(item, dict):
            raw_type = item.get("type", "any")
            if isinstance(raw_type, (dict, list)):
                raw_type = json.dumps(raw_type, ensure_ascii=False)
            result.append(Parameter(
                name=str(item.get("name", "")),
                type=str(raw_type) if raw_type is not None else "any",
            ))
    return result


def _extract_json(text: str) -> dict:
    # First try direct parse (json_mode often returns clean JSON)
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                    break
    raise ValueError(f"Could not extract JSON (len={len(text)})")


def _norm_path(path: str) -> str:
    """Normalize a relative path to './subdir' format. Collapses repeated ./ prefixes."""
    path = str(path or "").strip().replace("\\", "/")
    # Strip all leading ./ sequences
    while path.startswith("./"):
        path = path[2:]
    # Strip leading/trailing slashes
    path = path.strip("/")
    # Block path traversal
    if ".." in path:
        path = path.replace("..", "").strip("/")
    if not path or path == ".":
        return "./"
    return f"./{path}"
