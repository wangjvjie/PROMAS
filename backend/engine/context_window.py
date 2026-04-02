"""ContextWindow — token budget tracking and message compaction.

Compaction strategy (from claude-code-main's snip pattern):
  - Keep: system message + initial user message (head)
  - Keep: last KEEP_TAIL messages (recent context)
  - Compress: everything in between
  - Preserve: file-read tool results (needed for cross-file consistency)
  - Preserve: successfully written file contents (so agent doesn't need re-read)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("promas")

PRUNE_THRESHOLD = 0.88    # compact when prompt_tokens > max_tokens * this (was 0.82)
KEEP_TAIL = 12            # always keep this many recent messages (was 10)
MAX_PRESERVED_READS = 6   # max file-read results to keep in compressed summary


@dataclass
class ContextWindow:
    max_tokens: int = 128000
    _last_prompt_tokens: int = 0
    # Cache of written file contents — survives compaction
    _written_cache: dict[str, str] = field(default_factory=dict)

    def track(self, prompt_tokens: int, completion_tokens: int = 0):
        """Update with actual token counts from the API response."""
        self._last_prompt_tokens = prompt_tokens

    def record_written_file(self, file_key: str, content: str):
        """Remember a file's content so the agent doesn't need to re-read it.
        Called by the agent when finish() succeeds."""
        self._written_cache[file_key] = content

    def should_compact(self, messages: list[dict]) -> bool:
        if self._last_prompt_tokens > 0:
            return self._last_prompt_tokens > int(self.max_tokens * PRUNE_THRESHOLD)
        # Fallback: estimate from content length (~3.5 chars/token for code)
        estimated = sum(len(str(m.get("content") or "")) // 3 for m in messages)
        return estimated > int(self.max_tokens * PRUNE_THRESHOLD)

    def maybe_compact(self, messages: list[dict]) -> list[dict]:
        if not self.should_compact(messages):
            return messages
        if len(messages) <= KEEP_TAIL + 2:
            return messages
        return self._compact(messages)

    def _compact(self, messages: list[dict]) -> list[dict]:
        logger.info(
            f"Compacting context: {len(messages)} messages, "
            f"~{self._last_prompt_tokens} prompt tokens"
        )
        head = messages[:2]         # system + first user
        tail = messages[-KEEP_TAIL:]
        middle = messages[2:-KEEP_TAIL]

        preserved_reads: list[str] = []
        preserved_writes: list[str] = []
        summary_lines: list[str] = []

        for i, msg in enumerate(middle):
            role = msg.get("role", "")
            content = msg.get("content") or ""

            # Identify file-read tool results — preserve them
            if role == "tool" and len(content) > 300:
                prev = middle[i - 1] if i > 0 else {}
                tool_name = ""
                tool_args = {}
                if prev.get("tool_calls"):
                    tc = prev["tool_calls"][0]
                    tool_name = tc.get("function", {}).get("name", "")
                    try:
                        import json
                        tool_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except Exception:
                        pass

                if tool_name in ("read_file", "read_architecture", "search_code"):
                    label = tool_args.get("path") or tool_args.get("file_name") or tool_name
                    preserved_reads.append(f"[{tool_name}: {label}]\n{content}")
                    continue

            # Identify successful finish() results — preserve written file info
            if role == "tool" and ("File created successfully" in content or
                                    "has been updated successfully" in content):
                preserved_writes.append(content)
                continue

            # Summarise everything else
            if role == "assistant" and msg.get("tool_calls"):
                names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                summary_lines.append(f"→ called: {', '.join(names)}")
            elif role == "tool":
                summary_lines.append(f"  result: {content[:120]}...")
            elif role == "user" and content:
                summary_lines.append(f"[prompt]: {content[:120]}...")
            elif role == "assistant" and content:
                summary_lines.append(f"[think]: {content[:120]}...")

        parts = ["[Earlier conversation — compressed]"]
        if summary_lines:
            parts.append("\n".join(summary_lines[-24:]))

        # Inject written files cache so agent knows what it already wrote
        if self._written_cache:
            cache_summary = "\n".join(
                f"  ✓ {k} ({len(v)} chars, {len(v.splitlines())} lines)"
                for k, v in sorted(self._written_cache.items())
            )
            parts.append(f"[Files written in this session — use read_file if you need exact content]\n{cache_summary}")

        if preserved_writes:
            parts.append("[Write results]\n" + "\n".join(preserved_writes[-8:]))

        if preserved_reads:
            parts.append("[Key file reads preserved for cross-file consistency]")
            parts.extend(preserved_reads[-MAX_PRESERVED_READS:])

        compact_msg = {"role": "user", "content": "\n\n".join(parts)}
        result = head + [compact_msg] + tail
        logger.info(f"Compacted {len(messages)} → {len(result)} messages "
                     f"(preserved {len(preserved_reads)} reads, {len(preserved_writes)} writes, "
                     f"{len(self._written_cache)} cached files)")
        return result
