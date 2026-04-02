"""PromptBuilder — modular, chainable prompt assembly.

Instead of format() strings scattered everywhere, build prompts as named sections.
"""

from __future__ import annotations


class PromptBuilder:
    """Chainable builder for multi-section prompts."""

    def __init__(self):
        self._parts: list[str] = []

    def text(self, content: str) -> "PromptBuilder":
        if content and content.strip():
            self._parts.append(content.strip())
        return self

    def section(self, title: str, content: str) -> "PromptBuilder":
        if content and content.strip():
            self._parts.append(f"## {title}\n{content.strip()}")
        return self

    def section_if(self, condition: bool, title: str, content: str) -> "PromptBuilder":
        if condition:
            return self.section(title, content)
        return self

    def build(self) -> str:
        return "\n\n".join(self._parts)

    def __str__(self) -> str:
        return self.build()
