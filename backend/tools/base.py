"""Tool base class.

Each tool:
  - Has a name, description, input_schema (JSON Schema)
  - is_read_only: True = safe to run concurrently with other read-only tools
  - execute(**kwargs) -> str: runs the tool, returns observation string
  - to_schema() -> dict: returns OpenAI function-calling schema
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    name: str = ""
    description: str = ""
    is_read_only: bool = True

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        ...

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
