"""Base interfaces for PantryPal tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

ToolInput = dict[str, Any]
ToolOutput = dict[str, Any]


class Tool(ABC):
    """Abstract base class for agent tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, input_data: ToolInput) -> ToolOutput:
        """Run the tool with structured input and return structured output."""
