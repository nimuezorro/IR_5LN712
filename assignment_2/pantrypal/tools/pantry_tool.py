"""Tool for managing pantry memory."""

from __future__ import annotations

import re

from pantrypal.memory.pantry_memory import PantryMemory
from pantrypal.tools.base import Tool, ToolInput, ToolOutput


class PantryTool(Tool):
    """Add, remove, and list pantry ingredients."""

    name = "pantry"
    description = "Tracks ingredients the user has available."

    def __init__(self, memory: PantryMemory) -> None:
        """Create a pantry tool from a pantry memory instance."""

        self.memory = memory

    def run(self, input_data: ToolInput) -> ToolOutput:
        """Interpret a structured pantry command."""

        query = str(input_data.get("query", ""))
        normalized = query.strip()
        lower = normalized.lower()
        if lower.startswith("add "):
            items = self._split_items(normalized[4:])
            self.memory.add(items)
            content = f"Added: {', '.join(items)}"
        elif lower.startswith("remove "):
            items = self._split_items(normalized[7:])
            self.memory.remove(items)
            content = f"Removed: {', '.join(items)}"
        elif self._looks_like_pantry_add(lower):
            items = self._extract_pantry_items(normalized)
            self.memory.add(items)
            content = f"Added: {', '.join(items)}"
        elif self._looks_like_pantry_remove(lower):
            items = self._extract_pantry_items(normalized)
            self.memory.remove(items)
            content = f"Removed: {', '.join(items)}"
        else:
            items = self.memory.list_items()
            content = "Pantry is empty." if not items else "Pantry: " + ", ".join(items)
        return {
            "tool_name": self.name,
            "content": content,
            "items": self.memory.list_items(),
            "source_type": "memory",
        }

    def _split_items(self, text: str) -> list[str]:
        cleaned = re.sub(r"[.!?]", "", text)
        parts = re.split(r",|\band\b", cleaned, flags=re.IGNORECASE)
        return [item.strip().lower() for item in parts if item.strip()]

    def _looks_like_pantry_add(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "i usually keep",
                "i keep",
                "i have",
                "in my pantry",
                "stocked with",
            ]
        )

    def _looks_like_pantry_remove(self, text: str) -> bool:
        return any(phrase in text for phrase in ["out of", "no more", "used up"])

    def _extract_pantry_items(self, text: str) -> list[str]:
        lowered = text.lower()
        patterns = [
            r"i usually keep (?P<items>.+)",
            r"i keep (?P<items>.+)",
            r"i have (?P<items>.+)",
            r"in my pantry (?:i have|there is|there are)? (?P<items>.+)",
            r"stocked with (?P<items>.+)",
            r"(?:out of|no more|used up) (?P<items>.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return self._split_items(match.group("items"))
        return self._split_items(text)
