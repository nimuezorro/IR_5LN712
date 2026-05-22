"""Tool for managing dietary and taste preferences."""

from __future__ import annotations

import re

from pantrypal.memory.preferences_memory import PreferencesMemory
from pantrypal.tools.base import Tool, ToolInput, ToolOutput


class PreferencesTool(Tool):
    """Store and list dietary preferences, dislikes, and favorite cuisines."""

    name = "preferences"
    description = "Tracks dietary preferences, disliked ingredients, and favorite cuisines."

    dietary_terms = {
        "vegetarian",
        "vegan",
        "gluten-free",
        "gluten free",
        "dairy-free",
        "dairy free",
        "halal",
        "kosher",
        "low carb",
    }

    def __init__(self, memory: PreferencesMemory) -> None:
        """Create a preferences tool from persistent preferences memory."""

        self.memory = memory

    def run(self, input_data: ToolInput) -> ToolOutput:
        """Interpret a structured preferences command."""

        query = str(input_data.get("query", "")).strip()
        lower = query.lower()
        content = self._handle_query(query, lower)
        return {
            "tool_name": self.name,
            "content": content,
            "preferences": self.memory.to_dict(),
            "source_type": "memory",
        }

    def _handle_query(self, query: str, lower: str) -> str:
        if self._is_listing_query(lower):
            return self._format_preferences()

        disliked = self._extract_dislikes(query)
        if disliked:
            self.memory.add_disliked_ingredients(disliked)
            return f"Stored disliked ingredients: {', '.join(disliked)}"

        cuisines = self._extract_favorite_cuisines(query)
        if cuisines:
            self.memory.add_favorite_cuisines(cuisines)
            return f"Stored favorite cuisines: {', '.join(cuisines)}"

        dietary = self._extract_dietary_preferences(query)
        if dietary:
            self.memory.add_dietary_preferences(dietary)
            return f"Stored dietary preferences: {', '.join(dietary)}"

        return self._format_preferences()

    def _extract_dislikes(self, text: str) -> list[str]:
        patterns = [
            r"\bi hate (?P<items>.+)",
            r"\bi dislike (?P<items>.+)",
            r"\bi don't like (?P<items>.+)",
            r"\bavoid (?P<items>.+)",
        ]
        return self._extract_items(text, patterns)

    def _extract_favorite_cuisines(self, text: str) -> list[str]:
        patterns = [
            r"\bfavorite cuisines? (?:are|is) (?P<items>.+)",
            r"\bi love (?P<items>.+) cuisine",
            r"\bi like (?P<items>.+) cuisine",
        ]
        return self._extract_items(text, patterns)

    def _extract_dietary_preferences(self, text: str) -> list[str]:
        lower = text.lower()
        found = sorted(term for term in self.dietary_terms if term in lower)
        if found:
            return found
        patterns = [
            r"\bi am (?P<items>.+)",
            r"\bi'm (?P<items>.+)",
            r"\bmy diet is (?P<items>.+)",
        ]
        extracted = self._extract_items(text, patterns)
        return [item for item in extracted if item in self.dietary_terms]

    def _extract_items(self, text: str, patterns: list[str]) -> list[str]:
        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return self._split_items(match.group("items"))
        return []

    def _split_items(self, text: str) -> list[str]:
        cleaned = re.sub(r"[.!?]", "", text)
        cleaned = re.sub(r"\bcuisine\b", "", cleaned)
        parts = re.split(r",|\band\b", cleaned, flags=re.IGNORECASE)
        return [item.strip().lower() for item in parts if item.strip()]

    def _format_preferences(self) -> str:
        dietary = self.memory.list_dietary_preferences()
        disliked = self.memory.list_disliked_ingredients()
        cuisines = self.memory.list_favorite_cuisines()
        if not dietary and not disliked and not cuisines:
            return "No preferences stored yet."
        lines = [
            "Preferences:",
            f"- Dietary: {', '.join(dietary) if dietary else 'none'}",
            f"- Disliked ingredients: {', '.join(disliked) if disliked else 'none'}",
            f"- Favorite cuisines: {', '.join(cuisines) if cuisines else 'none'}",
        ]
        return "\n".join(lines)

    def _is_listing_query(self, text: str) -> bool:
        return any(phrase in text for phrase in ["list preferences", "show preferences"])
