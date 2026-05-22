"""Rule-based starter planner for choosing PantryPal tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """A simple tool execution plan."""

    tool_names: list[str]
    reason: str
    needs_substitutions: bool


class Planner:
    """Select tools using transparent keyword heuristics."""

    known_cuisines = {
        "chinese",
        "chinese-inspired",
        "greek",
        "indian",
        "indian-inspired",
        "italian",
        "mediterranean",
    }

    def plan(self, user_message: str) -> Plan:
        """Return a starter plan for the user message."""

        text = user_message.lower()
        if any(
            phrase in text
            for phrase in [
                "i hate",
                "i dislike",
                "i don't like",
                "avoid ",
                "favorite cuisine",
                "favourite cuisine",
                "i love",
                "i am vegan",
                "i'm vegan",
                "i am vegetarian",
                "i'm vegetarian",
                "gluten-free",
                "gluten free",
                "list preferences",
                "show preferences",
            ]
        ):
            return Plan(["preferences"], "The user is updating or listing preferences.", False)
        if text.startswith(("pantry", "add ", "remove ", "list pantry")) or any(
            phrase in text
            for phrase in ["i usually keep", "i keep", "i have", "out of", "no more"]
        ):
            return Plan(["pantry"], "The user is managing pantry memory.", False)
        if any(phrase in text for phrase in ["technique", "how do i", "how to", "method"]):
            return Plan(["web_search"], "The user is asking for a cooking technique.", False)
        if any(word in text for word in ["substitute", "replacement", "instead of"]):
            return Plan(
                ["pantry", "preferences", "recipe_search", "substitution"],
                "The user needs ingredient alternatives.",
                True,
            )
        if any(word in text for word in ["recipe", "cook", "make", "dinner", "meal"]):
            if self.asks_for_unfamiliar_cuisine(text):
                return Plan(
                    ["pantry", "preferences", "recipe_search", "web_search"],
                    "The user asks for a cuisine outside the local corpus.",
                    False,
                )
            return Plan(
                ["pantry", "preferences", "recipe_search"],
                "The user is asking for cooking suggestions.",
                False,
            )
        return Plan(
            ["pantry", "preferences", "recipe_search", "substitution"],
            "Broad query; try local retrieval first.",
            True,
        )

    def asks_for_unfamiliar_cuisine(self, text: str) -> bool:
        """Return whether the user appears to ask for an unknown cuisine."""

        for marker in [" cuisine", " food", " dish", " recipe"]:
            if marker not in text:
                continue
            before_marker = text.split(marker, maxsplit=1)[0].split()
            if not before_marker:
                continue
            candidate = before_marker[-1].strip(" .?!,")
            if candidate and candidate not in self.known_cuisines:
                return True
        return False
