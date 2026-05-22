"""JSON-backed preference memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreferencesMemory:
    """Persistent store for user food preferences.

    The model keeps separate typed collections so the agent can later reason
    differently about dietary constraints, disliked ingredients, and cuisines.
    """

    path: Path
    dietary_preferences: set[str] = field(default_factory=set)
    disliked_ingredients: set[str] = field(default_factory=set)
    favorite_cuisines: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "PreferencesMemory":
        """Load preferences from disk, creating an empty memory if missing."""

        if not path.exists():
            return cls(path=path)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(
            path=path,
            dietary_preferences=cls._normalize_many(data.get("dietary_preferences", [])),
            disliked_ingredients=cls._normalize_many(data.get("disliked_ingredients", [])),
            favorite_cuisines=cls._normalize_many(data.get("favorite_cuisines", [])),
        )

    def add_dietary_preferences(self, values: list[str]) -> None:
        """Add dietary preferences and persist them."""

        self.dietary_preferences.update(self._normalize_many(values))
        self.save()

    def add_disliked_ingredients(self, values: list[str]) -> None:
        """Add disliked ingredients and persist them."""

        self.disliked_ingredients.update(self._normalize_many(values))
        self.save()

    def add_favorite_cuisines(self, values: list[str]) -> None:
        """Add favorite cuisines and persist them."""

        self.favorite_cuisines.update(self._normalize_many(values))
        self.save()

    def remove_disliked_ingredients(self, values: list[str]) -> None:
        """Remove disliked ingredients and persist the change."""

        for value in self._normalize_many(values):
            self.disliked_ingredients.discard(value)
        self.save()

    def list_dietary_preferences(self) -> list[str]:
        """Return dietary preferences in stable sorted order."""

        return sorted(self.dietary_preferences)

    def list_disliked_ingredients(self) -> list[str]:
        """Return disliked ingredients in stable sorted order."""

        return sorted(self.disliked_ingredients)

    def list_favorite_cuisines(self) -> list[str]:
        """Return favorite cuisines in stable sorted order."""

        return sorted(self.favorite_cuisines)

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize preferences as JSON-compatible data."""

        return {
            "dietary_preferences": self.list_dietary_preferences(),
            "disliked_ingredients": self.list_disliked_ingredients(),
            "favorite_cuisines": self.list_favorite_cuisines(),
        }

    def save(self) -> None:
        """Persist preferences to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @staticmethod
    def _normalize_many(values: object) -> set[str]:
        """Normalize a collection of preference strings."""

        if not isinstance(values, (list, set, tuple)):
            return set()
        return {str(value).strip().lower() for value in values if str(value).strip()}
