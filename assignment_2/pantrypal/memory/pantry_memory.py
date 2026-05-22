"""JSON-backed pantry memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PantryMemory:
    """Persistent store for ingredients available to the user."""

    path: Path
    items: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "PantryMemory":
        """Load pantry memory from disk, creating an empty memory if missing."""

        if not path.exists():
            return cls(path=path)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(path=path, items={str(item).lower() for item in data.get("items", [])})

    def add(self, ingredients: list[str]) -> None:
        """Add ingredients to memory and persist the change."""

        self.items.update(item.strip().lower() for item in ingredients if item.strip())
        self.save()

    def remove(self, ingredients: list[str]) -> None:
        """Remove ingredients from memory and persist the change."""

        for ingredient in ingredients:
            self.items.discard(ingredient.strip().lower())
        self.save()

    def list_items(self) -> list[str]:
        """Return pantry items in stable sorted order."""

        return sorted(self.items)

    def save(self) -> None:
        """Persist pantry memory to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump({"items": self.list_items()}, file, indent=2)
