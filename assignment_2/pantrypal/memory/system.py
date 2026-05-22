"""High-level memory loader for PantryPal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pantrypal.memory.pantry_memory import PantryMemory
from pantrypal.memory.preferences_memory import PreferencesMemory


@dataclass
class MemorySystem:
    """Container for all persistent PantryPal memory stores."""

    pantry: PantryMemory
    preferences: PreferencesMemory

    @classmethod
    def load(cls, memory_dir: Path) -> "MemorySystem":
        """Load every memory store from the configured memory directory."""

        return cls(
            pantry=PantryMemory.load(memory_dir / "pantry_memory.json"),
            preferences=PreferencesMemory.load(memory_dir / "preferences_memory.json"),
        )

    def save(self) -> None:
        """Persist all memory stores."""

        self.pantry.save()
        self.preferences.save()
