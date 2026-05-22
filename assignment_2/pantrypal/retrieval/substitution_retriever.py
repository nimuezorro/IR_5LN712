"""Ingredient substitution retrieval over a local JSON corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pantrypal.retrieval.recipe_retriever import normalize_items, normalize_text
from pantrypal.retrieval.ranking import lexical_score, tokenize


@dataclass(frozen=True)
class SubstituteCandidate:
    """One substitute option for an ingredient."""

    name: str
    flavor_similarity: str
    cooking_use_notes: str

    @classmethod
    def from_raw(cls, raw: object, default_notes: str = "") -> "SubstituteCandidate":
        """Build a candidate from either new structured or legacy string data."""

        if isinstance(raw, dict):
            return cls(
                name=str(raw["name"]),
                flavor_similarity=str(raw.get("flavor_similarity", "unknown")),
                cooking_use_notes=str(raw.get("cooking_use_notes", default_notes)),
            )
        return cls(
            name=str(raw),
            flavor_similarity="unknown",
            cooking_use_notes=default_notes,
        )


@dataclass(frozen=True)
class SubstitutionEntry:
    """Substitution entry for a missing ingredient."""

    ingredient: str
    substitutes: list[SubstituteCandidate]
    cooking_use_notes: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubstitutionEntry":
        """Build a substitution entry from JSON data."""

        notes = str(data.get("cooking_use_notes", data.get("notes", "")))
        return cls(
            ingredient=str(data["ingredient"]),
            substitutes=[
                SubstituteCandidate.from_raw(item, default_notes=notes)
                for item in data.get("substitutes", [])
            ],
            cooking_use_notes=notes,
        )

    def searchable_terms(self) -> set[str]:
        """Return terms used for lexical entry retrieval."""

        text = " ".join(
            [
                self.ingredient,
                self.cooking_use_notes,
                *[
                    f"{candidate.name} {candidate.flavor_similarity} {candidate.cooking_use_notes}"
                    for candidate in self.substitutes
                ],
            ]
        )
        return tokenize(text)


@dataclass(frozen=True)
class RankedSubstitute:
    """Ranked substitute with evidence and tradeoff explanation."""

    name: str
    score: float
    confidence: float
    in_pantry: bool
    flavor_similarity: str
    cooking_use_notes: str
    explanation: str


@dataclass(frozen=True)
class SubstitutionSearchResult:
    """Structured retrieval result for one missing ingredient."""

    ingredient: str
    best_substitute: RankedSubstitute | None
    alternatives: list[RankedSubstitute]
    confidence: float
    evidence: dict[str, object]


class SubstitutionRetriever:
    """Retrieve and rank substitutions from a local JSON file."""

    def __init__(self, substitutions_path: Path) -> None:
        """Initialize the retriever with a substitutions corpus path."""

        self.substitutions_path = substitutions_path
        self._substitutions = self._load_substitutions()

    def search(
        self,
        missing: str,
        available: list[str] | None = None,
        limit: int = 5,
    ) -> SubstitutionSearchResult | None:
        """Retrieve ranked substitutes for a missing ingredient."""

        entry, entry_score = self._find_entry(missing)
        if entry is None:
            return None

        available_items = normalize_items(available or [])
        ranked = [
            self._rank_candidate(candidate, available_items, entry_score)
            for candidate in entry.substitutes
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        top = ranked[:limit]
        best = top[0] if top else None
        return SubstitutionSearchResult(
            ingredient=entry.ingredient,
            best_substitute=best,
            alternatives=top[1:],
            confidence=best.confidence if best else 0.0,
            evidence={
                "matched_entry": entry.ingredient,
                "entry_score": round(entry_score, 4),
                "available": available_items,
                "cooking_use_notes": entry.cooking_use_notes,
            },
        )

    def search_many(
        self,
        missing: list[str],
        available: list[str] | None = None,
        limit: int = 5,
    ) -> list[SubstitutionSearchResult]:
        """Retrieve substitutions for multiple missing ingredients."""

        results = []
        for ingredient in missing:
            result = self.search(ingredient, available=available, limit=limit)
            if result is not None:
                results.append(result)
        return results

    def _find_entry(self, missing: str) -> tuple[SubstitutionEntry | None, float]:
        query = normalize_text(missing)
        exact = [
            entry
            for entry in self._substitutions
            if query == normalize_text(entry.ingredient)
            or query in normalize_text(entry.ingredient)
            or normalize_text(entry.ingredient) in query
        ]
        if exact:
            return exact[0], 1.0

        ranked = [
            (entry, lexical_score(query, entry.searchable_terms()))
            for entry in self._substitutions
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        best_entry, best_score = ranked[0] if ranked else (None, 0.0)
        if best_score <= 0:
            return None, 0.0
        return best_entry, best_score

    def _rank_candidate(
        self,
        candidate: SubstituteCandidate,
        available: list[str],
        entry_score: float,
    ) -> RankedSubstitute:
        candidate_name = normalize_text(candidate.name)
        in_pantry = any(
            item == candidate_name or item in candidate_name or candidate_name in item
            for item in available
        )
        flavor_score = flavor_similarity_score(candidate.flavor_similarity)
        pantry_boost = 0.35 if in_pantry else 0.0
        score = min((0.45 * entry_score) + (0.35 * flavor_score) + pantry_boost, 1.0)
        confidence = round(score, 4)
        explanation = build_tradeoff_explanation(candidate, in_pantry)
        return RankedSubstitute(
            name=candidate.name,
            score=round(score, 4),
            confidence=confidence,
            in_pantry=in_pantry,
            flavor_similarity=candidate.flavor_similarity,
            cooking_use_notes=candidate.cooking_use_notes,
            explanation=explanation,
        )

    def _load_substitutions(self) -> list[SubstitutionEntry]:
        if not self.substitutions_path.exists():
            return []
        with self.substitutions_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return [SubstitutionEntry.from_dict(item) for item in data]


def flavor_similarity_score(value: str) -> float:
    """Map flavor similarity labels to numeric ranking scores."""

    normalized = normalize_text(value)
    if normalized in {"high", "very high", "close"}:
        return 1.0
    if normalized in {"medium", "moderate"}:
        return 0.7
    if normalized in {"low"}:
        return 0.4
    return 0.55


def build_tradeoff_explanation(candidate: SubstituteCandidate, in_pantry: bool) -> str:
    """Explain why a substitute was ranked and what tradeoff it has."""

    availability = "available in pantry" if in_pantry else "not currently in pantry"
    return (
        f"{candidate.name} has {candidate.flavor_similarity} flavor similarity and is "
        f"{availability}. {candidate.cooking_use_notes}"
    )
