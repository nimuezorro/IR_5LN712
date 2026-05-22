"""Tool for ingredient substitution retrieval."""

from __future__ import annotations

import re

from pantrypal.retrieval.substitution_retriever import SubstitutionRetriever
from pantrypal.tools.base import Tool, ToolInput, ToolOutput


class SubstitutionTool(Tool):
    """Retrieve substitutions for missing ingredients."""

    name = "substitution"
    description = "Finds local ingredient substitutions."

    def __init__(self, retriever: SubstitutionRetriever) -> None:
        """Create a substitution tool."""

        self.retriever = retriever

    def run(self, input_data: ToolInput) -> ToolOutput:
        """Search substitutions and format compact results."""

        query = str(input_data.get("query", ""))
        missing = self._get_missing_ingredients(input_data, query)
        available = [str(item) for item in input_data.get("available", [])]
        if not available:
            available = [str(item) for item in input_data.get("pantry_items", [])]
        limit = int(input_data.get("limit", 5))
        results = self.retriever.search_many(missing, available=available, limit=limit)
        if not results:
            return {
                "tool_name": self.name,
                "content": "No substitution matches found.",
                "results": [],
                "source_type": "local",
                "confidence": 0.0,
            }

        lines = []
        metadata = []
        for result in results:
            best = result.best_substitute
            if best is None:
                continue
            alternatives = ", ".join(item.name for item in result.alternatives) or "none"
            lines.append(f"- {result.ingredient}: best {best.name} ({best.confidence:.2f})")
            lines.append(f"  Alternatives: {alternatives}")
            lines.append(f"  Tradeoff: {best.explanation}")
            metadata.append(
                {
                    "ingredient": result.ingredient,
                    "best_substitute": self._candidate_to_dict(best),
                    "alternative_substitutes": [
                        self._candidate_to_dict(candidate)
                        for candidate in result.alternatives
                    ],
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "source": "local",
                }
            )
        confidence = max((item["confidence"] for item in metadata), default=0.0)
        return {
            "tool_name": self.name,
            "content": "\n".join(lines),
            "results": metadata,
            "source_type": "local",
            "confidence": confidence,
        }

    def _get_missing_ingredients(self, input_data: ToolInput, query: str) -> list[str]:
        missing_value = input_data.get("missing")
        if isinstance(missing_value, str):
            return [missing_value]
        if isinstance(missing_value, list):
            return [str(item) for item in missing_value]
        extracted = self._extract_missing_from_query(query)
        return extracted or [query]

    def _extract_missing_from_query(self, query: str) -> list[str]:
        lowered = query.lower().strip()
        patterns = [
            r"(?:substitute|replacement for|instead of|replace) (?P<ingredient>.+)",
            r"what can i use for (?P<ingredient>.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return [self._clean_ingredient(match.group("ingredient"))]
        return []

    def _clean_ingredient(self, value: str) -> str:
        return re.sub(r"[?!.]", "", value).strip()

    def _candidate_to_dict(self, candidate: object) -> dict[str, object]:
        return {
            "name": getattr(candidate, "name"),
            "score": getattr(candidate, "score"),
            "confidence": getattr(candidate, "confidence"),
            "in_pantry": getattr(candidate, "in_pantry"),
            "flavor_similarity": getattr(candidate, "flavor_similarity"),
            "cooking_use_notes": getattr(candidate, "cooking_use_notes"),
            "explanation": getattr(candidate, "explanation"),
        }
