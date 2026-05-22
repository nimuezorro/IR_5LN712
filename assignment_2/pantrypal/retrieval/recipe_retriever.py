"""Local BM25-style recipe retrieval over a JSON recipe corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pantrypal.retrieval.base import RetrievalDiagnostics
from pantrypal.retrieval.ranking import tokenize

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "can",
    "cook",
    "for",
    "have",
    "i",
    "in",
    "make",
    "meal",
    "recipe",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class Recipe:
    """Recipe document stored in the local recipe corpus."""

    id: str
    title: str
    ingredients: list[str]
    instructions: list[str]
    cuisine: str
    dietary_tags: list[str]
    prep_time: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        """Build a recipe from JSON data, accepting legacy starter fields."""

        title = str(data["title"])
        return cls(
            id=str(data.get("id", slugify(title))),
            title=title,
            ingredients=[str(item) for item in data.get("ingredients", [])],
            instructions=[str(item) for item in data.get("instructions", data.get("steps", []))],
            cuisine=str(data.get("cuisine", "unspecified")),
            dietary_tags=[str(item).lower() for item in data.get("dietary_tags", data.get("tags", []))],
            prep_time=str(data.get("prep_time", "unknown")),
        )

    def searchable_text(self) -> str:
        """Return the text indexed for BM25 ranking."""

        return " ".join(
            [
                self.title,
                self.cuisine,
                self.prep_time,
                *self.ingredients,
                *self.instructions,
                *self.dietary_tags,
            ]
        )


@dataclass(frozen=True)
class RecipeSearchResult:
    """Ranked recipe result with retrieval metadata."""

    recipe: Recipe
    score: float
    matched_ingredients: list[str]
    missing_ingredients: list[str]


@dataclass(frozen=True)
class RecipeSearchResponse:
    """Complete recipe retrieval response."""

    results: list[RecipeSearchResult]
    confidence: float
    use_web_fallback: bool
    threshold: float
    diagnostics: RetrievalDiagnostics


class RecipeRetriever:
    """Retrieve recipes from a local JSON corpus using BM25-style scoring."""

    def __init__(
        self,
        recipes_path: Path,
        confidence_threshold: float = 0.35,
        min_results: int = 3,
    ) -> None:
        """Initialize the retriever with a recipe corpus and fallback policy."""

        self.recipes_path = recipes_path
        self.confidence_threshold = confidence_threshold
        self.min_results = min_results
        self._recipes = self._load_recipes()
        self._documents = [tokenize(recipe.searchable_text()) for recipe in self._recipes]
        self._term_frequencies = [
            Counter(tokenize(recipe.searchable_text())) for recipe in self._recipes
        ]
        self._doc_freq = self._build_document_frequencies()
        self._avg_doc_len = self._average_document_length()

    def search(
        self,
        query: str,
        limit: int = 5,
        pantry_ingredients: list[str] | None = None,
        dietary_preferences: list[str] | None = None,
    ) -> RecipeSearchResponse:
        """Return top-k recipes with confidence and web-fallback signal."""

        pantry = normalize_items(pantry_ingredients or [])
        dietary = normalize_items(dietary_preferences or [])
        query_terms = meaningful_terms(query)
        candidate_ingredients = sorted(meaningful_terms(query) | set(pantry))

        ranked: list[RecipeSearchResult] = []
        for recipe in self._recipes:
            if not self._matches_dietary_preferences(recipe, dietary):
                continue
            matched = self._matched_ingredients(recipe.ingredients, candidate_ingredients)
            missing = self._missing_pantry_ingredients(recipe.ingredients, pantry)
            score = self._score_recipe(recipe, query_terms, pantry, matched, missing)
            if score <= 0:
                continue
            ranked.append(
                RecipeSearchResult(
                    recipe=recipe,
                    score=round(min(score, 1.0), 4),
                    matched_ingredients=matched,
                    missing_ingredients=missing,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        results = ranked[:limit]
        confidence = max((result.score for result in results), default=0.0)
        fallback_reason = self._fallback_reason(confidence, len(results))
        use_web_fallback = fallback_reason is not None
        diagnostics = RetrievalDiagnostics(
            confidence=confidence,
            confidence_threshold=self.confidence_threshold,
            result_count=len(results),
            min_results=self.min_results,
            use_web_fallback=use_web_fallback,
            fallback_reason=fallback_reason,
        )
        return RecipeSearchResponse(
            results=results,
            confidence=confidence,
            use_web_fallback=use_web_fallback,
            threshold=self.confidence_threshold,
            diagnostics=diagnostics,
        )

    def _fallback_reason(self, confidence: float, result_count: int) -> str | None:
        if confidence < self.confidence_threshold:
            return "low_confidence"
        if result_count < self.min_results:
            return "too_few_results"
        return None

    def _score_recipe(
        self,
        recipe: Recipe,
        query_terms: set[str],
        pantry: list[str],
        matched_ingredients: list[str],
        missing_ingredients: list[str],
    ) -> float:
        index = self._recipes.index(recipe)
        bm25 = self._bm25_score(index, query_terms)
        ingredient_match_ratio = len(matched_ingredients) / max(len(recipe.ingredients), 1)
        pantry_coverage = self._pantry_coverage(recipe.ingredients, pantry)
        missing_penalty = len(missing_ingredients) / max(len(recipe.ingredients), 1)
        raw_score = (
            (0.55 * normalize_bm25(bm25))
            + (0.25 * ingredient_match_ratio)
            + (0.25 * pantry_coverage)
            - (0.20 * missing_penalty)
        )
        return max(raw_score, 0.0)

    def _bm25_score(self, doc_index: int, query_terms: set[str]) -> float:
        if not query_terms or not self._recipes:
            return 0.0
        k1 = 1.5
        b = 0.75
        frequencies = self._term_frequencies[doc_index]
        doc_len = sum(frequencies.values())
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = self._doc_freq.get(term, 0)
            idf = math.log(1 + (len(self._recipes) - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1))
            score += idf * (term_frequency * (k1 + 1)) / denominator
        return score

    def _matched_ingredients(self, ingredients: list[str], candidates: list[str]) -> list[str]:
        matches = []
        for ingredient in ingredients:
            ingredient_norm = normalize_text(ingredient)
            ingredient_tokens = tokenize(ingredient_norm)
            if any(
                candidate in ingredient_norm
                or ingredient_norm in candidate
                or candidate in ingredient_tokens
                for candidate in candidates
            ):
                matches.append(ingredient)
        return matches

    def _missing_pantry_ingredients(self, ingredients: list[str], pantry: list[str]) -> list[str]:
        if not pantry:
            return []
        return [
            ingredient
            for ingredient in ingredients
            if not self._matched_ingredients([ingredient], pantry)
        ]

    def _pantry_coverage(self, ingredients: list[str], pantry: list[str]) -> float:
        if not ingredients or not pantry:
            return 0.0
        matched = self._matched_ingredients(ingredients, pantry)
        return len(matched) / len(ingredients)

    def _matches_dietary_preferences(self, recipe: Recipe, preferences: list[str]) -> bool:
        if not preferences:
            return True
        recipe_tags = set(normalize_items(recipe.dietary_tags))
        for preference in preferences:
            if preference == "vegetarian" and ("vegetarian" in recipe_tags or "vegan" in recipe_tags):
                continue
            if preference not in recipe_tags:
                return False
        return True

    def _build_document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in self._documents:
            frequencies.update(document)
        return frequencies

    def _average_document_length(self) -> float:
        if not self._term_frequencies:
            return 0.0
        return sum(sum(frequencies.values()) for frequencies in self._term_frequencies) / len(
            self._term_frequencies
        )

    def _load_recipes(self) -> list[Recipe]:
        if not self.recipes_path.exists():
            return []
        with self.recipes_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return [Recipe.from_dict(item) for item in data]


def normalize_bm25(score: float) -> float:
    """Map an unbounded BM25 score into a stable 0-1 confidence component."""

    return score / (score + 2.0) if score > 0 else 0.0


def normalize_items(items: list[str]) -> list[str]:
    """Normalize a list of ingredient or tag strings."""

    return [normalize_text(item) for item in items if normalize_text(item)]


def meaningful_terms(text: str) -> set[str]:
    """Return query terms useful for retrieval and ingredient matching."""

    return {term for term in tokenize(text) if len(term) > 2 and term not in STOPWORDS}


def normalize_text(text: str) -> str:
    """Normalize text for matching."""

    return re.sub(r"\s+", " ", str(text).strip().lower())


def slugify(text: str) -> str:
    """Create a stable ID from a recipe title."""

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "recipe"
