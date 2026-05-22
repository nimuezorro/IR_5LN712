"""Tool for local recipe retrieval."""

from __future__ import annotations

from pantrypal.retrieval.recipe_retriever import RecipeRetriever
from pantrypal.tools.base import Tool, ToolInput, ToolOutput


class RecipeSearchTool(Tool):
    """Retrieve relevant recipes from the local corpus."""

    name = "recipe_search"
    description = "Searches local recipes by ingredients, title, and tags."

    def __init__(self, retriever: RecipeRetriever) -> None:
        """Create a recipe search tool."""

        self.retriever = retriever

    def run(self, input_data: ToolInput) -> ToolOutput:
        """Search recipes and format compact results."""

        query = str(input_data.get("query", ""))
        limit = int(input_data.get("limit", 5))
        pantry_items = [str(item) for item in input_data.get("pantry_items", [])]
        dietary_preferences = [
            str(item) for item in input_data.get("dietary_preferences", [])
        ]
        response = self.retriever.search(
            query,
            limit=limit,
            pantry_ingredients=pantry_items,
            dietary_preferences=dietary_preferences,
        )
        if not response.results:
            return {
                "tool_name": self.name,
                "content": "No local recipe matches found.",
                "results": [],
                "source_type": "local",
                "confidence": 0.0,
                "use_web_fallback": True,
                "confidence_threshold": response.threshold,
                "fallback_reason": response.diagnostics.fallback_reason,
            }

        lines = []
        metadata = []
        for result in response.results:
            recipe = result.recipe
            matched = ", ".join(result.matched_ingredients) or "none"
            lines.append(
                f"- {recipe.title} ({result.score:.2f}): "
                f"{', '.join(recipe.ingredients)} | matched: {matched}"
            )
            metadata.append(
                {
                    "id": recipe.id,
                    "title": recipe.title,
                    "ingredients": recipe.ingredients,
                    "instructions": recipe.instructions,
                    "cuisine": recipe.cuisine,
                    "dietary_tags": recipe.dietary_tags,
                    "prep_time": recipe.prep_time,
                    "score": result.score,
                    "matched_ingredients": result.matched_ingredients,
                    "missing_ingredients": result.missing_ingredients,
                    "source": "local",
                }
            )
        return {
            "tool_name": self.name,
            "content": "\n".join(lines),
            "results": metadata,
            "source_type": "local",
            "confidence": response.confidence,
            "use_web_fallback": response.use_web_fallback,
            "confidence_threshold": response.threshold,
            "fallback_reason": response.diagnostics.fallback_reason,
            "diagnostics": {
                "confidence": response.diagnostics.confidence,
                "confidence_threshold": response.diagnostics.confidence_threshold,
                "result_count": response.diagnostics.result_count,
                "min_results": response.diagnostics.min_results,
                "use_web_fallback": response.diagnostics.use_web_fallback,
                "fallback_reason": response.diagnostics.fallback_reason,
            },
        }
