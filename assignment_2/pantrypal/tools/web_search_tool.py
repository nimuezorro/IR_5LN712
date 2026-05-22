"""Provider-based web search fallback tool."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pantrypal.tools.base import Tool, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """One structured web search result."""

    title: str
    snippet: str
    url: str
    confidence: float
    result_type: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the result for tool output."""

        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "confidence": self.confidence,
            "type": self.result_type,
            "source": "web",
        }


class SearchProvider(ABC):
    """Abstract search provider interface."""

    name: str

    @abstractmethod
    def search(self, query: str, search_type: str, limit: int) -> list[WebSearchResult]:
        """Return structured search results."""


class DisabledSearchProvider(SearchProvider):
    """Provider used when web search is intentionally disabled."""

    name = "disabled"

    def search(self, query: str, search_type: str, limit: int) -> list[WebSearchResult]:
        """Return no results."""

        return []


class MockSearchProvider(SearchProvider):
    """Deterministic provider for tests and offline demos."""

    name = "mock"

    def search(self, query: str, search_type: str, limit: int) -> list[WebSearchResult]:
        """Return mock results shaped like web evidence."""

        templates = {
            "recipe": [
                (
                    "Web Recipe: Pantry-Friendly {query}",
                    "A web fallback recipe result for {query}, useful when local recipes are sparse.",
                    "https://example.com/recipes/{slug}",
                    0.72,
                ),
                (
                    "Simple {query} Dinner Ideas",
                    "Ingredient-focused dinner suggestions gathered from mock web retrieval.",
                    "https://example.com/cooking/{slug}-ideas",
                    0.64,
                ),
            ],
            "substitution": [
                (
                    "Substitution Guide for {query}",
                    "Compares common substitutes and when each works best.",
                    "https://example.com/substitutions/{slug}",
                    0.70,
                )
            ],
            "technique": [
                (
                    "Cooking Technique: {query}",
                    "Step-by-step technique overview with practical cooking notes.",
                    "https://example.com/techniques/{slug}",
                    0.68,
                )
            ],
        }
        selected = templates.get(search_type, templates["recipe"])
        slug = urllib.parse.quote_plus(query.lower().replace(" ", "-"))
        return [
            WebSearchResult(
                title=title.format(query=query),
                snippet=snippet.format(query=query),
                url=url.format(slug=slug),
                confidence=confidence,
                result_type=search_type,
            )
            for title, snippet, url, confidence in selected[:limit]
        ]


class HttpSearchProvider(SearchProvider):
    """Minimal generic HTTP JSON search provider.

    The endpoint is expected to accept ``q`` and return either a top-level list
    or an object with a ``results`` list. Each result may contain ``title``,
    ``snippet``/``description``, ``url``/``link``, and optional ``confidence``.
    """

    name = "http"

    def __init__(self, endpoint: str, api_key: str | None = None) -> None:
        """Create an HTTP provider from environment configuration."""

        self.endpoint = endpoint
        self.api_key = api_key

    def search(self, query: str, search_type: str, limit: int) -> list[WebSearchResult]:
        """Run a configured HTTP search request."""

        url = self._build_url(query, search_type, limit)
        request = urllib.request.Request(url)
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_results = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_results, list):
            return []
        return [self._parse_result(item, search_type) for item in raw_results[:limit]]

    def _build_url(self, query: str, search_type: str, limit: int) -> str:
        separator = "&" if "?" in self.endpoint else "?"
        params = urllib.parse.urlencode({"q": query, "type": search_type, "limit": limit})
        return f"{self.endpoint}{separator}{params}"

    def _parse_result(self, item: Any, search_type: str) -> WebSearchResult:
        if not isinstance(item, dict):
            return WebSearchResult(str(item), "", "", 0.3, search_type)
        return WebSearchResult(
            title=str(item.get("title", "Untitled result")),
            snippet=str(item.get("snippet", item.get("description", ""))),
            url=str(item.get("url", item.get("link", ""))),
            confidence=float(item.get("confidence", item.get("score", 0.5))),
            result_type=search_type,
        )


class WebSearchTool(Tool):
    """Fallback search for recipes, substitutions, and cooking techniques."""

    name = "web_search"
    description = "Fallback web retrieval for recipes, substitutions, and cooking techniques."

    def __init__(self, provider: SearchProvider | None = None) -> None:
        """Create the web search tool with a configurable provider."""

        self.provider = provider or MockSearchProvider()

    def run(self, input_data: ToolInput) -> ToolOutput:
        """Run web search and return structured evidence."""

        query = str(input_data.get("query", ""))
        search_type = self._search_type(input_data, query)
        limit = int(input_data.get("limit", 5))
        try:
            results = self.provider.search(query, search_type, limit)
        except Exception as error:  # pragma: no cover - defensive provider boundary
            logger.exception("Web search provider failed")
            return {
                "tool_name": self.name,
                "content": f"Web search failed gracefully: {error}",
                "query": query,
                "provider": self.provider.name,
                "configured": True,
                "results": [],
                "source_type": "web",
                "confidence": 0.0,
                "error": str(error),
            }

        result_dicts = [result.to_dict() for result in results]
        confidence = max((result.confidence for result in results), default=0.0)
        return {
            "tool_name": self.name,
            "content": self._format_content(results),
            "query": query,
            "provider": self.provider.name,
            "configured": self.provider.name != "disabled",
            "results": result_dicts,
            "source_type": "web",
            "search_type": search_type,
            "confidence": confidence,
        }

    def _search_type(self, input_data: ToolInput, query: str) -> str:
        explicit = str(input_data.get("search_type", "")).lower()
        if explicit in {"recipe", "substitution", "technique"}:
            return explicit
        lower = query.lower()
        if any(word in lower for word in ["substitute", "replacement", "instead of"]):
            return "substitution"
        if any(word in lower for word in ["technique", "how do i", "how to", "method"]):
            return "technique"
        return "recipe"

    def _format_content(self, results: list[WebSearchResult]) -> str:
        if not results:
            return "No web search results found."
        return "\n".join(
            f"- {result.title} ({result.confidence:.2f}) {result.url}: {result.snippet}"
            for result in results
        )


def build_search_provider(
    provider_name: str,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> SearchProvider:
    """Build a search provider from configuration."""

    normalized = provider_name.strip().lower()
    if normalized == "disabled":
        return DisabledSearchProvider()
    if normalized == "http":
        if not endpoint:
            raise ValueError("PANTRYPAL_WEB_SEARCH_ENDPOINT is required for http search.")
        return HttpSearchProvider(endpoint=endpoint, api_key=api_key)
    return MockSearchProvider()
