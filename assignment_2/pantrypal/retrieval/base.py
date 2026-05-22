"""Shared retrieval abstractions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Confidence and fallback metadata returned by retrievers."""

    confidence: float
    confidence_threshold: float
    result_count: int
    min_results: int
    use_web_fallback: bool
    fallback_reason: str | None = None
