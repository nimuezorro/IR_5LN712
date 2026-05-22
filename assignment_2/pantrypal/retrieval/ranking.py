"""Small ranking utilities for local information retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


def tokenize(text: str) -> set[str]:
    """Return lower-cased word tokens for simple lexical matching."""

    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def lexical_score(query: str, document_terms: Iterable[str]) -> float:
    """Score a document by query-term overlap.

    The score is intentionally simple for a starter project: it behaves like a
    normalized overlap coefficient and is easy to replace with BM25 later.
    """

    query_terms = tokenize(query)
    doc_terms = {term.lower() for term in document_terms}
    if not query_terms or not doc_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)
