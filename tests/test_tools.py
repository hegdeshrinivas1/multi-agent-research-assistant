"""Unit tests for agentresearch.tools."""

import pytest
from agentresearch.tools import search_tavily


def test_empty_query():
    """Empty query should return an empty list without error."""
    assert search_tavily("") == []
    assert search_tavily("   ") == []


def test_fallback_search():
    """Search without TAVILY_API_KEY should return structured fallback results."""
    results = search_tavily("Quantum Computing Breakthroughs", max_results=3)
    assert len(results) > 0
    for item in results:
        assert "title" in item
        assert "url" in item
        assert "content" in item
        assert len(item["title"]) > 0
        assert item["url"].startswith("http")
