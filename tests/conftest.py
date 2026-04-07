from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env before any test module is collected, so that os.environ-based
# pytest.mark.skipif conditions (e.g. GROQ_API_KEY checks) see the values.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# src/ uses bare imports (no package prefix), so it must be on sys.path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# tests/ must be on sys.path so that `from evals.xxx import ...` resolves
# to tests/evals/ from within any eval test file.
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


@pytest.fixture
def make_tool_result():
    """Return a helper that constructs a ToolResult dict."""
    from state import ToolResult  # noqa: F401 — imported for type reference

    def _make(tool: str, args: dict, result: str) -> dict:
        return {"tool": tool, "args": args, "result": result}

    return _make


@pytest.fixture
def state_factory():
    """Return a factory for building minimal valid AgentState dicts."""

    def _make(
        query: str = "test query",
        hop_count: int = 0,
        answer: str | None = None,
        messages: list[dict] | None = None,
        tool_results: list | None = None,
        tagged_repos: list[str] | None = None,
    ) -> dict:
        return {
            "messages": messages or [{"role": "user", "content": query}],
            "tool_results": tool_results or [],
            "files_read": [],
            "query": query,
            "hop_count": hop_count,
            "answer": answer,
            "tagged_repos": tagged_repos or [],
        }

    return _make
