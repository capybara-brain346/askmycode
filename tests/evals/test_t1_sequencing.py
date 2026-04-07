"""T1 — Tool Sequencing Evals.

Tests that the agent calls tools in a sensible order for each query type.
Tools are mocked to return instantly with canned data so only the LLM's
planning decisions (not file contents) are under test.

Each test:
  1. Patches nodes.TOOL_FUNCTIONS with instant stubs.
  2. Runs compiled_graph.invoke() with a real LLM call.
  3. Extracts the tool call sequence from final_state["tool_results"].
  4. Asserts structural constraints on that sequence.
  5. Logs the result to logs/eval_results.csv.

Requires GROQ_API_KEY to be set; tests are automatically skipped otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is on path (conftest.py handles this for collected tests, but
# module-level imports need it resolved at collection time too).
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.csv_logger import EvalLogger  # noqa: E402
from evals.mock_tools import instant_tool_functions  # noqa: E402

_logger = EvalLogger()

pytestmark = pytest.mark.eval_llm

skip_no_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping LLM eval",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(monkeypatch, query: str, tagged_repos: list[str] | None = None) -> dict:
    """Patch tool execution and run the full agentic loop."""
    import nodes
    from graph import compiled_graph

    monkeypatch.setattr(nodes, "TOOL_FUNCTIONS", instant_tool_functions())

    initial_state: dict = {
        "messages": [{"role": "user", "content": query}],
        "tool_results": [],
        "files_read": [],
        "query": query,
        "hop_count": 0,
        "answer": None,
        "tagged_repos": tagged_repos or [],
    }
    return compiled_graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# T1-01 — "What repos are available?" should call list_repos and stop in 1 hop
# ---------------------------------------------------------------------------


@skip_no_key
def test_t1_01_list_repos(monkeypatch):
    query = "What repos are available?"
    final = _run(monkeypatch, query)

    seq = [tr["tool"] for tr in final["tool_results"]]
    passed = "list_repos" in seq and final["hop_count"] <= 2

    _logger.log(
        category="T1",
        test_id="T1-01",
        query=query,
        tool_sequence=seq,
        hop_count=final["hop_count"],
        grounding_score=None,
        passed=passed,
    )

    assert "list_repos" in seq, f"Expected list_repos in tool sequence, got: {seq}"
    assert final["hop_count"] <= 2, (
        f"Expected ≤ 2 hops for a simple repo-listing query, got {final['hop_count']}"
    )


# ---------------------------------------------------------------------------
# T1-02 — Auth module question: search_code must come before any read_file_tool
# ---------------------------------------------------------------------------


@skip_no_key
def test_t1_02_search_before_read(monkeypatch):
    query = "What does the auth module do in capynodes-backend?"
    final = _run(monkeypatch, query, tagged_repos=["capynodes-backend"])

    seq = [tr["tool"] for tr in final["tool_results"]]

    has_search = "search_code" in seq
    has_read = "read_file_tool" in seq

    # If the agent reads at all, it must have searched first
    if has_read:
        first_search = next((i for i, t in enumerate(seq) if t == "search_code"), None)
        first_read = next((i for i, t in enumerate(seq) if t == "read_file_tool"), None)
        search_precedes_read = (first_search is not None) and (
            first_search < first_read
        )
    else:
        # If no read occurred, search alone is acceptable
        search_precedes_read = has_search

    passed = search_precedes_read

    _logger.log(
        category="T1",
        test_id="T1-02",
        query=query,
        tool_sequence=seq,
        hop_count=final["hop_count"],
        grounding_score=None,
        passed=passed,
    )

    assert passed, f"Expected search_code before read_file_tool. Tool sequence: {seq}"


# ---------------------------------------------------------------------------
# T1-03 — Repo overview: get_repo_metadata or list_repos should be used,
#          not a blind read_file_tool as the very first call
# ---------------------------------------------------------------------------


@skip_no_key
def test_t1_03_cheap_call_for_overview(monkeypatch):
    query = "What is capybaradb about?"
    final = _run(monkeypatch, query, tagged_repos=["capybaradb"])

    seq = [tr["tool"] for tr in final["tool_results"]]

    cheap_tools = {"get_repo_metadata", "list_repos", "get_file_tree", "search_code"}
    first_tool = seq[0] if seq else None
    passed = first_tool in cheap_tools

    _logger.log(
        category="T1",
        test_id="T1-03",
        query=query,
        tool_sequence=seq,
        hop_count=final["hop_count"],
        grounding_score=None,
        passed=passed,
    )

    assert passed, (
        f"Expected a cheap orientation call first (got '{first_tool}'). "
        f"Full sequence: {seq}"
    )


# ---------------------------------------------------------------------------
# T1-04 — "Where is X implemented?" → search_code must be called
# ---------------------------------------------------------------------------


@skip_no_key
def test_t1_04_search_for_implementation(monkeypatch):
    query = "Where is rate limiting implemented in capynodes-backend?"
    final = _run(monkeypatch, query, tagged_repos=["capynodes-backend"])

    seq = [tr["tool"] for tr in final["tool_results"]]
    passed = "search_code" in seq

    _logger.log(
        category="T1",
        test_id="T1-04",
        query=query,
        tool_sequence=seq,
        hop_count=final["hop_count"],
        grounding_score=None,
        passed=passed,
    )

    assert passed, f"Expected search_code in tool sequence, got: {seq}"


# ---------------------------------------------------------------------------
# T1-05 — Ambiguous cross-repo query: list_repos must be the first tool
# ---------------------------------------------------------------------------


@skip_no_key
def test_t1_05_scope_before_explore(monkeypatch):
    query = "Compare how capybaradb and knowflow handle vector storage"
    final = _run(monkeypatch, query)

    seq = [tr["tool"] for tr in final["tool_results"]]
    first_tool = seq[0] if seq else None
    passed = first_tool == "list_repos"

    _logger.log(
        category="T1",
        test_id="T1-05",
        query=query,
        tool_sequence=seq,
        hop_count=final["hop_count"],
        grounding_score=None,
        passed=passed,
    )

    assert passed, (
        f"Expected list_repos as the first tool for a cross-repo query, "
        f"got '{first_tool}'. Full sequence: {seq}"
    )
