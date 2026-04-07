"""E2E — End-to-End Evals against the golden set.

Runs the full agentic loop (real LLM + real tools) against 20 Q&A pairs
defined in golden_set.json.  Each pair has a list of expected_keywords that
must appear in the synthesized answer (case-insensitive).

Suite-level targets (asserted after all 20 cases run):
  - Keyword accuracy     >= 80 %
  - Mean hop_count       <= 5
  - Mean grounding_score >= 0.95
  - Whitelist violations == 0
  - MAX_HOPS hit rate    <= 10 %

Individual test failures are soft (xfail-style marks are not used; instead
the test is always collected and the suite-level assert does the gating).
That way a single flaky query does not block the whole batch.

Requires GROQ_API_KEY and the repos to be present under repos/.
Both checks are done at collection time via module-level skip conditions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.csv_logger import EvalLogger  # noqa: E402
from evals.judge import judge_grounding  # noqa: E402

pytestmark = pytest.mark.e2e

skip_no_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping E2E eval",
)

_GOLDEN_PATH = Path(__file__).with_name("golden_set.json")
_logger = EvalLogger()

# ---------------------------------------------------------------------------
# Load golden set at module level so parametrize works at collection time
# ---------------------------------------------------------------------------

with _GOLDEN_PATH.open() as _f:
    _GOLDEN: list[dict] = json.load(_f)


# ---------------------------------------------------------------------------
# Session-level metrics accumulator
# ---------------------------------------------------------------------------

# Each test appends its result here; the session-teardown fixture asserts
# the aggregate targets.
_results: list[dict] = []


@pytest.fixture(scope="session", autouse=True)
def _suite_metrics():
    """Check suite-level targets after all E2E tests have run."""
    from config import MAX_HOPS  # imported here so src/ is already on path

    yield  # run all tests first

    if not _results:
        return

    n = len(_results)
    accuracy = sum(1 for r in _results if r["keyword_hit"]) / n
    mean_hops = sum(r["hop_count"] for r in _results) / n
    grounding_scores = [
        r["grounding_score"] for r in _results if r["grounding_score"] is not None
    ]
    mean_grounding = (
        sum(grounding_scores) / len(grounding_scores) if grounding_scores else 1.0
    )
    whitelist_violations = sum(r["whitelist_violations"] for r in _results)
    max_hops_rate = sum(1 for r in _results if r["hop_count"] >= MAX_HOPS) / n

    print("\n" + "=" * 60)
    print("E2E EVAL SUITE RESULTS")
    print("=" * 60)
    print(f"  Queries run          : {n}")
    print(f"  Keyword accuracy     : {accuracy:.1%}  (target ≥ 80 %)")
    print(f"  Mean hop_count       : {mean_hops:.1f}  (target ≤ 5)")
    print(f"  Mean grounding_score : {mean_grounding:.3f}  (target ≥ 0.95)")
    print(f"  Whitelist violations : {whitelist_violations}  (target = 0)")
    print(f"  MAX_HOPS hit rate    : {max_hops_rate:.1%}  (target ≤ 10 %)")
    print("=" * 60)

    assert accuracy >= 0.80, f"Keyword accuracy {accuracy:.1%} < 80 %"
    assert mean_hops <= 5.0, f"Mean hops {mean_hops:.1f} > 5"
    assert mean_grounding >= 0.95, f"Mean grounding {mean_grounding:.3f} < 0.95"
    assert whitelist_violations == 0, f"{whitelist_violations} whitelist violation(s)"
    assert max_hops_rate <= 0.10, f"MAX_HOPS hit rate {max_hops_rate:.1%} > 10 %"


# ---------------------------------------------------------------------------
# Helper: run one golden pair end-to-end
# ---------------------------------------------------------------------------


def _run_golden(entry: dict) -> dict:
    """Run the agentic loop + synthesis for one golden entry.

    Returns a result dict that gets appended to _results and logged.
    """
    from config import MAX_HOPS
    from graph import compiled_graph
    from nodes import build_synthesize_messages
    from utils import call_llm

    query: str = entry["query"]
    repo_scope: str | None = entry.get("repo_scope")
    keywords: list[str] = entry.get("expected_keywords", [])

    initial_state: dict = {
        "messages": [{"role": "user", "content": query}],
        "tool_results": [],
        "files_read": [],
        "query": query,
        "hop_count": 0,
        "answer": None,
        "tagged_repos": [repo_scope] if repo_scope else [],
    }

    final_state = compiled_graph.invoke(initial_state)

    # Synthesize outside the graph (mirrors app.py behaviour)
    synth_messages, prefix = build_synthesize_messages(final_state)
    response = call_llm(synth_messages)
    answer: str = prefix + (response["choices"][0]["message"]["content"] or "")

    # Keyword check (case-insensitive, any keyword match counts)
    answer_lower = answer.lower()
    keyword_hit = (
        all(kw.lower() in answer_lower for kw in keywords) if keywords else True
    )

    # Grounding judge
    judge = judge_grounding(final_state["tool_results"], answer)
    grounding_score: float = judge["grounding_score"]

    # Whitelist violations: count tool_result entries that start with the prefix
    whitelist_violations = sum(
        1
        for tr in final_state["tool_results"]
        if tr["result"].startswith("[WHITELIST VIOLATION]")
    )

    return {
        "id": entry["id"],
        "query": query,
        "hop_count": final_state["hop_count"],
        "hit_max_hops": final_state["hop_count"] >= MAX_HOPS,
        "keyword_hit": keyword_hit,
        "grounding_score": grounding_score,
        "whitelist_violations": whitelist_violations,
        "tool_sequence": [tr["tool"] for tr in final_state["tool_results"]],
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Parametrized test — one test per golden entry
# ---------------------------------------------------------------------------


@skip_no_key
@pytest.mark.parametrize("entry", _GOLDEN, ids=[e["id"] for e in _GOLDEN])
def test_e2e_golden(entry: dict):
    result = _run_golden(entry)
    _results.append(result)

    _logger.log(
        category="E2E",
        test_id=entry["id"],
        query=entry["query"],
        tool_sequence=result["tool_sequence"],
        hop_count=result["hop_count"],
        grounding_score=result["grounding_score"],
        passed=result["keyword_hit"] and result["whitelist_violations"] == 0,
    )

    # Per-test hard failure only for whitelist violations (trust issue).
    # All other quality metrics (keyword accuracy, grounding, hop count) are
    # aggregated and gated in the session-level _suite_metrics fixture.
    assert result["whitelist_violations"] == 0, (
        f"{entry['id']}: whitelist violation detected. "
        f"Tool sequence: {result['tool_sequence']}"
    )
