"""G2 — Grounding Evals.

Tests that the synthesized answer is grounded in tool_results — every code
claim is traceable back to what the agent actually read.

Eval method: LLM-as-judge (judge_grounding).  Each test constructs a
(tool_results, answer) pair directly; no full agent run is needed, so these
tests are fast and deterministic on the input side.

Pass conditions
---------------
G2-01  Correct reference             → grounding_score >= 0.95, no violations
G2-02  Empty results + hallucination → at least one violation flagged
G2-03  Truncated-file reference      → at least one violation flagged
G2-04  Wrong repo attribution        → at least one violation flagged
G2-05  Fabricated line number        → at least one violation flagged

Requires GROQ_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.csv_logger import EvalLogger  # noqa: E402
from evals.judge import judge_grounding  # noqa: E402

_logger = EvalLogger()

pytestmark = pytest.mark.eval_llm

skip_no_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping LLM eval",
)

# ---------------------------------------------------------------------------
# Shared fixtures (inline dicts — no file I/O needed)
# ---------------------------------------------------------------------------

# A realistic read_file_tool result that an agent would accumulate
_REAL_FILE_RESULT = {
    "tool": "read_file_tool",
    "args": {"repo": "capybaradb", "path": "capybaradb/main.py"},
    "result": (
        "# capybaradb/main.py\n"
        "\n"
        "class CapybaraDB:\n"
        '    """In-memory vector database backed by NumPy arrays."""\n'
        "\n"
        "    def __init__(self, dim: int = 768) -> None:\n"
        "        self.dim = dim\n"
        "        self._vectors: list = []\n"
        "        self._metadata: list = []\n"
        "\n"
        "    def insert(self, vector, metadata=None):\n"
        '        """Insert a vector with optional metadata."""\n'
        "        self._vectors.append(vector)\n"
        "        self._metadata.append(metadata or {})\n"
        "\n"
        "    def query(self, vector, top_k=5):\n"
        '        """Return the top_k most similar vectors."""\n'
        "        raise NotImplementedError\n"
    ),
}

_TRUNCATED_FILE_RESULT = {
    "tool": "read_file_tool",
    "args": {"repo": "capybaradb", "path": "capybaradb/storage.py"},
    "result": (
        "# capybaradb/storage.py\n"
        "\n"
        "import numpy as np\n"
        "\n"
        "def save(vectors, path):\n"
        "    np.save(path, vectors)\n"
        "\n"
        "[truncated at 200 KB]"
    ),
}


# ---------------------------------------------------------------------------
# G2-01 — Correct reference: answer cites real content → high grounding score
# ---------------------------------------------------------------------------


@skip_no_key
def test_g2_01_grounded_answer():
    test_id = "G2-01"
    tool_results = [_REAL_FILE_RESULT]
    answer = (
        "The `CapybaraDB` class in `capybaradb/main.py` is an in-memory vector "
        "database. It stores vectors internally via the `insert` method:\n\n"
        "```python\n"
        "def insert(self, vector, metadata=None):\n"
        "    self._vectors.append(vector)\n"
        "    self._metadata.append(metadata or {})\n"
        "```\n\n"
        "Querying is done via `query(vector, top_k=5)`, which raises "
        "`NotImplementedError` in the current implementation."
    )

    result = judge_grounding(tool_results, answer)
    passed = result["grounding_score"] >= 0.95

    _logger.log(
        category="G2",
        test_id=test_id,
        query="(constructed) correct reference",
        tool_sequence=["read_file_tool"],
        hop_count=1,
        grounding_score=result["grounding_score"],
        passed=passed,
    )

    assert passed, (
        f"Expected grounding_score >= 0.95 for a grounded answer, "
        f"got {result['grounding_score']:.3f}. Violations: {result['violations']}"
    )


# ---------------------------------------------------------------------------
# G2-02 — Hallucination: empty tool_results, answer invents a function
# ---------------------------------------------------------------------------


@skip_no_key
def test_g2_02_hallucination_flagged():
    test_id = "G2-02"
    tool_results: list[dict] = []
    answer = (
        "The `store_embeddings_in_redis()` function in `capybaradb/cache.py` "
        "handles caching of vector embeddings using a Redis connection pool. "
        "It uses the `REDIS_URL` environment variable to configure the client."
    )

    result = judge_grounding(tool_results, answer)
    passed = len(result["violations"]) > 0

    _logger.log(
        category="G2",
        test_id=test_id,
        query="(constructed) hallucination on empty context",
        tool_sequence=[],
        hop_count=0,
        grounding_score=result["grounding_score"],
        passed=passed,
    )

    assert passed, (
        f"Expected judge to flag violations when answer invents content not in "
        f"tool_results. Score: {result['grounding_score']:.3f}, "
        f"violations: {result['violations']}"
    )


# ---------------------------------------------------------------------------
# G2-03 — Truncated file: answer references content beyond the truncation point
# ---------------------------------------------------------------------------


@skip_no_key
def test_g2_03_beyond_truncation_flagged():
    test_id = "G2-03"
    tool_results = [_TRUNCATED_FILE_RESULT]
    # Answer claims to describe a function that would appear after line ~50,
    # well beyond the truncation point visible in the result.
    answer = (
        "In `capybaradb/storage.py`, the `load_shard(shard_id, mmap=False)` "
        "function loads a specific vector shard from disk. It accepts an "
        "optional `mmap` flag to memory-map the file instead of loading it "
        "fully into RAM — useful for large indexes."
    )

    result = judge_grounding(tool_results, answer)
    passed = len(result["violations"]) > 0

    _logger.log(
        category="G2",
        test_id=test_id,
        query="(constructed) content beyond truncation point",
        tool_sequence=["read_file_tool"],
        hop_count=1,
        grounding_score=result["grounding_score"],
        passed=passed,
    )

    assert passed, (
        f"Expected judge to flag references beyond truncation point. "
        f"Score: {result['grounding_score']:.3f}, violations: {result['violations']}"
    )


# ---------------------------------------------------------------------------
# G2-04 — Wrong repo attribution: content is from knowflow but answer says
#          it's in capybaradb
# ---------------------------------------------------------------------------


@skip_no_key
def test_g2_04_wrong_repo_flagged():
    test_id = "G2-04"
    tool_results = [
        {
            "tool": "read_file_tool",
            "args": {"repo": "knowflow", "path": "src/storage.py"},
            "result": (
                "# knowflow/src/storage.py\n"
                "\n"
                "class Neo4jStorage:\n"
                '    """Stores knowledge graph nodes in Neo4j."""\n'
                "\n"
                "    def upsert_node(self, node_id: str, properties: dict) -> None:\n"
                "        ...\n"
            ),
        }
    ]
    # Answer misattributes the class to the wrong repo
    answer = (
        "The `Neo4jStorage` class is defined in `capybaradb/src/storage.py` "
        "(not `knowflow`). It provides `upsert_node(node_id, properties)` "
        "for writing graph nodes."
    )

    result = judge_grounding(tool_results, answer)
    passed = len(result["violations"]) > 0

    _logger.log(
        category="G2",
        test_id=test_id,
        query="(constructed) wrong repo attribution",
        tool_sequence=["read_file_tool"],
        hop_count=1,
        grounding_score=result["grounding_score"],
        passed=passed,
    )

    assert passed, (
        f"Expected judge to flag wrong repo attribution. "
        f"Score: {result['grounding_score']:.3f}, violations: {result['violations']}"
    )


# ---------------------------------------------------------------------------
# G2-05 — Fabricated line number: answer cites :L999 but snippet is from L5
# ---------------------------------------------------------------------------


@skip_no_key
def test_g2_05_fabricated_line_number():
    test_id = "G2-05"
    tool_results = [
        {
            "tool": "search_code",
            "args": {"query": "class CapybaraDB", "repos": ["capybaradb"]},
            "result": (
                '[{"repo": "capybaradb", "file": "capybaradb/main.py", '
                '"line_number": 5, "snippet": "class CapybaraDB:"}]'
            ),
        }
    ]
    # Answer fabricates a line number wildly different from what was returned
    answer = (
        "The `CapybaraDB` class is defined at `capybaradb/main.py:L999`. "
        "This is the primary entry point for the database."
    )

    result = judge_grounding(tool_results, answer)
    passed = len(result["violations"]) > 0

    _logger.log(
        category="G2",
        test_id=test_id,
        query="(constructed) fabricated line number",
        tool_sequence=["search_code"],
        hop_count=1,
        grounding_score=result["grounding_score"],
        passed=passed,
    )

    assert passed, (
        f"Expected judge to flag fabricated line number (:L999 vs actual L5). "
        f"Score: {result['grounding_score']:.3f}, violations: {result['violations']}"
    )
