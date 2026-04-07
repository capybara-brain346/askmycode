"""Canned tool responses and a patcher for deterministic agent runs.

Usage in tests
--------------
    import nodes
    monkeypatch.setattr(nodes, "TOOL_FUNCTIONS", instant_tool_functions())

    final = compiled_graph.invoke(initial_state)
    tool_seq = [tr["tool"] for tr in final["tool_results"]]
"""

from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# Canned results — realistic enough for the LLM to form a coherent tool
# sequence, but returned instantly without touching the filesystem.
# ---------------------------------------------------------------------------

CANNED: dict[str, object] = {
    "list_repos": [
        "atjbot",
        "capybaradb",
        "capynodes-backend",
        "knowflow",
        "pagedattention-simulation",
        "sqs-implementation",
    ],
    "get_file_tree": {
        "dirs": ["src", "tests", "docs", "scripts"],
        "files": [
            {"name": "README.md", "size_bytes": 3200},
            {"name": "pyproject.toml", "size_bytes": 512},
        ],
    },
    "read_file_tool": (
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
        "        ...\n"
    ),
    "search_code": [
        {
            "repo": "capybaradb",
            "file": "capybaradb/main.py",
            "line_number": 5,
            "snippet": "class CapybaraDB:",
        },
        {
            "repo": "capybaradb",
            "file": "capybaradb/main.py",
            "line_number": 10,
            "snippet": "    def insert(self, vector, metadata=None):",
        },
    ],
    "get_repo_metadata": {
        "repo": "capybaradb",
        "file_count": 47,
        "last_modified": "2025-12-01T10:00:00",
        "top_extensions": {".py": 34, ".md": 6, ".json": 4, ".toml": 3},
        "readme_excerpt": (
            "CapybaraDB is a lightweight, dependency-light vector database "
            "for local RAG prototypes. It stores vectors as NumPy arrays and "
            "supports cosine, dot-product, and L2 similarity search."
        ),
    },
}


def _stub(result: object) -> Callable:
    def _fn(**kwargs):  # noqa: ANN001 — test stub
        return result

    return _fn


def instant_tool_functions() -> dict[str, Callable]:
    """Return a TOOL_FUNCTIONS-compatible dict where every tool returns instantly."""
    return {name: _stub(result) for name, result in CANNED.items()}
