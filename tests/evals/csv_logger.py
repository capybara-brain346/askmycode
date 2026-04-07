"""Append-only CSV logger for eval runs.

Each run writes one row to logs/eval_results.csv, creating the file and
header on first use.  Designed to be imported by any eval test file.

Usage
-----
    from evals.csv_logger import EvalLogger

    _logger = EvalLogger()
    _logger.log(
        category="T1",
        test_id="T1-01",
        query="What repos are available?",
        tool_sequence=["list_repos"],
        hop_count=1,
        grounding_score=None,
        passed=True,
    )
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_LOG = Path(__file__).resolve().parents[2] / "logs" / "eval_results.csv"

_HEADERS = [
    "timestamp",
    "category",
    "test_id",
    "query",
    "tool_sequence",
    "hop_count",
    "grounding_score",
    "passed",
]


class EvalLogger:
    def __init__(self, path: Path = _DEFAULT_LOG) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_HEADERS).writeheader()

    def log(
        self,
        *,
        category: str,
        test_id: str,
        query: str,
        tool_sequence: list[str],
        hop_count: int,
        grounding_score: float | None,
        passed: bool,
    ) -> None:
        row = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "category": category,
            "test_id": test_id,
            "query": query,
            "tool_sequence": ",".join(tool_sequence),
            "hop_count": hop_count,
            "grounding_score": (
                "" if grounding_score is None else f"{grounding_score:.3f}"
            ),
            "passed": "PASS" if passed else "FAIL",
        }
        with self._path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_HEADERS).writerow(row)
        # Mirror to stdout so CI logs capture it without opening the CSV
        print(
            f"[eval] {row['category']} {row['test_id']} {row['passed']} "
            f"hops={hop_count} grounding={row['grounding_score'] or '-'}",
            file=sys.stderr,
        )
