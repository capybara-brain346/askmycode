from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class WhitelistViolation(Exception):
    """Raised when a tool request targets a non-whitelisted repo or path."""


class ToolResult(TypedDict):
    tool: str
    args: dict
    result: str


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]

    tool_results: Annotated[list[ToolResult], operator.add]

    files_read: list[list[str]]

    query: str

    hop_count: int

    answer: str | None

    tagged_repos: list[str]
