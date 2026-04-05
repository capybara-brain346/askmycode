from __future__ import annotations

import json

from .config import CONTEXT_BUDGET_CHARS, MAX_HOPS, call_llm
from .logger import get_logger
from .state import AgentState, ToolResult, WhitelistViolation
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

_log = get_logger("nodes")

PLAN_SYSTEM = """\
You are Gitwhisper, an expert software engineer answering questions about code repositories.

You have access to the following tools to inspect whitelisted repositories:
- list_repos: list all available repos
- get_file_tree: list files/directories at a path inside a repo
- read_file_tool: read the raw text of a file
- search_code: grep across repos for a string
- get_repo_metadata: get high-level repo info (file count, README, etc.)

Strategy:
1. Always call list_repos first if you don't know which repos are available.
2. Use search_code early to locate relevant code; then read specific files.
3. Navigate the file tree before doing blind reads.
4. Never read the same (repo, path) twice.
5. When you have enough evidence to answer the question fully, do NOT call any more tools — \
just produce a final answer WITHOUT calling any tool.

Be methodical. Ground every claim in what you actually read.
"""

OBSERVE_SYSTEM = """\
You are reviewing the latest tool call result to decide whether you have enough information \
to answer the user's query, or whether you need to call more tools.

You MUST respond with valid JSON only — no markdown fences, no extra text:
{
  "decision": "loop" | "synthesize",
  "reasoning": "<one sentence explanation>"
}

Choose "synthesize" if:
- The tool results contain sufficient evidence to answer the query fully and accurately.
- You have already collected the key files / code snippets needed.

Choose "loop" if:
- Critical information is still missing.
- You need to read more files or run more searches.
- Do NOT loop if you are just being thorough — stop when you have enough.
"""

SYNTHESIZE_SYSTEM = """\
You are Gitwhisper, answering a developer's question about their code repositories.

Rules:
- Ground every claim in the tool results provided. Do not invent code or file paths.
- Cite files as `repo/path/to/file.py:L42` inline.
- Use fenced code blocks with language tags for all code snippets.
- If evidence is incomplete, say so explicitly rather than guessing.
- Answer the question directly. Do not summarise every file read — just answer.
- Do not mention the tool-call process.
"""


def _last_message(state: AgentState) -> dict:
    return state["messages"][-1]


def _format_tool_results_for_context(tool_results: list[ToolResult]) -> str:
    lines: list[str] = []
    for i, tr in enumerate(tool_results, 1):
        args_str = json.dumps(tr["args"], ensure_ascii=False)
        lines.append(f"--- Result {i}: {tr['tool']}({args_str}) ---")
        lines.append(tr["result"])
        lines.append("")
    return "\n".join(lines)


def _trim_tool_results_to_budget(tool_results: list[ToolResult]) -> list[ToolResult]:
    total = sum(len(tr["result"]) for tr in tool_results)
    trimmed = list(tool_results)
    while total > CONTEXT_BUDGET_CHARS and len(trimmed) > 1:
        removed = trimmed.pop(0)
        total -= len(removed["result"])
    return trimmed


def plan_node(state: AgentState) -> dict:
    hop = state["hop_count"] + 1
    _log.info("[plan] hop=%d | querying LLM for next tool call", hop)
    messages = [{"role": "system", "content": PLAN_SYSTEM}] + state["messages"]
    response = call_llm(messages, tools=TOOL_SCHEMAS)
    assistant_msg: dict = response["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls") or []
    if tool_calls:
        names = [tc["function"]["name"] for tc in tool_calls]
        _log.info("[plan] hop=%d | model chose tool(s): %s", hop, names)
    else:
        _log.info("[plan] hop=%d | model chose to answer directly (no tool calls)", hop)
    return {"messages": [assistant_msg]}


def tools_node(state: AgentState) -> dict:
    last = _last_message(state)
    tool_calls: list[dict] = last.get("tool_calls") or []
    _log.info("[tools] executing %d tool call(s)", len(tool_calls))

    tool_messages: list[dict] = []
    tool_results: list[ToolResult] = []

    for tc in tool_calls:
        call_id: str = tc["id"]
        fn_name: str = tc["function"]["name"]
        try:
            raw_args = tc["function"].get("arguments", "{}")
            args: dict = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}

        _log.info("[tools] → %s(%s)", fn_name, json.dumps(args, ensure_ascii=False))
        func = TOOL_FUNCTIONS.get(fn_name)
        if func is None:
            result_str = f"[ERROR] Unknown tool: {fn_name}"
            _log.warning("[tools] unknown tool requested: %s", fn_name)
        else:
            try:
                result = func(**args)
                result_str = (
                    json.dumps(result, ensure_ascii=False, indent=2)
                    if not isinstance(result, str)
                    else result
                )
                _log.info("[tools] ✓ %s completed (%d chars)", fn_name, len(result_str))
            except WhitelistViolation as exc:
                result_str = f"[WHITELIST VIOLATION] {exc}"
                _log.warning("[tools] whitelist violation in %s: %s", fn_name, exc)
            except Exception as exc:
                result_str = f"[ERROR] {type(exc).__name__}: {exc}"
                _log.error(
                    "[tools] error in %s: %s: %s", fn_name, type(exc).__name__, exc
                )

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str,
            }
        )
        tool_results.append(ToolResult(tool=fn_name, args=args, result=result_str))

    return {
        "messages": tool_messages,
        "tool_results": tool_results,
    }


def observe_node(state: AgentState) -> dict:
    new_hop = state["hop_count"] + 1
    _log.info("[observe] hop=%d | evaluating tool results", new_hop)

    # Update files_read from any read_file_tool calls in the latest results
    files_read: list[list[str]] = list(state.get("files_read") or [])
    already = {tuple(pair) for pair in files_read}
    for tr in state["tool_results"]:
        if tr["tool"] == "read_file_tool":
            pair = [tr["args"].get("repo", ""), tr["args"].get("path", "")]
            if tuple(pair) not in already:
                files_read.append(pair)
                already.add(tuple(pair))

    trimmed_results = _trim_tool_results_to_budget(state["tool_results"])

    context_str = _format_tool_results_for_context(trimmed_results)
    observe_messages = [
        {"role": "system", "content": OBSERVE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Query: {state['query']}\n\n"
                f"Hop: {new_hop} / {MAX_HOPS}\n\n"
                f"Files already read: {json.dumps(files_read)}\n\n"
                f"Tool results so far:\n{context_str}"
            ),
        },
    ]

    if new_hop >= MAX_HOPS:
        _log.warning(
            "[observe] hop=%d | MAX_HOPS reached — forcing synthesize", new_hop
        )
        return {
            "hop_count": new_hop,
            "files_read": files_read,
            "answer": "READY",
        }

    response = call_llm(observe_messages, json_mode=True)
    raw_content: str = response["choices"][0]["message"]["content"]

    try:
        decision_obj: dict = json.loads(raw_content)
        decision: str = decision_obj.get("decision", "loop")
        reasoning: str = decision_obj.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        decision = "loop"
        reasoning = "(parse error — defaulting to loop)"

    _log.info(
        "[observe] hop=%d | decision=%s | reason: %s",
        new_hop,
        decision.upper(),
        reasoning,
    )
    return {
        "hop_count": new_hop,
        "files_read": files_read,
        "answer": "READY" if decision == "synthesize" else None,
    }


def synthesize_node(state: AgentState) -> dict:
    _log.info(
        "[synthesize] writing final answer | %d tool result(s) | %d hop(s)",
        len(state["tool_results"]),
        state["hop_count"],
    )
    trimmed_results = _trim_tool_results_to_budget(state["tool_results"])
    context_str = _format_tool_results_for_context(trimmed_results)

    prefix = ""
    if state["hop_count"] >= MAX_HOPS:
        prefix = "*Reached max search depth. Answer based on partial results.*\n\n"

    synthesize_messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
        {
            "role": "user",
            "content": (f"Query: {state['query']}\n\nTool results:\n{context_str}"),
        },
    ]

    history = [
        m
        for m in state["messages"]
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]
    if history:
        synthesize_messages = (
            [{"role": "system", "content": SYNTHESIZE_SYSTEM}]
            + history
            + synthesize_messages[1:]
        )

    response = call_llm(synthesize_messages)
    answer_text: str = response["choices"][0]["message"]["content"]

    return {"answer": prefix + answer_text}
