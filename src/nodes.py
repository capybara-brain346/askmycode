from __future__ import annotations

import json

from config import CONTEXT_BUDGET_CHARS, MAX_HOPS
from utils import call_llm
from logger import get_logger
from state import AgentState, ToolResult, WhitelistViolation
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = get_logger("nodes")

PLAN_SYSTEM = """\
You are askmycode, an expert software engineer who answers precise questions about codebases by reading them directly.

Tools available:
- list_repos       — list all available repos
- get_file_tree    — list files/directories at a path inside a repo
- read_file_tool   — read the raw text of a file
- search_code      — grep across repos for a pattern (regex supported); returns matching lines with context
- get_repo_metadata — high-level repo info: file count, top-level tree, README excerpt

## Exploration strategy

**Go targeted first:**
- Use search_code before navigating trees. Searching for a symbol or string is almost always faster than browsing.
- Only call get_repo_metadata when you need broad orientation about an unfamiliar repo.
- Read files where the answer actually lives — not tangentially related ones.

**Query-specific tactics:**
- "How does X work" → find the definition with search_code, then read it; read callers/tests only if necessary.
- "Where is X defined" → search_code for the symbol name first.
- "What does this repo do" → get_repo_metadata + top-level module structure is usually enough.
- "Why does X happen" → trace the call chain: definition → callers → config.

**When to stop calling tools:**
- You have the specific code, config, or explanation needed to answer accurately → answer directly without calling any tool.
- Do NOT call another tool "just to be sure". If the code is clear, trust it.
- Never read the same (repo, path) twice.

Be precise. Cite evidence. Never guess or hallucinate file contents.
"""

OBSERVE_SYSTEM = """\
You are deciding whether the agent has enough evidence to answer the user's query, or must keep searching.

Respond with valid JSON only — no markdown, no extra text:
{"decision": "loop" | "synthesize", "reasoning": "<one sentence>"}

## Choose "synthesize" when
- The specific code, config, or explanation needed to answer the question has been read.
- The core definition or implementation is in the results — callers/tests are optional unless directly asked about.
- Continued searching is unlikely to reveal new essential information.

## Choose "loop" when
- A file or symbol directly required for the answer has not been read yet.
- search_code returned a reference but the actual definition was never opened.
- The query has multiple parts and some remain completely unaddressed.

## Bias toward "synthesize"
Over-exploration wastes hops and adds noise. An answer grounded in partial but correct evidence is better than an endless loop. When in doubt and the key information is present, synthesize.
"""

SYNTHESIZE_SYSTEM = """\
You are askmycode, a senior software engineer writing a precise answer to a developer's question about their codebase.

## Rules
- **Ground every claim** in the tool results. Do not invent code, file paths, or behaviour.
- **Cite sources** inline as `repo/path/to/file.py:L42` or `repo/path/to/file.py:L10-25`.
- **Code blocks**: use fenced blocks with language tags for all snippets. Quote only what the answer needs — keep them short and relevant.
- **Structure**: open with the direct answer, then explain. Use headers, bullet lists, or numbered steps when the answer has multiple parts.
- **Uncertainty**: if evidence is incomplete or ambiguous, say so explicitly. Never guess or fill gaps with plausible-sounding but unread content.
- **Tone**: write as a knowledgeable colleague. No filler, no "based on the tool results I can see that…" — just the answer.
- Do not mention the search process or tool calls.
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
    logger.debug("plan hop=%d", hop)
    messages = [{"role": "system", "content": PLAN_SYSTEM}] + state["messages"]
    response = call_llm(messages, tools=TOOL_SCHEMAS)
    assistant_msg: dict = response["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls") or []
    if tool_calls:
        names = [tc["function"]["name"] for tc in tool_calls]
        logger.debug("plan hop=%d tools=%s", hop, names)
    else:
        logger.info("plan hop=%d direct_answer=true", hop)
    return {"messages": [assistant_msg]}


def tools_node(state: AgentState) -> dict:
    last = _last_message(state)
    tool_calls: list[dict] = last.get("tool_calls") or []
    logger.debug("tools count=%d", len(tool_calls))

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

        logger.debug(
            "tool_call fn=%s args=%s", fn_name, json.dumps(args, ensure_ascii=False)
        )
        func = TOOL_FUNCTIONS.get(fn_name)
        if func is None:
            result_str = f"[ERROR] Unknown tool: {fn_name}"
            logger.warning("tool_unknown fn=%s", fn_name)
        else:
            try:
                result = func(**args)
                result_str = (
                    json.dumps(result, ensure_ascii=False, indent=2)
                    if not isinstance(result, str)
                    else result
                )
                logger.debug(
                    "tool_done fn=%s result_chars=%d", fn_name, len(result_str)
                )
            except WhitelistViolation as exc:
                result_str = f"[WHITELIST VIOLATION] {exc}"
                logger.warning("tool_whitelist_violation fn=%s error=%s", fn_name, exc)
            except Exception as exc:
                result_str = f"[ERROR] {type(exc).__name__}: {exc}"
                logger.error(
                    "tool_error fn=%s type=%s message=%s",
                    fn_name,
                    type(exc).__name__,
                    exc,
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
    logger.debug("observe hop=%d", new_hop)

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
        logger.warning("observe hop=%d max_hops=true forcing=synthesize", new_hop)
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

    if decision == "synthesize":
        logger.info(
            "observe hop=%d decision=synthesize reason=%s",
            new_hop,
            reasoning,
        )
    else:
        logger.debug(
            "observe hop=%d decision=loop reason=%s",
            new_hop,
            reasoning,
        )
    return {
        "hop_count": new_hop,
        "files_read": files_read,
        "answer": "READY" if decision == "synthesize" else None,
    }


def build_synthesize_messages(state: AgentState) -> tuple[list[dict], str]:
    """Return (messages, prefix) for the synthesis LLM call.

    The prefix is prepended to the answer when MAX_HOPS was reached.
    """
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

    return synthesize_messages, prefix
