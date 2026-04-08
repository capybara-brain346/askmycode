import json
import uuid

from config import CONTEXT_BUDGET_CHARS, MAX_HOPS
from logger import get_logger
from state import AgentState, ToolResult, WhitelistViolation
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from utils import call_llm

logger = get_logger("nodes")

PLAN_SYSTEM = """\
You are askmycode, an expert software engineer who answers precise questions about codebases by reading them directly.

## Mandatory rule
You have NO prior knowledge of these repositories. You MUST call at least one tool on every turn — do not answer from memory or training data. If you already know the answer from your training, that knowledge may be wrong or outdated for this specific codebase. Always verify by reading the actual files.

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

**Conceptual / feature queries** (e.g. "does this repo do web scraping?", "how is auth handled?", "where is caching?"):
- The concept may not appear verbatim in the code. Search for the *libraries or primitives* the concept implies:
  - web scraping / HTTP fetching → `requests|httpx|BeautifulSoup|scrapy|playwright|aiohttp|urllib`
  - authentication / auth → `jwt|token|OAuth|authenticate|login|password`
  - caching → `cache|redis|memcached|lru_cache|ttl`
  - queuing / messaging → `queue|celery|sqs|rabbitmq|kafka|publish|subscribe`
- Use `|` alternation in a **single** search_code call to cover all synonyms at once.
- search_code supports full extended regex (grep -E): `(scrape|fetch).*url` is valid.

**When a search returns no results:**
- Do NOT conclude the concept is absent after one failed search.
- Do NOT jump to get_file_tree browsing — that is slow and unlikely to help.
- Pivot immediately: try alternative library names, root concepts, or abbreviated terms.
- Only after two or more targeted searches all return nothing should you consider the feature absent.

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
Over-exploration wastes hops and adds noise. An answer grounded in partial but correct evidence is better than an endless loop. When the key information has already been read, synthesize.

**Exception — choose "loop" when evidence is absent:**
- All search_code calls returned zero results AND no relevant file has been opened.
- Searches used only one narrow keyword for a concept that has many synonyms or library names — the agent should try alternatives before giving up.
- The query cannot be answered from the current results without guessing or inventing detail.
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
    if total <= CONTEXT_BUDGET_CHARS:
        return list(tool_results)
    keep_from = 0
    for i, tr in enumerate(tool_results):
        if total <= CONTEXT_BUDGET_CHARS:
            break
        if i < len(tool_results) - 1:
            total -= len(tr["result"])
            keep_from = i + 1
    return list(tool_results[keep_from:])


def _forced_tool_call(state: AgentState) -> dict:
    tagged = state.get("tagged_repos") or []
    if tagged:
        fn_name = "get_repo_metadata"
        fn_args = json.dumps({"repo": tagged[0]})
    else:
        fn_name = "list_repos"
        fn_args = "{}"
    call_id = f"forced_{uuid.uuid4().hex[:8]}"
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": fn_name, "arguments": fn_args},
            }
        ],
    }


def plan_node(state: AgentState) -> dict:
    hop = state["hop_count"] + 1
    logger.debug("plan hop=%d", hop)

    system_content = PLAN_SYSTEM
    tagged = state.get("tagged_repos") or []
    if tagged:
        repo_list = ", ".join(f"'{r}'" for r in tagged)
        system_content = (
            PLAN_SYSTEM
            + f"\n\n## Repo scope\nThe user has tagged the following repos: {repo_list}. "
            "Restrict your exploration to these repos unless the question explicitly requires others."
        )

    messages = [{"role": "system", "content": system_content}] + state["messages"]
    try:
        response = call_llm(messages, tools=TOOL_SCHEMAS)
    except Exception as exc:
        logger.error(
            "plan_node call_llm failed hop=%d type=%s message=%s",
            hop,
            type(exc).__name__,
            exc,
        )
        raise
    assistant_msg: dict = response["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls") or []

    # If the model skipped tools on the very first hop (no prior exploration),
    # inject a mandatory bootstrapping call so synthesis never runs on zero evidence.
    if not tool_calls and hop == 1 and not state.get("tool_results"):
        assistant_msg = _forced_tool_call(state)
        logger.warning(
            "plan hop=%d no_tool_calls forced=%s",
            hop,
            assistant_msg["tool_calls"][0]["function"]["name"],
        )
    elif tool_calls:
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

    try:
        response = call_llm(observe_messages, json_mode=True)
    except Exception as exc:
        logger.error(
            "observe_node call_llm failed hop=%d type=%s message=%s",
            new_hop,
            type(exc).__name__,
            exc,
        )
        raise
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
