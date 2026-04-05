# Gitwhisper — Agentic Loop PRD

| Field | Value |
|---|---|
| Version | 0.2 — Draft |
| Author | Piyush Choudhari |
| Status | In Review |
| Last Updated | April 2026 |
| Framework | LangGraph + Gemma 4 (Ollama) |
| Scope | Agentic loop + Streamlit UI — no auth, no deployment |

---

## 1. Overview

This document specifies requirements for the **agentic loop only**: the core reasoning engine that sits between user input and the final answer.

The loop runs on Gemma 4 via Ollama (local), uses LangGraph to model the state machine, and reads code exclusively from a VM with cloned repos — no vector databases, no embeddings, no pre-indexing. All intelligence is in the agent's ability to plan, read, and reason.

> **Why no RAG?** Embeddings require index maintenance per repo update and introduce retrieval error. A reasoning loop over live file reads is more accurate for code — `grep` never hallucinates a function signature.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Answer natural language questions about any whitelisted repo accurately and with specific code references
- Complete most queries in under 5 tool-call iterations
- Never hallucinate file paths, function names, or code that does not exist in the repos
- Enforce whitelist boundaries strictly — the agent must never access repos not in the config
- Degrade gracefully when `MAX_HOPS` is reached: return a partial answer, not a failure
- Support multi-turn conversation using session history across the loop

### 2.2 Non-Goals

- UI, authentication, or session management
- Repo cloning, syncing, or VM infrastructure setup
- Code execution or write access to any file
- Answering questions about repos not in the whitelist
- Real-time repo updates within a session — cloned snapshot is the source of truth

---

## 3. Architecture

### 3.1 Design Philosophy

The agentic loop is a pure read-and-reason system. No retrieval index is built. The agent navigates the codebase the way a senior engineer would on a new machine: check what repos exist, scan directory structure, search for relevant terms, read specific files.

This is slower than vector search but produces grounded, accurate, verifiable answers.

### 3.2 Loop Overview

```
User query
    │
    ▼
┌─────────────┐
│  plan_node  │  ← Gemma 4 decides which tool to call
└──────┬──────┘
       │ tool call
       ▼
┌─────────────┐
│  tool_node  │  ← LangGraph ToolNode executes the call
└──────┬──────┘
       │ tool result
       ▼
┌──────────────┐
│ observe_node │  ← Gemma 4 reviews result, decides: loop or synthesize
└──────┬───────┘
       │
  ┌────┴─────┐
  │          │
loop      synthesize
  │          │
  └──►plan   ▼
       ┌──────────────────┐
       │  synthesize_node │  ← Gemma 4 writes grounded answer
       └──────────────────┘
```

### 3.3 Loop Phases

| Phase | LangGraph Node | Responsibility |
|---|---|---|
| Plan | `plan_node` | Gemma 4 reasons about the query and decides which tool to call next. Receives full state: query, history, tool_results so far, hop_count. |
| Dispatch | `tool_node` | LangGraph `ToolNode` executes the selected tool call. Returns structured `ToolResult` appended to state. |
| Observe | `observe_node` | Gemma 4 reviews the new tool result in context. Updates plan. Decides: loop again or synthesize. |
| Synthesize | `synthesize_node` | Gemma 4 writes the final answer grounded in `tool_results`. Formats code blocks. Adds file path citations. |

### 3.4 LangGraph Graph Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

graph = StateGraph(AgentState)

graph.add_node("plan",       plan_node)
graph.add_node("tools",      ToolNode(tools=[
                                 list_repos,
                                 get_file_tree,
                                 read_file,
                                 search_code,
                                 get_repo_metadata,
                             ]))
graph.add_node("observe",    observe_node)
graph.add_node("synthesize", synthesize_node)

graph.set_entry_point("plan")
graph.add_edge("plan",    "tools")
graph.add_edge("tools",   "observe")

graph.add_conditional_edges("observe", route_or_synthesize, {
    "loop":       "plan",
    "synthesize": "synthesize",
})

graph.add_edge("synthesize", END)
```

---

## 4. Tool Specifications

All tools are Python functions registered with LangGraph's `ToolNode`. They access the cloned repo VM via the local filesystem. The tool layer enforces whitelist boundaries independently of the prompt — no path outside the whitelist is accessible even if Gemma 4 requests it.

### 4.1 `list_repos`

```python
def list_repos() -> list[str]:
    """Return the names of all whitelisted repos."""
```

- No arguments
- Reads from `config.json` whitelist
- Always called first if agent needs to establish repo scope
- Returns repo names only, not paths

### 4.2 `get_file_tree`

```python
def get_file_tree(repo: str, path: str = "/") -> FileTree:
    """List directories and files in a repo at a given path."""
```

- Performs `os.walk` on the cloned repo directory
- Agent should call this before `read_file` to avoid blind reads
- Returns a nested structure of dirs and file names with sizes
- Rejects any `repo` not in the whitelist at the filesystem level

### 4.3 `read_file`

```python
def read_file(repo: str, path: str) -> str:
    """Return the raw contents of a file in a whitelisted repo."""
```

- Returns file content as a string
- Files over `MAX_FILE_SIZE` (200 KB) are truncated with a notice
- Agent should record `(repo, path)` in `files_read` to avoid redundant calls
- Rejects paths outside the whitelisted repo root

### 4.4 `search_code`

```python
def search_code(query: str, repos: list[str] = []) -> list[SearchResult]:
    """Run a grep-style search across one or more whitelisted repos."""
```

- Runs `grep -rn` under the hood across cloned repo dirs
- `repos` defaults to all whitelisted repos if empty
- Returns a list of `{ repo, file, line_number, snippet }` objects
- Fast — use early in the loop to narrow scope before reading full files

### 4.5 `get_repo_metadata`

```python
def get_repo_metadata(repo: str) -> RepoMeta:
    """Return high-level metadata about a repo."""
```

- Returns: last commit date, primary language, file count, README first 500 chars
- Cheap call — no file reading
- Useful for recruiter-facing queries like "what is this project about?"

---

## 5. Agent State Schema

LangGraph manages state as a typed dictionary passed between nodes. All fields are defined at session start.

```python
class AgentState(TypedDict):
    query:        str                    # Original user message. Never mutated.
    history:      list[Message]          # Full conversation turns for multi-turn context.
    tool_results: list[ToolResult]       # Accumulated results. Appended, never overwritten.
    plan:         str                    # Agent's current reasoning. Overwritten each iteration.
    hop_count:    int                    # Iterations so far. Checked against MAX_HOPS.
    answer:       str | None             # Populated only when loop is complete.
    files_read:   set[tuple[str, str]]   # (repo, path) pairs already read. Prevents redundancy.
```

### 5.1 State Initialisation

```python
initial_state: AgentState = {
    "query":        user_message,
    "history":      conversation_history,
    "tool_results": [],
    "plan":         "",
    "hop_count":    0,
    "answer":       None,
    "files_read":   set(),
}
```

---

## 6. Hard Constraints

| Constraint | Value | Rationale |
|---|---|---|
| `MAX_HOPS` | 10 | Hard cap on tool-call iterations per query. Agent synthesizes with whatever it has if hit. |
| `MAX_FILE_SIZE` | 200 KB | Files larger than this are truncated. Prevents context overflow. |
| `MAX_FILES_PER_HOP` | 3 | Agent may read at most 3 files per iteration. Forces prioritisation. |
| `CONTEXT_BUDGET` | 80K tokens | Soft limit on total `tool_results` context passed to Gemma 4. Oldest results are summarised if exceeded. |
| `WHITELIST_ENFORCEMENT` | Strict | Tool layer rejects non-whitelisted paths at the filesystem level, not just in the prompt. |
| `TIMEOUT` | 60s | Wall-clock timeout per query. Agent returns a partial answer with a note if exceeded. |

---

## 7. Routing Logic

The `observe_node` decides after each tool call whether to loop or synthesize. This is the only branching point in the graph.

```python
def route_or_synthesize(state: AgentState) -> str:
    if state["hop_count"] >= MAX_HOPS:
        return "synthesize"
    if state["answer"] is not None:
        return "synthesize"
    return "loop"
```

Gemma 4 signals readiness to synthesize by populating the `answer` field in the observe step with a non-None placeholder. The actual answer text is written by `synthesize_node`.

### 7.1 Observe Node Prompt Structure

```
You are analyzing the results of a tool call made to answer this query:
<query>{state.query}</query>

Tool results so far:
<tool_results>{state.tool_results}</tool_results>

Current hop: {state.hop_count} / {MAX_HOPS}

Decide:
- If you have enough information to answer the query fully, set answer = "READY".
- If you need more information, describe your next tool call in the plan field.
- Do not read files you have already read: {state.files_read}
```

---

## 8. Synthesis Requirements

The `synthesize_node` must produce answers that meet the following requirements:

- **Grounded**: every code claim must reference a specific file and line range from `tool_results`
- **Formatted**: code snippets in fenced code blocks with language tag
- **Cited**: file paths cited as `repo/path/to/file.py:L42` inline
- **Honest**: if the agent did not find conclusive evidence, say so explicitly rather than guessing
- **Concise**: answer the query directly, do not summarise every file read

### 8.1 Synthesize Node Prompt Structure

```
You are answering a question about a developer's GitHub repositories.
You have read the following files and search results:
<tool_results>{state.tool_results}</tool_results>

Query: {state.query}

Answer the query directly. Cite specific files and line numbers.
Use fenced code blocks for all code. If evidence is incomplete, say so.
Do not mention the tool call process — just answer the question.
```

---

## 9. Failure Modes and Handling

| Failure | Behaviour |
|---|---|
| `MAX_HOPS` reached | Synthesize with available context. Prefix answer with: *"Reached max search depth. Answer based on partial results."* |
| Tool call raises exception | Log error, skip result, continue loop. Decrement remaining hops by 1. |
| File too large | Truncate at `MAX_FILE_SIZE`. Append notice to result: *"[truncated at 200 KB]"* |
| Gemma 4 requests non-whitelisted repo | Tool layer raises `WhitelistViolation`. Log and return error result. Do not expose the whitelist contents in the error message. |
| Context budget exceeded | Summarise oldest `tool_results` entries in place before passing state to Gemma 4. |
| Timeout | Return partial answer with note. Log `hop_count` reached at timeout. |

---

## 10. Open Questions

- **VM sync strategy**: how often are cloned repos updated? Per session? Nightly cron? On-demand pull before each query?
- **Context compression**: when `CONTEXT_BUDGET` is approached, should oldest results be dropped or summarised by a separate cheap model call?
- **Multi-repo queries**: if a query spans 3+ repos, the hop budget gets consumed fast. Consider a pre-planning step that ranks repos by relevance before the main loop.
- **Streaming**: should the synthesize step stream tokens to the UI, or return the full answer at once? Streaming is better UX but requires LangGraph streaming mode setup.
- **Gemma 4 tool-calling reliability**: needs empirical testing. If structured tool call output is unreliable, may need a JSON-extraction fallback in `plan_node`.