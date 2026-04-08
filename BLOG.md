# askmycode: How I Built an Agent That Reads My Code Like a Senior Engineer

> *Ask a question. Watch it grep, navigate, and reason its way to a precise, cited answer — without a single vector embedding.*

---

## 1. Introduction

Every developer has been in this situation: a colleague asks "how does auth work in that service?" and you spend five minutes digging through your own codebase to remember. Now multiply that by every recruiter, collaborator, or future-you who wants to understand a repo they haven't touched in months.

I built **askmycode** to solve this. It's a Streamlit chat app where you can ask natural-language questions about any of your GitHub repositories and get back a precise, source-cited answer — file paths, line numbers, actual code snippets, no hand-waving.

```
How is rate limiting implemented in @capynodes-backend?
```

```
Rate limiting is enforced in capynodes-backend via the check_rate_limit
function in middleware/rate_limit.py:L14-38. It uses a sliding-window
counter stored in Redis...
```

What makes this interesting is *how* it works under the hood. There's no search index, no vector database, no pre-processing step. The agent reads your code the same way you would on a fresh machine: it scans directory trees, greps for symbols, and opens files until it has enough to answer. This post walks through every design decision.

---

## 2. Indexing vs. Agentic Retrieval

The conventional approach to "chat with your codebase" is **Retrieval-Augmented Generation (RAG)**:

1. Chunk every file into small pieces.
2. Embed each chunk with a model like `text-embedding-ada-002`.
3. Store embeddings in a vector database (Pinecone, Chroma, Weaviate, …).
4. At query time, embed the question, retrieve the nearest chunks, and pass them to an LLM.

RAG is fast and scales to massive corpora. But for code, it has real problems:

| Problem | Why it matters for code |
|---|---|
| **Index staleness** | Every `git push` potentially invalidates cached embeddings. You need a sync pipeline or your answers drift from reality. |
| **Chunking is lossy** | A function split across a chunk boundary loses context. A class split from its imports becomes ambiguous. |
| **Retrieval errors are silent** | If the relevant chunk ranked 6th instead of 1st, the LLM never sees it and may hallucinate a plausible-but-wrong answer. |
| **Fixed granularity** | Embeddings can't decide "I need the whole file" vs. "I only need lines 20-35". |

**Agentic retrieval** takes a different approach. Instead of pre-building an index, the agent *plans and acts* at query time:

```
User query
    ↓
Agent decides: "I should search for the symbol first"
    ↓
search_code("check_rate_limit") → file: middleware/rate_limit.py, line 14
    ↓
Agent decides: "I need to read that file"
    ↓
read_file_tool(repo, "middleware/rate_limit.py") → actual source code
    ↓
Agent decides: "I have enough to answer"
    ↓
Synthesize grounded answer
```

The key insight from the `AGENTIC_LOOP.md` design doc:

> *`grep` never hallucinates a function signature.*

Every tool call returns ground-truth data from the actual file system. The agent can't invent a result — it either finds the code or it doesn't. This makes answers verifiable and the failure mode honest ("I couldn't find evidence of X") rather than subtly wrong.

The trade-off is latency. A RAG system retrieves in milliseconds; an agentic loop may take 3-10 tool calls. For code Q&A, where correctness matters more than speed, this trade-off is almost always worth making.

---

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Qwen3-32B via Groq | Fast inference, strong tool-calling, excellent code reasoning |
| **Agentic framework** | LangGraph | First-class state machine primitives, conditional edges, streaming |
| **UI** | Streamlit | Zero-boilerplate chat UI with real-time status updates |
| **Code search** | `grep -rn -E` (subprocess) | Fast, dependency-free, exact-match on live files |
| **Repo management** | `git clone` (subprocess) | Snapshots repos locally; no remote API calls at query time |
| **Config** | `config.json` | Simple `{name: owner/repo}` whitelist |
| **Logging** | Python `logging` (rotating file) | Structured key=value pairs, 5 MB rotating log |
| **Evals** | pytest + LLM-as-judge | Tool sequencing and grounding checks |

The stack is deliberately minimal. No ORM, no message queue, no container orchestration — just Python, a Groq API key, and git. The repo runs locally with `uv sync && uv run streamlit run src/app.py`.

Repos are configured in `config.json` and auto-cloned on first startup:

```json
{
  "repos": {
    "capybaradb": "capybara-brain346/capybaradb",
    "knowflow":   "capybara-brain346/knowflow"
  }
}
```

Any directory already present under `repos/` is also picked up automatically, so local-only repos work too.

---

## 4. LangGraph Flow

The agent is modelled as a state machine with three nodes and two conditional edges. Here's the actual graph definition from `src/graph.py`:

```python
from langgraph.graph import END, StateGraph

graph = StateGraph(AgentState)

graph.add_node("plan",       plan_node)
graph.add_node("tools_node", tools_node)
graph.add_node("observe",    observe_node)

graph.set_entry_point("plan")

# plan → tools_node  (if LLM emitted a tool call)
# plan → END         (if LLM answered directly)
graph.add_conditional_edges("plan", should_continue,
    {"tools_node": "tools_node", END: END})

graph.add_edge("tools_node", "observe")

# observe → plan     (if more evidence needed)
# observe → END      (if ready to synthesize, or MAX_HOPS reached)
graph.add_conditional_edges("observe", route_from_observe,
    {"plan": "plan", END: END})
```

Visually:

```
          ┌─────────────────────────────────────────┐
          │                                         │
 query ──►│ plan_node ──► tools_node ──► observe_node │──► END (synthesize)
          │     ▲                           │       │
          │     └────────── loop ───────────┘       │
          └─────────────────────────────────────────┘
```

### The State Object

All data flows through a single `AgentState` TypedDict, appended to (never overwritten) by each node:

```python
class AgentState(TypedDict):
    messages:      Annotated[list[dict], operator.add]  # full conversation + tool messages
    tool_results:  Annotated[list[ToolResult], operator.add]  # accumulated evidence
    files_read:    list[list[str]]   # (repo, path) pairs — deduplication guard
    query:         str               # original question, never mutated
    hop_count:     int               # iteration counter, checked against MAX_HOPS=10
    answer:        str | None        # READY sentinel set by observe_node
    tagged_repos:  list[str]         # @-mentioned repos for scoping
```

### Node Responsibilities

**`plan_node`** — Calls the LLM with the full system prompt and accumulated message history. The LLM decides which tool to call next (or answers directly if it already has enough). The system prompt encodes the exploration strategy:

- *Go targeted first*: `search_code` before `get_file_tree` before `read_file_tool`.
- *Use regex alternation*: `jwt|token|OAuth|authenticate` to cover synonyms in one call.
- *Don't re-read*: the `files_read` list is injected into context so the agent skips already-read files.

**`tools_node`** — Executes the tool calls emitted by `plan_node`. Each tool is dispatched from the `TOOL_FUNCTIONS` registry. Results are appended to `tool_results` and returned as `role: tool` messages for the LLM's next turn. Errors (whitelist violations, file-not-found, grep timeouts) are caught and returned as error strings — the agent sees them and can adapt.

**`observe_node`** — This is the decision node. It asks the LLM a focused question in JSON mode: *"Do you have enough to answer, or do you need more?"* The response is a simple `{"decision": "loop" | "synthesize", "reasoning": "..."}`. If `hop_count >= MAX_HOPS`, it forces `synthesize` regardless of the LLM's preference.

The routing function:

```python
def route_from_observe(state: AgentState) -> Literal["plan", "__end__"]:
    if state.get("answer") is not None or state["hop_count"] >= MAX_HOPS:
        return END
    return "plan"
```

**Synthesis** — After the loop ends, `build_synthesize_messages` constructs a final prompt with all accumulated `tool_results` and the original query. The LLM writes a grounded answer, citing specific files and line numbers. This step happens outside the graph in `app.py` using a streaming call so the user sees tokens arrive in real time.

---

## 5. Tool Details

All tools are plain Python functions that access cloned repos on the local filesystem. The tool layer enforces whitelist boundaries independently of the LLM — no amount of prompt manipulation can make a tool read outside the config-defined repos.

### Security: Double Validation

Every tool validates both the repo name *and* the resolved path:

```python
def _validate_repo(repo: str) -> Path:
    whitelist = get_whitelist()
    if repo not in whitelist:
        raise WhitelistViolation(f"Repo '{repo}' is not in the whitelist.")
    return whitelist[repo].resolve()

def _validate_path(repo_root: Path, path: str) -> Path:
    candidate = (repo_root / path.lstrip("/")).resolve()
    if not candidate.is_relative_to(repo_root):
        raise WhitelistViolation("Path resolves outside the repo root.")
    return candidate
```

The `is_relative_to` check defeats path traversal attacks (`../../etc/passwd`). The whitelist check means even a jailbroken LLM prompt can't read an unconfigured repo.

### Tool Reference

| Tool | Signature | Purpose |
|---|---|---|
| `list_repos` | `() → list[str]` | Returns all whitelisted repo names. Called first when the agent needs to establish scope. |
| `get_file_tree` | `(repo, path="/") → dict` | Lists immediate files and dirs at a path. Returns name + size for files. |
| `read_file_tool` | `(repo, path) → str` | Returns raw file content. Truncates at 200 KB with a `[truncated at 200 KB]` notice. |
| `search_code` | `(query, repos=[]) → list[dict]` | Runs `grep -rn -E` across one or more repos. Returns `{repo, file, line_number, snippet}` objects, capped at 50 results. |
| `get_repo_metadata` | `(repo) → dict` | Returns file count, last-modified timestamp, top file extensions, and the first 500 chars of the README. Fast and cheap — no full file reads. |

### How the Agent Uses Them

A typical query like *"How does auth work in capynodes-backend?"* follows this pattern:

1. **`search_code`** with `jwt|token|OAuth|authenticate|login` → finds `middleware/auth_jwt.py:L14`
2. **`read_file_tool`** on that file → reads the actual implementation
3. **observe** decides: "I have the definition. Synthesize."

A high-level overview query like *"What is capybaradb?"* follows a different pattern:

1. **`get_repo_metadata`** → gets file count, README excerpt, primary language
2. **observe** decides: "README answers the question. Synthesize."

The `search_code` tool is the workhorse. Because it takes a full extended regex, the agent can cover many synonyms in a single call, keeping the hop count low.

---

## 6. Evals and Safeguards

Shipping an LLM-based system without evals is like shipping code without tests: you might be fine, but you won't know when you break something. askmycode has two eval suites under `tests/evals/`.

### T1: Tool Sequencing Evals

These tests assert that the agent calls tools in a *sensible order* — they don't care about the content of answers, only about the structural properties of the tool call sequence.

Tools are patched with instant stubs that return canned data, so only the LLM's planning decisions are under test. This makes the tests fast and deterministic on the infrastructure side.

| Test | Query | Assertion |
|---|---|---|
| T1-01 | "What repos are available?" | `list_repos` is called, ≤ 2 hops |
| T1-02 | "What does the auth module do?" | `search_code` appears **before** `read_file_tool` |
| T1-03 | "What is capybaradb about?" | First call is a cheap orientation tool (not a blind file read) |
| T1-04 | "Where is rate limiting implemented?" | `search_code` is called |
| T1-05 | "Compare vector storage across repos" | `list_repos` is the very first call |

T1-02 is particularly important. A naive agent might blindly call `read_file_tool` on a guessed path. The correct strategy is to search first, then read what the search points to. The test enforces this:

```python
if has_read:
    first_search = next(i for i, t in enumerate(seq) if t == "search_code")
    first_read   = next(i for i, t in enumerate(seq) if t == "read_file_tool")
    assert first_search < first_read
```

### G2: Grounding Evals

These tests use an **LLM-as-judge** pattern. Given a `(tool_results, answer)` pair, a judge LLM scores the answer on grounding: is every claim in the answer traceable back to something in `tool_results`?

| Test | Scenario | Expectation |
|---|---|---|
| G2-01 | Answer correctly cites real file content | Grounding score ≥ 0.95 |
| G2-02 | Empty tool results, answer invents a function | At least one violation flagged |
| G2-03 | Answer references content beyond a truncation point | Violation flagged |
| G2-04 | Answer attributes code to the wrong repo | Violation flagged |
| G2-05 | Answer cites `:L999` when the actual line is `L5` | Violation flagged |

G2-02 through G2-05 are **negative tests** — they construct adversarial inputs and assert the judge catches the problem. This gives confidence that the grounding evaluator itself is calibrated, not just rubber-stamping everything as correct.

### E2E: Golden Set

The E2E suite runs the full agent against 20 golden queries across five categories (function lookup, dependency tracing, repo overview, implementation location, feature presence). Each golden case specifies expected keywords that must appear in the answer:

```json
{
    "id": "E-08",
    "category": "dependency_tracing",
    "query": "Where is JWT authentication implemented in knowflow?",
    "repo_scope": "knowflow",
    "expected_keywords": ["jwt", "auth_service"]
}
```

Results are logged to `logs/eval_results.csv` so you can track pass rate across runs.

### Hard Constraints

Beyond evals, several hard limits are enforced in code:

| Constraint | Value | Effect |
|---|---|---|
| `MAX_HOPS` | 10 | Loop hard-stops; agent synthesizes with whatever it has |
| `MAX_FILE_SIZE` | 200 KB | Large files truncated with a notice |
| `CONTEXT_BUDGET_CHARS` | 320,000 (~80K tokens) | Oldest tool results dropped from context if exceeded |
| `TIMEOUT_SECONDS` | 60 | Per-query wall-clock limit |
| `GREP_MAX_RESULTS` | 50 | Search results capped to prevent context flooding |
| Whitelist enforcement | Strict | Path traversal blocked at the filesystem level |

---

## 7. Conclusion

askmycode is a case study in choosing the right abstraction for the problem.

RAG is the obvious choice for "chat with documents" because documents are static, chunking is straightforward, and vector similarity is a reasonable proxy for relevance. Code is different. Functions call other functions. Auth logic lives across three files. A module's purpose is only clear when you understand its callers. These relationships are navigational, not distributional — and navigational retrieval is what agents are good at.

The LangGraph state machine keeps the agent honest: each tool call either advances toward the answer or triggers a `synthesize` decision. The whitelist layer keeps it safe: no prompt injection can exfiltrate files outside the config. The eval suite keeps it calibrated: grounding and sequencing are tested, not assumed.

The entire thing runs locally with one API key and no external databases. If you want to point it at your own repos, edit `config.json` and run:

```bash
uv sync
uv run streamlit run src/app.py
```

The source code is on GitHub: **[capybara-brain346/askmycode](https://github.com/capybara-brain346/askmycode)**

---

*Built by [Piyush Choudhari](https://www.piyushchoudhari.me) — AI & Backend Engineer specialising in agentic systems and LLM evaluation.*
