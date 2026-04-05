from __future__ import annotations

import json
import operator
import re

import streamlit as st

from config import DEFAULT_MODEL
from utils import call_llm_stream, ensure_repos, get_whitelist
from graph import compiled_graph
from logger import get_logger
from nodes import build_synthesize_messages
from state import AgentState

logger = get_logger("app")


@st.cache_resource
def _ensure_repos_once() -> dict[str, str]:
    return ensure_repos()


st.set_page_config(page_title="askmycode", page_icon="🔍", layout="centered")
st.title("🔍 Askmycode")
st.caption("Ask questions about your code repositories.")

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.markdown(
        """
    ### [Piyush Choudhari](https://www.piyushchoudhari.me)
    *AI & Backend Engineer*
    
    Specialized in Agentic Systems, LLM Evaluation, and High-Performance Backend Design.
    
    [GitHub](https://github.com/piyushchoudhari) | [LinkedIn](https://linkedin.com/in/piyush-choudhari) | [X](https://x.com/piyush_choudhari) | [Email](mailto:piyush@piyushchoudhari.me)
    
    ---
    **Stack**
    - Python, TypeScript, C++, Node.js
    - Agentic Systems, LLM Evaluation
    - Research, RAG & Search
    
    ---
    [View Source](https://github.com/piyushchoudhari/askmycode)
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.header("Configuration")
    try:
        st.write(f"**Model:** `{DEFAULT_MODEL}`")

        clone_status = _ensure_repos_once()
        for repo_name, status in clone_status.items():
            if status.startswith("error:"):
                st.error(f"`{repo_name}`: {status}")
            elif status == "local not found":
                st.warning(f"`{repo_name}`: local path not found")

        whitelist = get_whitelist()
        if whitelist:
            st.write("**Available repos:**")
            for name in sorted(whitelist.keys()):
                st.write(f"- `{name}`")
        else:
            st.warning(
                "No repos found. Add directories to `repos/` or populate "
                "`config.json` → `repos` to get started."
            )
    except Exception as exc:
        st.error(f"Could not load config: {exc}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.display_messages = []
        st.session_state.history = []
        st.rerun()

if user_input := st.chat_input("Ask a question about your repos…"):
    logger.info("query length=%d", len(user_input))
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    initial_state: AgentState = {
        "messages": st.session_state.history
        + [{"role": "user", "content": user_input}],
        "tool_results": [],
        "files_read": [],
        "query": user_input,
        "hop_count": 0,
        "answer": None,
    }

    with st.chat_message("assistant"):
        answer = "(No answer produced.)"
        accumulated: AgentState = {**initial_state}

        try:
            with st.status("Thinking…", expanded=True) as status:
                for update in compiled_graph.stream(
                    initial_state, stream_mode="updates"
                ):
                    for node_name, node_output in update.items():
                        if node_name == "tools_node":
                            new_results = node_output.get("tool_results", [])
                            for tr in new_results:
                                args_str = json.dumps(tr["args"], ensure_ascii=False)
                                st.write(f"**`{tr['tool']}`** `{args_str}`")
                            # merge list fields
                            accumulated["tool_results"] = operator.add(
                                accumulated.get("tool_results") or [],
                                new_results,
                            )
                            accumulated["messages"] = operator.add(
                                accumulated.get("messages") or [],
                                node_output.get("messages", []),
                            )
                        elif node_name == "observe":
                            hop = node_output.get("hop_count", accumulated["hop_count"])
                            accumulated["hop_count"] = hop
                            accumulated["files_read"] = node_output.get(
                                "files_read", accumulated.get("files_read")
                            )
                            if node_output.get("answer") is not None:
                                accumulated["answer"] = node_output["answer"]
                            status.update(label=f"Hop {hop}…")
                        elif node_name == "plan":
                            accumulated["messages"] = operator.add(
                                accumulated.get("messages") or [],
                                node_output.get("messages", []),
                            )

                hop_count = accumulated["hop_count"]
                tool_results = accumulated.get("tool_results") or []
                status.update(label="Synthesizing…")
                # state="complete" is set after synthesis finishes

            synth_messages, prefix = build_synthesize_messages(accumulated)
            logger.info(
                "synthesizing hops=%d tool_calls=%d",
                hop_count,
                len(tool_results),
            )
            raw = "".join(call_llm_stream(synth_messages))

            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if think_match:
                think_content = think_match.group(1).strip()
                if think_content:
                    with st.expander("Reasoning", expanded=False):
                        st.markdown(think_content)

            if prefix:
                st.markdown(prefix)
            st.markdown(clean)
            answer = prefix + clean
            logger.info("query_done answer_chars=%d", len(answer))

            status.update(
                label=f"Done — {hop_count} hop(s), {len(tool_results)} tool call(s)",
                state="complete",
                expanded=True,
            )

            if tool_results:
                with st.expander(
                    f"{hop_count} hop(s), {len(tool_results)} tool call(s)",
                    expanded=False,
                ):
                    for i, tr in enumerate(tool_results, 1):
                        args_str = json.dumps(tr["args"], ensure_ascii=False)
                        st.write(f"**{i}. `{tr['tool']}`** `{args_str}`")
                        with st.expander("Result", expanded=False):
                            st.code(
                                tr["result"][:2000]
                                + ("…" if len(tr["result"]) > 2000 else "")
                            )

        except Exception as exc:
            answer = f"An error occurred: {exc}"
            logger.error("query_failed error=%s message=%s", type(exc).__name__, exc)
            st.error(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": answer})
