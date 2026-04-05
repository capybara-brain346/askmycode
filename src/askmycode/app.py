from __future__ import annotations

import json

import streamlit as st

from askmycode.config import load_config
from askmycode.graph import compiled_graph
from askmycode.logger import get_logger
from askmycode.state import AgentState

logger = get_logger("app")

st.set_page_config(page_title="Gitwhisper", page_icon="🔍", layout="centered")
st.title("🔍 Gitwhisper")
st.caption("Ask questions about your code repositories.")

if "display_messages" not in st.session_state:
    st.session_state.display_messages: list[dict] = []

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("Configuration")
    try:
        cfg = load_config()
        st.write(f"**Model:** `{cfg.get('model', 'N/A')}`")
        repos = cfg.get("repos", {})
        from askmycode.config import get_whitelist

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
    logger.info("=== query received | length=%d chars ===", len(user_input))
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
        with st.spinner("Thinking…"):
            try:
                final_state: AgentState = compiled_graph.invoke(initial_state)
                answer: str = final_state.get("answer") or "(No answer produced.)"

                tool_results = final_state.get("tool_results", [])
                hop_count = final_state.get("hop_count", 0)
                logger.info(
                    "=== query done | hops=%d tool_calls=%d answer_chars=%d ===",
                    hop_count,
                    len(tool_results),
                    len(answer),
                )

                st.markdown(answer)

                with st.expander(
                    f"{hop_count} hop(s), {len(tool_results)} tool call(s)"
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
                logger.error("=== query failed: %s: %s ===", type(exc).__name__, exc)
                st.error(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": answer})
