from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from config import MAX_HOPS
from nodes import observe_node, plan_node, synthesize_node, tools_node
from state import AgentState


def should_continue(state: AgentState) -> Literal["tools_node", "synthesize"]:
    last = state["messages"][-1]
    tool_calls = last.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        return "tools_node"
    return "synthesize"


def route_from_observe(state: AgentState) -> Literal["plan", "synthesize"]:
    if state.get("answer") is not None or state["hop_count"] >= MAX_HOPS:
        return "synthesize"
    return "plan"


_builder = StateGraph(AgentState)

_builder.add_node("plan", plan_node)
_builder.add_node("tools_node", tools_node)
_builder.add_node("observe", observe_node)
_builder.add_node("synthesize", synthesize_node)

_builder.set_entry_point("plan")

_builder.add_conditional_edges(
    "plan",
    should_continue,
    {"tools_node": "tools_node", "synthesize": "synthesize"},
)

_builder.add_edge("tools_node", "observe")

_builder.add_conditional_edges(
    "observe",
    route_from_observe,
    {"plan": "plan", "synthesize": "synthesize"},
)

_builder.add_edge("synthesize", END)

compiled_graph = _builder.compile()
