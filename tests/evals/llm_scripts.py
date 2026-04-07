"""Builders for scripted call_llm return values.

These replicate the exact dict shape produced by utils._completion_to_dict()
so that any code consuming call_llm()'s return value works unchanged.

Shapes
------
Tool call response  → choices[0].message has a non-empty tool_calls list
Direct response     → choices[0].message has content, no tool_calls
Observe response    → direct response whose content is a JSON string
                      matching the OBSERVE_SYSTEM schema
"""

from __future__ import annotations

import json
import uuid


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def make_tool_response(fn_name: str, args: dict | None = None) -> dict:
    """Build a call_llm return value that requests a single tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": _call_id(),
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(args or {}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }


def make_direct_response(content: str) -> dict:
    """Build a call_llm return value with no tool calls."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }


def make_observe_response(decision: str, reasoning: str = "") -> dict:
    """Build an observe-node call_llm return value (json_mode=True)."""
    payload = {
        "decision": decision,
        "reasoning": reasoning
        or f"Evidence is {'sufficient' if decision == 'synthesize' else 'incomplete'}.",
    }
    return make_direct_response(json.dumps(payload))
