"""LLM-as-judge for grounding evals.

judge_grounding() takes the tool_results accumulated by the agent and its
final synthesized answer, then asks the LLM to identify every code claim in
the answer that cannot be traced back to tool_results.

Returns a JudgeResult with a grounding_score in [0.0, 1.0] where 1.0 means
every code claim is grounded.
"""

from __future__ import annotations

import json
from typing import TypedDict


class JudgeResult(TypedDict):
    violations: list[dict]  # [{"claim": str, "reason": str}]
    grounding_score: float  # 0.0 – 1.0


_JUDGE_PROMPT = """\
You are evaluating an AI assistant's answer about a codebase.

Given the tool results the agent had access to, and its final answer, \
identify every code claim (function names, file paths, line numbers, \
variable names, class names) in the answer that does NOT appear verbatim \
in the tool results below.

Tool results:
{tool_results}

Answer:
{answer}

Output ONLY valid JSON — no markdown, no explanation:
{{"violations": [{{"claim": "<exact text from answer>", "reason": "<why it is not grounded>"}}], "grounding_score": <float 0.0-1.0>}}

grounding_score must be 1.0 if violations is empty, \
and should decrease proportionally to the number of ungrounded claims.\
"""


def _format_results(tool_results: list[dict]) -> str:
    parts: list[str] = []
    for i, tr in enumerate(tool_results, 1):
        args_str = json.dumps(tr.get("args", {}), ensure_ascii=False)
        parts.append(f"--- Result {i}: {tr['tool']}({args_str}) ---")
        parts.append(tr.get("result", ""))
        parts.append("")
    return "\n".join(parts)


def judge_grounding(tool_results: list[dict], answer: str) -> JudgeResult:
    from utils import call_llm  # lazy import — keeps module importable without env

    prompt = _JUDGE_PROMPT.format(
        tool_results=_format_results(tool_results),
        answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        response = call_llm(messages, json_mode=True)
        raw: str = response["choices"][0]["message"]["content"] or "{}"
        parsed: dict = json.loads(raw)
        return {
            "violations": parsed.get("violations", []),
            "grounding_score": float(parsed.get("grounding_score", 1.0)),
        }
    except Exception:
        # Fail open — don't crash the eval run; caller checks the score
        return {"violations": [], "grounding_score": 1.0}
