from __future__ import annotations

import re
import subprocess
from pathlib import Path

from collections.abc import Iterator

from groq import Groq

from config import (
    CLONE_TIMEOUT_SECONDS,
    DEFAULT_MODEL,
    GROQ_API_KEY,
    PROJECT_ROOT,
    REPOS_ROOT,
    TIMEOUT_SECONDS,
    load_config,
)
from logger import get_logger

logger = get_logger("utils")

_SHORT_FORM_RE = re.compile(r"^[\w][\w.\-]*/[\w][\w.\-]*(?:\.git)?$")


def _parse_repo_spec(name: str, spec: str) -> tuple[str | None, Path]:
    spec = spec.strip()
    if spec.lower().startswith(("https://", "http://")):
        return spec, REPOS_ROOT / name
    if spec.startswith("git@"):
        return spec, REPOS_ROOT / name
    p = Path(spec)
    local = p if p.is_absolute() else PROJECT_ROOT / p
    if local.exists():
        return None, local
    if _SHORT_FORM_RE.match(spec):
        return f"https://github.com/{spec}", REPOS_ROOT / name
    return None, local


def get_whitelist() -> dict[str, Path]:
    cfg = load_config()
    whitelist: dict[str, Path] = {}
    for name, spec in cfg.get("repos", {}).items():
        if spec:
            _, local = _parse_repo_spec(name, spec)
        else:
            local = REPOS_ROOT / name
        whitelist[name] = local
    if REPOS_ROOT.exists():
        for child in REPOS_ROOT.iterdir():
            if child.is_dir() and child.name not in whitelist:
                whitelist[child.name] = child
    return whitelist


def ensure_repos() -> dict[str, str]:
    cfg = load_config()
    results: dict[str, str] = {}
    for name, spec in cfg.get("repos", {}).items():
        spec = spec or ""
        try:
            clone_url, local = _parse_repo_spec(name, spec)
        except Exception as exc:
            results[name] = f"error: could not parse spec -- {exc}"
            continue
        if local.exists():
            results[name] = "exists"
            continue
        if clone_url is None:
            results[name] = "local not found"
            logger.warning("repo '%s': local path %s does not exist", name, local)
            continue
        logger.info("repo '%s': cloning %s", name, clone_url)
        try:
            proc = subprocess.run(
                ["git", "clone", clone_url, str(local)],
                timeout=CLONE_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                results[name] = "cloned"
                logger.info("repo '%s': cloned successfully", name)
            else:
                msg = (proc.stderr or proc.stdout).strip().splitlines()[-1]
                results[name] = f"error: {msg}"
                logger.error("repo '%s': clone failed -- %s", name, msg)
        except FileNotFoundError:
            results[name] = "error: git not found on PATH"
            logger.error("repo '%s': git executable not found", name)
        except subprocess.TimeoutExpired:
            results[name] = f"error: clone timed out after {CLONE_TIMEOUT_SECONDS} s"
            logger.error("repo '%s': clone timed out", name)
        except Exception as exc:
            results[name] = f"error: {exc}"
            logger.error("repo '%s': unexpected error -- %s", name, exc)
    return results


def _groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=GROQ_API_KEY, timeout=TIMEOUT_SECONDS, max_retries=5)


def _completion_to_dict(completion) -> dict:
    """Convert a Groq ChatCompletion object to the dict shape that nodes.py expects."""
    choices = []
    for choice in completion.choices:
        msg = choice.message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        msg_dict: dict = {"role": msg.role, "content": msg.content}
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        choices.append(
            {
                "message": msg_dict,
                "finish_reason": choice.finish_reason,
            }
        )
    usage = {}
    if completion.usage:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    return {"choices": choices, "usage": usage}


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    json_mode: bool = False,
) -> dict:
    model: str = DEFAULT_MODEL
    tool_names = [t["function"]["name"] for t in (tools or [])]
    logger.debug(
        "llm_call model=%s tool_count=%d json_mode=%s",
        model,
        len(tool_names),
        json_mode,
    )
    kwargs: dict = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = _groq_client().chat.completions.create(**kwargs)
    data = _completion_to_dict(completion)
    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    usage = data.get("usage", {})
    logger.debug(
        "llm_done finish=%s prompt_tokens=%s completion_tokens=%s",
        finish_reason,
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
    )
    return data


def call_llm_stream(messages: list[dict]) -> Iterator[str]:
    """Streaming variant for synthesis — yields text tokens as they arrive."""
    model: str = DEFAULT_MODEL
    logger.debug("llm_stream model=%s", model)
    stream = _groq_client().chat.completions.create(
        model=model, messages=messages, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
