"""Constants, configuration loading, and the core LLM call function."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .logger import get_logger

load_dotenv()

logger = get_logger("config")

MAX_HOPS: int = 10
MAX_FILE_SIZE: int = 200 * 1024  # 200 KB in bytes
MAX_FILES_PER_HOP: int = 3
CONTEXT_BUDGET_CHARS: int = 320_000  # ≈ 80 K tokens at 4 chars/token
TIMEOUT_SECONDS: int = 60

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_REPOS_ROOT = _PROJECT_ROOT / "repos"


def load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return json.load(f)


def get_whitelist() -> dict[str, Path]:
    cfg = load_config()
    whitelist: dict[str, Path] = {}
    for name, rel in cfg.get("repos", {}).items():
        if rel:
            # If a custom path is provided use it, otherwise derive from repos/
            p = Path(rel) if Path(rel).is_absolute() else _PROJECT_ROOT / rel
        else:
            p = _REPOS_ROOT / name
        whitelist[name] = p
    if _REPOS_ROOT.exists():
        for child in _REPOS_ROOT.iterdir():
            if child.is_dir() and child.name not in whitelist:
                whitelist[child.name] = child
    return whitelist


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    json_mode: bool = False,
) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    cfg = load_config()
    model: str = cfg.get("model", "google/gemma-4-31b-it")

    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    tool_names = [t["function"]["name"] for t in (tools or [])]
    logger.debug(
        "LLM call | model=%s tools=%s json_mode=%s",
        model,
        tool_names or "none",
        json_mode,
    )

    response = httpx.post(
        OPENROUTER_URL,
        json=payload,
        headers=headers,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    # OpenRouter can return a 200 with an error body instead of choices.
    if "error" in data or "choices" not in data:
        err = data.get("error", data)
        err_code = err.get("code", "?") if isinstance(err, dict) else "?"
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.error(
            "LLM error | code=%s message=%s",
            err_code,
            err_msg,
        )
        raise RuntimeError(f"OpenRouter error {err_code}: {err_msg}")

    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    usage = data.get("usage", {})
    logger.debug(
        "LLM done  | finish=%s prompt_tokens=%s completion_tokens=%s",
        finish_reason,
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
    )
    return data
