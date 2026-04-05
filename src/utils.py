from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import httpx

from config import (
    CLONE_TIMEOUT_SECONDS,
    DEFAULT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
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


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    json_mode: bool = False,
) -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    cfg = load_config()
    model: str = cfg.get("model", DEFAULT_MODEL)
    payload: dict = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    tool_names = [t["function"]["name"] for t in (tools or [])]
    logger.debug(
        "LLM call | model=%s tools=%s json_mode=%s",
        model,
        tool_names or "none",
        json_mode,
    )
    max_retries = 5
    backoff = 5.0
    for attempt in range(max_retries):
        response = httpx.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=TIMEOUT_SECONDS
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else backoff * (2**attempt)
            logger.warning(
                "LLM 429 rate-limited | attempt=%d/%d | waiting %.1fs",
                attempt + 1,
                max_retries,
                wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue
        response.raise_for_status()
        break
    data = response.json()
    if "error" in data or "choices" not in data:
        err = data.get("error", data)
        err_code = err.get("code", "?") if isinstance(err, dict) else "?"
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.error("LLM error | code=%s message=%s", err_code, err_msg)
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
