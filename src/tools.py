import datetime
import os
import subprocess
from pathlib import Path
from typing import Callable

from config import GREP_MAX_RESULTS, GREP_TIMEOUT_SECONDS, MAX_FILE_SIZE
from state import WhitelistViolation
from utils import get_whitelist


def _validate_repo(repo: str) -> Path:
    whitelist = get_whitelist()
    if repo not in whitelist:
        raise WhitelistViolation(
            f"Repo '{repo}' is not in the whitelist. "
            "Use list_repos to see available repos."
        )
    root = whitelist[repo].resolve()
    if not root.exists():
        raise WhitelistViolation(
            f"Repo '{repo}' is whitelisted but its directory does not exist: {root}"
        )
    return root


def _validate_path(repo_root: Path, path: str) -> Path:
    rel = path.lstrip("/")
    candidate = (repo_root / rel).resolve()
    if not candidate.is_relative_to(repo_root):
        raise WhitelistViolation(
            f"Path '{path}' resolves outside the repo root. Access denied."
        )
    return candidate


def list_repos() -> list[str]:
    return sorted(get_whitelist().keys())


def get_file_tree(repo: str, path: str = "/") -> dict:
    root = _validate_repo(repo)
    target = _validate_path(root, path)

    if not target.exists():
        return {"error": f"Path '{path}' does not exist in repo '{repo}'."}

    dirs: list[str] = []
    files: list[dict] = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.is_dir():
                dirs.append(entry.name)
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                files.append({"name": entry.name, "size_bytes": size})
    except PermissionError as exc:
        return {"error": str(exc)}

    return {"dirs": dirs, "files": files}


def read_file_tool(repo: str, path: str) -> str:
    root = _validate_repo(repo)
    target = _validate_path(root, path)

    if not target.is_file():
        return f"[ERROR] '{path}' is not a file in repo '{repo}'."

    raw = target.read_bytes()
    truncated = False
    if len(raw) > MAX_FILE_SIZE:
        raw = raw[:MAX_FILE_SIZE]
        truncated = True

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return f"[ERROR] Could not decode '{path}' as text."

    if truncated:
        text += f"\n\n[truncated at {MAX_FILE_SIZE // 1024} KB]"
    return text


def search_code(query: str, repos: list[str] | None = None) -> list[dict]:
    if not repos:
        repos = list_repos()

    whitelist = get_whitelist()
    results: list[dict] = []

    for repo_name in repos:
        if repo_name not in whitelist:
            continue
        root = whitelist[repo_name].resolve()
        if not root.exists():
            continue

        try:
            proc = subprocess.run(
                ["grep", "-rn", "-E", "--include=*", "-I", query, str(root)],
                capture_output=True,
                text=True,
                timeout=GREP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            results.append({"repo": repo_name, "error": "grep timed out"})
            continue
        except FileNotFoundError:
            results.append(
                {"repo": repo_name, "error": "grep not found on this system"}
            )
            continue

        for line in proc.stdout.splitlines()[:GREP_MAX_RESULTS]:
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_abs, line_no, snippet = parts[0], parts[1], parts[2]
            try:
                file_rel = str(Path(file_abs).relative_to(root))
            except ValueError:
                file_rel = file_abs
            results.append(
                {
                    "repo": repo_name,
                    "file": file_rel,
                    "line_number": int(line_no),
                    "snippet": snippet.strip(),
                }
            )
        if len(results) >= GREP_MAX_RESULTS:
            break

    return results


def get_repo_metadata(repo: str) -> dict:
    root = _validate_repo(repo)

    ext_counts: dict[str, int] = {}
    file_count = 0
    last_mtime: float = 0.0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                st = fp.stat()
            except OSError:
                continue
            file_count += 1
            if st.st_mtime > last_mtime:
                last_mtime = st.st_mtime
            suffix = fp.suffix.lower() or "(no ext)"
            ext_counts[suffix] = ext_counts.get(suffix, 0) + 1

    top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    last_modified = (
        datetime.datetime.fromtimestamp(
            last_mtime, tz=datetime.timezone.utc
        ).isoformat()
        if last_mtime
        else "unknown"
    )

    readme_excerpt = ""
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = root / readme_name
        if readme_path.is_file():
            try:
                readme_excerpt = readme_path.read_text(errors="replace")[:500]
            except OSError:
                pass
            break

    return {
        "repo": repo,
        "file_count": file_count,
        "last_modified": last_modified,
        "top_extensions": dict(top_exts),
        "readme_excerpt": readme_excerpt,
    }


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_repos",
            "description": "Return the names of all whitelisted repos available for inspection.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_tree",
            "description": (
                "List the immediate directories and files inside a repo at a given path. "
                "Call this before read_file_tool to avoid blind reads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Whitelisted repo name."},
                    "path": {
                        "type": "string",
                        "description": "Directory path inside the repo. Defaults to '/' (root).",
                        "default": "/",
                    },
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_tool",
            "description": (
                "Return the raw text contents of a file in a whitelisted repo. "
                "Files over 200 KB are truncated. Record the (repo, path) to avoid re-reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Whitelisted repo name."},
                    "path": {
                        "type": "string",
                        "description": "File path inside the repo.",
                    },
                },
                "required": ["repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Run a grep-style search across one or more whitelisted repos. "
                "Returns up to 50 matches: [{repo, file, line_number, snippet}]. "
                "Use this early to narrow scope before reading full files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Extended-regex search string (grep -E). "
                            "Use | for alternation to cover synonyms in one shot: "
                            "e.g. `scrape|requests|httpx|fetch` or `jwt|token|authenticate`."
                        ),
                    },
                    "repos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Repo names to search. Defaults to all whitelisted repos.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_metadata",
            "description": (
                "Return high-level metadata about a repo: last modified date, "
                "file count, top file extensions, and README excerpt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Whitelisted repo name."},
                },
                "required": ["repo"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable] = {
    "list_repos": list_repos,
    "get_file_tree": get_file_tree,
    "read_file_tool": read_file_tool,
    "search_code": search_code,
    "get_repo_metadata": get_repo_metadata,
}
