from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.json"
REPOS_ROOT: Path = PROJECT_ROOT / "repos"

LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE: Path = LOG_DIR / "askmycode.log"
LOG_NAME: str = "askmycode"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT: int = 3

OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_MODEL: str = "qwen/qwen3.6-plus:free"

MAX_HOPS: int = 10
MAX_FILES_PER_HOP: int = 3
CONTEXT_BUDGET_CHARS: int = 320_000  # ≈ 80 K tokens at 4 chars/token

MAX_FILE_SIZE: int = 200 * 1024  # 200 KB in bytes

TIMEOUT_SECONDS: int = 60

GREP_MAX_RESULTS: int = 50
GREP_TIMEOUT_SECONDS: int = 15

CLONE_TIMEOUT_SECONDS: int = 120


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return json.load(f)
