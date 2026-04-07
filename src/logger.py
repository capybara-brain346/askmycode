import logging
import sys
from logging.handlers import RotatingFileHandler

from config import (
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_FILE,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_NAME,
)

LOGGER_NAME = LOG_NAME

_RESET = "\033[0m"
_COLOURS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[35m",  # magenta
}


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, _RESET)
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


def _resolve_level() -> int:
    level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(level, int):
        level = logging.INFO
    return level


def _setup_logger() -> logging.Logger:
    root = logging.getLogger(LOGGER_NAME)
    if root.handlers:
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False

    LOG_DIR.mkdir(exist_ok=True)

    terminal_level = _resolve_level()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(terminal_level)
    stream_handler.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    return root


logger = _setup_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logger
