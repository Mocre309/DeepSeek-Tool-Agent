"""Logging setup with a defence-in-depth secret redaction filter."""

from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)([A-Za-z0-9_\-]{6,})"),
)

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


class _SecretFilter(logging.Filter):
    """Never let an API key reach a log sink, even by accident."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern in _SECRET_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(r"\1***", redacted)
            else:
                redacted = pattern.sub("sk-***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_SecretFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party noise that nobody debugging an agent wants to read.
    for noisy in ("httpx", "httpcore", "urllib3", "ddgs", "primp", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
