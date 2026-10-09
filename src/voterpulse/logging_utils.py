"""Structured logging: ``timestamp level logger event key=value ...`` lines.

Usage::

    log = get_logger(__name__)
    log.info("feed_fetched", extra={"fields": {"feed": "jobs", "entries": 42}})
"""

from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False


class KeyValueFormatter(logging.Formatter):
    """Formats records as one greppable line with key=value context fields."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [ts, record.levelname, record.name, record.getMessage()]
        fields: dict[str, Any] = getattr(record, "fields", None) or {}
        for key, value in fields.items():
            text = str(value).replace("\n", " ")
            if " " in text or not text:
                text = '"' + text.replace('"', "'") + '"'
            parts.append(f"{key}={text}")
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", log_file: Path | None = None,
                  max_bytes: int = 2_000_000, backup_count: int = 3) -> None:
    """Configure the ``voterpulse`` logger tree once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("voterpulse")
    root.setLevel(level.upper())
    root.propagate = False
    formatter = KeyValueFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # read-only disk etc. — console logging still works
            root.warning("log_file_unavailable", extra={"fields": {"path": log_file, "error": exc}})
    _CONFIGURED = True


def setup_from_config() -> None:
    from voterpulse.config import get_config

    cfg = get_config()
    log_cfg = cfg.settings.logging
    setup_logging(
        log_cfg.level,
        cfg.resolve_path(log_cfg.file) if log_cfg.file else None,
        log_cfg.max_bytes,
        log_cfg.backup_count,
    )


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("voterpulse"):
        name = f"voterpulse.{name}"
    return logging.getLogger(name)


def fields(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """Shorthand: ``log.info("event", extra=fields(a=1))``."""
    return {"fields": kwargs}
