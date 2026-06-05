"""Verbosity levels and structured output helpers."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERBOSE = 2
SIMPLE = 1
QUIET = 0

_verbosity: int = SIMPLE


def set_verbosity(level: int) -> None:
    global _verbosity
    _verbosity = level


def get_verbosity() -> int:
    return _verbosity


def is_verbose() -> bool:
    return _verbosity >= VERBOSE


def is_quiet() -> bool:
    return _verbosity <= QUIET


class StageCounter:
    """Accumulates per-stage counts, prints summary at end."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.counts: dict[str, int] = {}

    def add(self, category: str, count: int = 1) -> None:
        self.counts[category] = self.counts.get(category, 0) + count

    def summary_line(self) -> str:
        parts = [f"{v} {k}" for k, v in self.counts.items() if v > 0]
        if parts:
            return f"{self.label}: {', '.join(parts)}"
        return f"{self.label}: nothing to do"


def log_service_error(
    logger: logging.Logger,
    message: str,
    detail: str,
    exc: BaseException | None = None,
) -> None:
    if is_verbose():
        logger.exception("%s\n%s", message, detail)
    elif exc:
        logger.error("%s: %s", message, exc)
    else:
        logger.error("%s: %s", message, detail)


def write_fail_log(path: Path, entry: dict[str, Any]) -> None:
    """Append a JSONL entry to the fail log file."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
