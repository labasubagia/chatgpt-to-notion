"""Time helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo


def resolve_timezone(timezone_name: str | None = None):
    """Resolve timezone: provided IANA name, or system default."""
    if timezone_name:
        return ZoneInfo(timezone_name)
    return datetime.now().astimezone().tzinfo


def format_duration(total_seconds: float) -> str:
    total_seconds_int = max(0, int(total_seconds))
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    seconds = total_seconds_int % 60

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
    else:
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
    return " ".join(parts)
