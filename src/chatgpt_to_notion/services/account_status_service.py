"""Account status service."""

from datetime import date, datetime, timedelta, timezone

from ..adapters import sqlite_store
from ..adapters.config_loader import get_account_names
from ..shared.time import format_duration, resolve_timezone


def _make_ready_row(account_name: str, service: str) -> dict[str, str]:
    return {
        "Account": account_name,
        "Service": service,
        "Next Wait": "Ready",
        "Next Cooldown": "0s",
        "Fully Ready In": "0s",
        "Total Wait": "0s",
        "Ready Generate?": "✅",
    }


def _db_has_valid_data(account_name: str) -> bool:
    return sqlite_store.has_generations(account_name)


def _get_activity_status_for_date(
    *,
    account_name: str,
    service: str,
    now: datetime,
    target_date: date,
) -> tuple[datetime, dict[str, str] | None]:
    tz = now.tzinfo or timezone.utc
    date_start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=tz)
    date_end = date_start + timedelta(days=1)
    cooldown_threshold = now - timedelta(days=1)

    stats = sqlite_store.get_activity_stats(
        account=account_name,
        date_start=date_start.astimezone(timezone.utc).isoformat(),
        date_end=date_end.astimezone(timezone.utc).isoformat(),
        cooldown_threshold=cooldown_threshold.astimezone(timezone.utc).isoformat(),
    )
    if stats is None:
        return now, None

    total_count = stats["total"]
    active_count = stats["active_count"]

    if active_count == 0:
        return now, _make_ready_row(account_name, service)

    first_active = datetime.fromisoformat(stats["first_active"])
    last_active = datetime.fromisoformat(stats["last_active"])
    if first_active.tzinfo is None:
        first_active = first_active.replace(tzinfo=timezone.utc)
    if last_active.tzinfo is None:
        last_active = last_active.replace(tzinfo=timezone.utc)

    first_active_local = first_active.astimezone(tz)
    last_active_local = last_active.astimezone(tz)

    next_wait = first_active_local + timedelta(days=1)
    status_msg = f"{active_count}/{total_count} to wait"
    ready_generate = (
        f"❌  ({status_msg})" if active_count == total_count else f"⚠️  ({status_msg})"
    )
    total_wait = (last_active_local - first_active_local).total_seconds()

    return next_wait, {
        "Account": account_name,
        "Service": service,
        "Next Wait": next_wait.strftime("%Y-%m-%d %H:%M:%S"),
        "Next Cooldown": format_duration((next_wait - now).total_seconds()),
        "Fully Ready In": format_duration(
            (last_active_local + timedelta(days=1) - now).total_seconds()
        ),
        "Total Wait": format_duration(total_wait),
        "Ready Generate?": ready_generate,
    }


def get_account_activity_statuses(
    *,
    config_path: str | None = None,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    account_names = get_account_names(config_path)
    if not account_names:
        return [], []

    tz = resolve_timezone(timezone_name)
    if now is None:
        now = datetime.now(tz)
    today_date = now.date()
    yesterday_date = (now - timedelta(days=1)).date()

    today_rows: list[dict[str, str]] = []
    yesterday_rows: list[dict[str, str]] = []
    today_sortable: list[tuple[datetime, dict[str, str]]] = []
    yesterday_sortable: list[tuple[datetime, dict[str, str]]] = []

    for account_name in account_names:
        has_data = _db_has_valid_data(account_name)

        if not has_data:
            ready_row = _make_ready_row(account_name, "chatgpt")
            today_sortable.append((now, ready_row))
            yesterday_sortable.append((now, ready_row))
            continue

        today_key, today_row = _get_activity_status_for_date(
            account_name=account_name,
            service="chatgpt",
            now=now,
            target_date=today_date,
        )
        yesterday_key, yesterday_row = _get_activity_status_for_date(
            account_name=account_name,
            service="chatgpt",
            now=now,
            target_date=yesterday_date,
        )
        if today_row is not None:
            today_sortable.append((today_key, today_row))
        if yesterday_row is not None:
            yesterday_sortable.append((yesterday_key, yesterday_row))
        else:
            ready_row = _make_ready_row(account_name, "chatgpt")
            yesterday_sortable.append((now, ready_row))

    for _, row in sorted(today_sortable, key=lambda item: item[0]):
        today_rows.append(row)
    for _, row in sorted(yesterday_sortable, key=lambda item: item[0]):
        yesterday_rows.append(row)
    return today_rows, yesterday_rows
