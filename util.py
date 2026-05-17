import asyncio
import logging
import os
import shutil
import tomllib
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from models import (
    AppConfig,
    ChatGPTImageGeneration,
    NotionContext,
    ProviderContext,
    ResolvedConfig,
    RuntimeOptions,
)

MAX_RETRIES = 5
MAX_CONCURRENT_DOWNLOADS = 10
MAX_CONCURRENT_REQUESTS = 10
HTTP_TIMEOUT_SECONDS = 30

OUTPUT_PATH = "./output"

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "config.toml"
console = Console(width=120)


def save_to_dataset(
    dataset: str,
    data: Sequence[dict] | Sequence[BaseModel],
    keep_days: int = 2,
    display_days: int = 1,
    options: RuntimeOptions | None = None,
) -> None:
    if dataset is None:
        return
    if len(data) == 0:
        print("No generations to save to dataset.")
        return

    account = _resolve_account_name(dataset, options)
    if account is None:
        return

    dict_data: list[dict]
    if data and isinstance(data[0], BaseModel):
        dict_data = [item.model_dump() for item in data]  # type: ignore[union-attr]
    else:
        dict_data = list(data)  # type: ignore[arg-type]

    generations = [ChatGPTImageGeneration(**d) for d in dict_data]
    import db

    db.upsert_generations(account, generations)

    today_count = len(
        [g for g in generations if _is_within_days(g.created_at, display_days)]
    )
    print(f"✅ Saved dataset for account '{account}' (Total Today: {today_count})\n")


def _is_within_days(created_at: str, days: int) -> bool:
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return False


def _resolve_account_name(
    dataset: str, options: RuntimeOptions | None = None
) -> str | None:
    if options and options.account:
        return options.account
    parts = dataset.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "history" and "_" in parts[-1]:
        filename = parts[-1]
        name_without_ext = filename.replace(".csv", "")
        if "_" in name_without_ext:
            account_name, _ = name_without_ext.rsplit("_", 1)
            return account_name
    return None


def get_uploaded_generation_ids(
    dataset: str | None, options: RuntimeOptions | None = None
) -> set[str]:
    if not dataset:
        return set()
    account = _resolve_account_name(dataset, options)
    if account is None:
        return set()
    import db

    return db.get_uploaded_ids(account)


def mark_generations_uploaded(
    dataset: str | None,
    generation_ids: set[str],
    options: RuntimeOptions | None = None,
) -> None:
    if not dataset or not generation_ids:
        return
    account = _resolve_account_name(dataset, options)
    if account is None:
        return
    import db

    db.mark_uploaded(account, generation_ids)


def get_output_path(input_path_str: str, is_dir=False) -> Path:
    input_path = Path(input_path_str)

    if input_path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {input_path_str}")

    base_dir: Path = Path(OUTPUT_PATH).resolve()
    final_path: Path = (base_dir / input_path).resolve()
    if base_dir not in final_path.parents:
        raise ValueError("Path attempts to escape the output directory!")

    if is_dir:
        final_path.mkdir(parents=True, exist_ok=True)
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return final_path


def _resolve_folder_path(folder_str: str | None, default_subpath: str) -> Path:
    if not folder_str:
        return Path(OUTPUT_PATH).resolve() / default_subpath
    path = Path(os.path.expanduser(folder_str))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_history_folder(options: RuntimeOptions | None = None) -> Path:
    options = options or RuntimeOptions()
    cli_folder = options.history_folder
    if cli_folder:
        return _resolve_folder_path(cli_folder, "")
    if options.config_path or options.account:
        try:
            resolved = resolve_config(options)
            config_folder = resolved.history_folder
            if config_folder:
                return _resolve_folder_path(config_folder, "")
        except ValueError:
            pass
    return Path(OUTPUT_PATH).resolve() / "history"


def get_image_folder(options: RuntimeOptions | None = None) -> Path:
    options = options or RuntimeOptions()
    cli_folder = options.image_folder
    if cli_folder:
        return _resolve_folder_path(cli_folder, "")
    if options.config_path or options.account:
        try:
            resolved = resolve_config(options)
            config_folder = resolved.image_folder
            if config_folder:
                return _resolve_folder_path(config_folder, "")
        except ValueError:
            pass
    return Path(OUTPUT_PATH).resolve() / "images"


def get_history_csv_path(
    account_name: str,
    service: str,
    options: RuntimeOptions | None = None,
) -> Path:
    folder = get_history_folder(options)
    return folder / f"{account_name}_{service}.csv"


def clean_output_path() -> None:
    base_dir: Path = Path(OUTPUT_PATH).resolve()
    if not base_dir.exists():
        return
    for item in base_dir.iterdir():
        if item.is_file() and item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def should_retry_http(exception: Exception) -> bool:
    if isinstance(exception, aiohttp.ClientResponseError):
        return http_retryable(exception.status)
    if isinstance(exception, aiohttp.ClientError):
        status = getattr(exception, "status", None)
        if status is not None:
            return http_retryable(status)
        return True
    if isinstance(
        exception,
        aiohttp.ClientConnectorError | aiohttp.ClientTimeout | asyncio.TimeoutError,
    ):
        return True
    return False


def retry_http():
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(should_retry_http),
        reraise=True,
    )


def http_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or status_code == 403 or status_code >= 500


def get_http_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)


def _load_toml_config(config_path: str | None = None) -> AppConfig | None:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return AppConfig.model_validate(tomllib.load(f))


def resolve_config(options: RuntimeOptions | None = None) -> ResolvedConfig:
    options = options or RuntimeOptions()
    app_config = _load_toml_config(options.config_path)

    if not app_config or not app_config.accounts:
        raise ValueError(
            "Missing configuration. Create config.toml with at least one account."
        )

    account_name = options.account
    if not account_name and len(app_config.accounts) == 1:
        account_name = next(iter(app_config.accounts))
    if not account_name:
        raise ValueError("No account selected. Pass --account.")
    account = app_config.accounts.get(account_name)
    if account is None:
        raise ValueError(f"Unknown account: {account_name}")
    notion = app_config.notion.model_copy(deep=True)
    if account.notion_database_id:
        notion.database_id = account.notion_database_id

    shared = app_config.shared
    history_folder = account.history_folder or shared.history_folder
    image_folder = account.image_folder or shared.image_folder

    return ResolvedConfig(
        account_name=account_name,
        account=account.model_copy(
            update={
                "user_agent": account.user_agent or shared.user_agent,
            }
        ),
        notion=notion,
        history_folder=history_folder,
        image_folder=image_folder,
    )


def get_provider_context(
    provider: Literal["chatgpt"],
    options: RuntimeOptions | None = None,
) -> ProviderContext:
    resolved = resolve_config(options)
    headers = {
        "Authorization": f"Bearer {resolved.account.authorization_token.strip()}",
        "User-Agent": (resolved.account.user_agent or "").strip(),
        "Content-Type": "application/json",
    }
    return ProviderContext(
        provider=provider,
        headers=headers,
        notion=resolved.notion,
        account_name=resolved.account_name,
    )


def get_notion_context(options: RuntimeOptions | None = None) -> NotionContext:
    resolved = resolve_config(options)
    notion = resolved.notion
    return NotionContext(
        headers={
            "Authorization": f"Bearer {(notion.api_key or '').strip()}",
            "Notion-Version": notion.version,
            "Content-Type": "application/json",
        },
        database_id=notion.database_id,
        account_name=resolved.account_name,
    )


def validate_runtime_config(
    required_vars: list[str], options: RuntimeOptions | None = None
) -> None:
    resolved = resolve_config(options)
    missing: list[str] = []
    for key in required_vars:
        match key:
            case "NOTION_API_KEY":
                if not (resolved.notion.api_key or "").strip():
                    missing.append(key)
            case "NOTION_DATABASE_ID":
                if not (resolved.notion.database_id or "").strip():
                    missing.append(key)
            case "CHATGPT_AUTHORIZATION_TOKEN":
                if not resolved.account.authorization_token.strip():
                    missing.append(key)
            case "CHATGPT_USER_AGENT":
                if not (resolved.account.user_agent or "").strip():
                    missing.append(key)
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")


def get_account_names(config_path: str | None = None) -> list[str]:
    app_config = _load_toml_config(config_path)
    if not app_config or not app_config.accounts:
        return []
    return list(app_config.accounts.keys())


def print_account_log_header(
    *,
    action: str,
    account_name: str,
    position: int,
    total: int,
) -> None:
    console.print()
    console.print(Rule(title=f"[{position}/{total}] {action}"))
    console.print(f"Account: {account_name}")
    console.print(Rule())


def print_account_log_footer(
    *,
    action: str,
    account_name: str,
    position: int,
    total: int,
) -> None:
    console.print(f"Finished [{position}/{total}] {action} for {account_name}")
    console.print(Rule())


def _format_duration(total_seconds: float) -> str:
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


def get_account_activity_statuses(
    *,
    config_path: str | None = None,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:

    account_names = get_account_names(config_path)
    if not account_names:
        return [], []

    tz = (
        ZoneInfo(timezone_name) if timezone_name else datetime.now().astimezone().tzinfo
    )
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


def _db_has_valid_data(account_name: str) -> bool:
    import db

    generations = db.get_generations(account_name)
    return len(generations) > 0


def _get_activity_status_for_date(
    *,
    account_name: str,
    service: str,
    now: datetime,
    target_date: date,
) -> tuple[datetime, dict[str, str] | None]:
    import db

    generations = db.get_generations(account_name, include_uploaded=True)
    if not generations:
        return now - timedelta(days=1), None

    created_at_list: list[datetime] = []
    for g in generations:
        try:
            dt = datetime.fromisoformat(g.created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_local = dt.astimezone(now.tzinfo)
            created_at_list.append(dt_local)
        except (ValueError, TypeError):
            continue

    if not created_at_list:
        return now - timedelta(days=1), None

    selected = [dt for dt in created_at_list if dt.date() == target_date]
    if not selected:
        return now, None

    total_count = len(selected)
    cooldown_threshold = now - timedelta(days=1)
    active_items = [dt for dt in selected if dt > cooldown_threshold]
    active_count = len(active_items)

    if active_count == 0:
        return now, _make_ready_row(account_name, service)

    first_active = min(active_items)
    last_active = max(active_items)
    next_wait = first_active + timedelta(days=1)
    status_msg = f"{active_count}/{total_count} to wait"
    ready_generate = (
        f"❌  ({status_msg})" if active_count == total_count else f"⚠️  ({status_msg})"
    )
    total_wait = (last_active - first_active).total_seconds()

    return next_wait, {
        "Account": account_name,
        "Service": service,
        "Next Wait": next_wait.strftime("%Y-%m-%d %H:%M:%S"),
        "Next Cooldown": _format_duration((next_wait - now).total_seconds()),
        "Fully Ready In": _format_duration(
            (last_active + timedelta(days=1) - now).total_seconds()
        ),
        "Total Wait": _format_duration(total_wait),
        "Ready Generate?": ready_generate,
    }


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    file_path: str,
    headers: dict[str, str] | None = None,
) -> None:
    async with session.get(url, headers=headers or {}) as response:
        response.raise_for_status()
        content = await response.read()
        Path(file_path).write_bytes(content)
