import asyncio
import logging
import shutil
import tomllib
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from models import (
    AppConfig,
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
) -> None:
    if dataset is None:
        return
    if len(data) == 0:
        print("No generations to save to dataset.")
        return

    dict_data: list[dict]
    if data and isinstance(data[0], BaseModel):
        dict_data = [item.model_dump() for item in data]  # type: ignore[union-attr]
    else:
        dict_data = list(data)  # type: ignore[arg-type]

    file_path = get_output_path(dataset)
    df_new = pd.DataFrame(dict_data)
    if "uploaded_at" not in df_new.columns:
        df_new["uploaded_at"] = ""
    df_final = _merge_recent_rows_by_id(
        target_path=file_path,
        df_new=df_new,
        keep_days=keep_days,
    )

    df_today = df_final.copy()
    if "created_at" in df_today.columns and display_days:
        df_today["created_at"] = pd.to_datetime(
            df_today["created_at"], utc=True, format="ISO8601", errors="coerce"
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=display_days)
        df_today = df_today.dropna(subset=["created_at"])
        df_today = df_today[df_today["created_at"] >= cutoff]

    df_final.to_csv(file_path, index=False)
    print(f"✅ Saved dataset to {file_path} (Total Today: {len(df_today)})\n")


def _merge_recent_rows_by_id(
    *,
    target_path: Path,
    df_new: pd.DataFrame,
    keep_days: int,
) -> pd.DataFrame:
    uploaded_at_by_id: dict[str, str] = {}
    if target_path.exists():
        df_existing = pd.read_csv(target_path)
        if "id" in df_existing.columns and "uploaded_at" in df_existing.columns:
            existing_uploaded = df_existing.dropna(subset=["id", "uploaded_at"])
            existing_uploaded = existing_uploaded[
                existing_uploaded["uploaded_at"].astype(str).str.strip() != ""
            ]
            uploaded_at_by_id = dict(
                zip(
                    existing_uploaded["id"].astype(str),
                    existing_uploaded["uploaded_at"].astype(str),
                    strict=False,
                )
            )
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    if "uploaded_at" not in df_combined.columns:
        df_combined["uploaded_at"] = ""

    if "id" in df_combined.columns:
        df_combined = df_combined.dropna(subset=["id"]).copy()
        df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
        if uploaded_at_by_id:
            existing_uploaded_at = df_combined["id"].astype(str).map(uploaded_at_by_id)
            current_uploaded_at = df_combined["uploaded_at"].fillna("").astype(str)
            df_combined["uploaded_at"] = current_uploaded_at.mask(
                current_uploaded_at.str.strip() == "",
                existing_uploaded_at,
            ).fillna("")

    if "created_at" in df_combined.columns:
        df_combined["created_at"] = pd.to_datetime(
            df_combined["created_at"], utc=True, format="ISO8601", errors="coerce"
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        df_combined = df_combined.dropna(subset=["created_at"]).copy()
        df_combined = df_combined[df_combined["created_at"] >= cutoff]
        df_combined = df_combined.sort_values(by="created_at", ascending=False)

    return df_combined


def get_uploaded_generation_ids(dataset: str | None) -> set[str]:
    if not dataset:
        return set()
    file_path = get_output_path(dataset)
    if not file_path.exists():
        return set()

    df = pd.read_csv(file_path, usecols=lambda column: column in {"id", "uploaded_at"})
    if df.empty or "id" not in df.columns or "uploaded_at" not in df.columns:
        return set()

    uploaded = df.dropna(subset=["id", "uploaded_at"]).copy()
    uploaded = uploaded[uploaded["uploaded_at"].astype(str).str.strip() != ""]
    return set(uploaded["id"].astype(str))


def mark_generations_uploaded(dataset: str | None, generation_ids: set[str]) -> None:
    if not dataset or not generation_ids:
        return
    file_path = get_output_path(dataset)
    if not file_path.exists():
        return

    df = pd.read_csv(file_path)
    if df.empty or "id" not in df.columns:
        return
    if "uploaded_at" not in df.columns:
        df["uploaded_at"] = ""
    df["uploaded_at"] = df["uploaded_at"].fillna("").astype(str)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    mask = df["id"].astype(str).isin(generation_ids)
    df.loc[mask, "uploaded_at"] = uploaded_at
    df.to_csv(file_path, index=False)
    print(f"✅ Marked {int(mask.sum())} rows uploaded in {file_path}")


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


def clean_output_path() -> None:
    # Except .gitkeep, force remove all files and folders in OUTPUT_PATH
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
    """Determine if an HTTP exception should be retried"""
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
    """Reusable retry decorator for async HTTP requests"""
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
    """Get default HTTP timeout configuration"""
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
    return ResolvedConfig(
        account_name=account_name,
        account=account.model_copy(
            update={
                "user_agent": account.user_agent or app_config.shared.user_agent,
            }
        ),
        notion=notion,
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


def _activity_csv_path(
    *,
    account_name: str,
    service: str,
) -> Path:
    return Path(OUTPUT_PATH).resolve() / "history" / f"{account_name}_{service}.csv"


def get_account_activity_statuses(
    *,
    config_path: str | None = None,
    timezone_name: str | None = None,
) -> list[dict[str, str]]:
    account_names = get_account_names(config_path)
    if not account_names:
        return []

    tz = (
        ZoneInfo(timezone_name) if timezone_name else datetime.now().astimezone().tzinfo
    )
    now = datetime.now(tz)
    rows: list[dict[str, str]] = []
    sortable_rows: list[tuple[datetime, dict[str, str]]] = []

    for account_name in account_names:
        csv_path = _activity_csv_path(
            account_name=account_name,
            service="chatgpt",
        )
        sort_key, row = _get_activity_status_for_csv(
            account_name=account_name,
            service="chatgpt",
            csv_path=csv_path,
            now=now,
        )
        sortable_rows.append((sort_key, row))

    for _, row in sorted(sortable_rows, key=lambda item: item[0]):
        rows.append(row)
    return rows


def _get_activity_status_for_csv(
    *,
    account_name: str,
    service: str,
    csv_path: Path,
    now: datetime,
) -> tuple[datetime, dict[str, str]]:
    ready_row = {
        "Account": account_name,
        "Next Wait": "Ready",
        "Next Cooldown": "0s",
        "Fully Ready In": "0s",
        "Total Wait": "0s",
        "Ready Generate?": "✅",
    }
    if not csv_path.exists():
        return now - timedelta(days=1), ready_row

    try:
        df = pd.read_csv(csv_path, usecols=["created_at"])
    except Exception:
        return now - timedelta(days=1), ready_row

    if df.empty:
        return now - timedelta(days=1), ready_row

    created_at = pd.to_datetime(
        df["created_at"], utc=True, format="ISO8601", errors="coerce"
    ).dropna()
    if created_at.empty:
        return now - timedelta(days=1), ready_row

    created_at = created_at.dt.tz_convert(now.tzinfo)
    cooldown_threshold = now - timedelta(days=1)
    active_items = created_at[created_at > cooldown_threshold]
    total_count = len(created_at)

    if active_items.empty:
        return now, ready_row

    first_active = active_items.min()
    last_active = active_items.max()
    next_wait = first_active + timedelta(days=1)
    count_waiting = len(active_items)
    status_msg = f"{count_waiting}/{total_count} to wait"
    ready_generate = (
        f"❌  ({status_msg})" if count_waiting >= total_count else f"⚠️  ({status_msg})"
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
    """Download an image from URL to file path"""
    async with session.get(url, headers=headers or {}) as response:
        response.raise_for_status()
        content = await response.read()
        Path(file_path).write_bytes(content)
