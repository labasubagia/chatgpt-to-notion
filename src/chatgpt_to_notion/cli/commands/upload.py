"""Upload CLI commands."""

from pathlib import Path
from typing import Any

import aiohttp
import typer

from ...adapters.chatgpt_api import (
    count_library_images,
    delete_library_images_by_query,
    get_headers,
)
from ...adapters.config_loader import (
    get_account_names,
    resolve_config,
    validate_runtime_config,
)
from ...domain.models import RuntimeOptions
from ...services import upload_sequential
from ...shared.console import (
    print_account_log_footer,
    print_account_log_header,
    safe_async_run,
)
from ...shared.verbosity import QUIET, VERBOSE, set_verbosity


def validate_db_id(db_id: str | None) -> str | None:
    if db_id is None:
        return None
    if not db_id or len(db_id) < 10:
        raise typer.BadParameter("Notion database ID must be a valid ID")
    return db_id


def _resolve_target_accounts(account: str | None, config: str | None) -> list[str]:
    if account:
        return [account]
    accounts = get_account_names(config)
    if not accounts:
        raise typer.BadParameter("No accounts found in config.")
    return accounts


def _build_runtime_options(
    config: str | None,
    account: str | None,
    history_folder: str | None,
    image_folder: str | None,
) -> list[tuple[str, RuntimeOptions]]:
    target_accounts = _resolve_target_accounts(account, config)
    return [
        (
            acc,
            RuntimeOptions(
                config_path=config,
                account=acc,
                history_folder=history_folder,
                image_folder=image_folder,
            ),
        )
        for acc in target_accounts
    ]


def _validate_and_resolve(
    options: RuntimeOptions, db_id: str | None, required_vars: list[str]
) -> Any:
    validate_runtime_config(required_vars, options=options)
    return resolve_config(options)


# Subcommand: upload history
def upload_history(
    history_folder: str | None = typer.Option(
        None, "--history-folder", help="Path to ChatGPT export folder"
    ),
    image_folder: str | None = typer.Option(
        None,
        "--image-folder",
        help=(
            "Path to folder containing images "
            "(default: output/images, supports absolute path)"
        ),
    ),
    db_id: str | None = typer.Option(
        None, help="Notion Database ID", callback=validate_db_id
    ),
    delete_conversation: bool = typer.Option(
        False, help="Delete conversation after upload"
    ),
    from_history: bool = typer.Option(
        True, help="Use history data as source (default: true for this subcommand)"
    ),
    all: bool = typer.Option(
        False, "--all", help="Load all data from history (not just today's)"
    ),
    no_cache: bool = typer.Option(
        False, help="Bypass SQLite cache and fetch fresh from API"
    ),
    limit: int = typer.Option(100, help="Limit number of image generations to process"),
    timezone: str | None = typer.Option(
        None, help="IANA timezone name, e.g. Asia/Jakarta"
    ),
    config: str | None = typer.Option(None, help="Path to TOML config file"),
    account: str | None = typer.Option(None, help="Named account from TOML config"),
    conversation_delay: float = typer.Option(
        2.0, "--conversation-delay", help="Delay between conversations (seconds)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-file output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Show only summary counts"),
    fail_log: Path | None = typer.Option(  # noqa: B008
        None, "--fail-log", help="Write failed items to a JSONL log file"
    ),
) -> None:
    """Upload image generations from ChatGPT history to Notion."""
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together")
    if verbose:
        set_verbosity(VERBOSE)
    elif quiet:
        set_verbosity(QUIET)

    required_vars = [
        "NOTION_API_KEY",
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
        "CHATGPT_COOKIE_STRING",
    ]
    if db_id is None:
        required_vars.append("NOTION_DATABASE_ID")

    account_options = _build_runtime_options(
        config, account, history_folder, image_folder
    )
    total_accounts = len(account_options)

    for index, (_target_account, options) in enumerate(account_options, start=1):
        resolved = _validate_and_resolve(options, db_id, required_vars)
        effective_db_id = db_id or resolved.notion.database_id
        if effective_db_id is None:
            raise typer.BadParameter("db_id must be provided")

        effective_keep_days = None if all else 3

        print_account_log_header(
            action="Upload History",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )

        kwargs: dict[str, Any] = {
            "image_folder": image_folder,
            "db_id": effective_db_id,
            "upload_to_notion": True,
            "delete_conversation": delete_conversation,
            "delete_library": False,
            "delete_library_all": False,
            "account": resolved.account_name,
            "check_notion_api": True,
            "from_history": True,
            "limit": limit,
            "keep_days": effective_keep_days,
            "timezone_name": timezone,
            "no_cache": no_cache,
            "options": options,
            "fail_log_path": fail_log,
            "conversation_delay": conversation_delay,
        }
        ok = safe_async_run(upload_sequential.upload_to_notion_sequential(**kwargs))
        if not ok:
            continue
        print_account_log_footer(
            action="Upload History",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


# Subcommand: upload verify
def upload_verify(
    history_folder: str | None = typer.Option(
        None, "--history-folder", help="Path to ChatGPT export folder"
    ),
    image_folder: str | None = typer.Option(
        None,
        "--image-folder",
        help=(
            "Path to folder containing images "
            "(default: output/images, supports absolute path)"
        ),
    ),
    db_id: str | None = typer.Option(
        None, help="Notion Database ID", callback=validate_db_id
    ),
    check_notion_api: bool = typer.Option(
        True, help="Check Notion API even when uploaded_at is already set"
    ),
    config: str | None = typer.Option(None, help="Path to TOML config file"),
    account: str | None = typer.Option(None, help="Named account from TOML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-file output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Show only summary counts"),
) -> None:
    """Verify upload status against Notion (no upload)."""
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together")
    if verbose:
        set_verbosity(VERBOSE)
    elif quiet:
        set_verbosity(QUIET)

    required_vars = [
        "NOTION_API_KEY",
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
        "CHATGPT_COOKIE_STRING",
    ]
    if db_id is None:
        required_vars.append("NOTION_DATABASE_ID")

    account_options = _build_runtime_options(
        config, account, history_folder, image_folder
    )
    total_accounts = len(account_options)

    for index, (_target_account, options) in enumerate(account_options, start=1):
        resolved = _validate_and_resolve(options, db_id, required_vars)
        effective_db_id = db_id or resolved.notion.database_id
        if effective_db_id is None:
            raise typer.BadParameter("db_id must be provided")

        print_account_log_header(
            action="Verify Upload",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )

        kwargs: dict[str, Any] = {
            "image_folder": image_folder,
            "db_id": effective_db_id,
            "upload_to_notion": False,
            "delete_conversation": False,
            "delete_library": False,
            "delete_library_all": False,
            "account": resolved.account_name,
            "check_notion_api": check_notion_api,
            "from_history": True,
            "limit": 100,
            "keep_days": 3,
            "timezone_name": None,
            "no_cache": False,
            "options": options,
            "fail_log_path": None,
            "conversation_delay": 0,
        }
        ok = safe_async_run(upload_sequential.upload_to_notion_sequential(**kwargs))
        if not ok:
            continue
        print_account_log_footer(
            action="Verify Upload",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


# Subcommand: upload library
def upload_library(
    delete: bool = typer.Option(
        False, "--delete", help="Delete matching images from library"
    ),
    delete_all: bool = typer.Option(
        False, "--delete-all", help="Delete ALL images from library"
    ),
    query: str | None = typer.Argument(
        None, help="Search query to filter library images"
    ),
    max_concurrent: int = typer.Option(
        10, "--max-concurrent", help="Max concurrent delete requests"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to TOML config file"
    ),
    account: str | None = typer.Option(
        None, "--account", help="Named account from TOML config"
    ),
) -> None:
    """Manage ChatGPT library images (list/delete)."""
    if not delete and not delete_all:
        raise typer.BadParameter("Use --delete or --delete-all")

    target_accounts = [account] if account else get_account_names(config)
    for target_account in target_accounts:
        options = RuntimeOptions(config_path=config, account=target_account)
        headers = get_headers(options)

        async def run(headers: dict[str, str], target_account: str) -> None:
            async with aiohttp.ClientSession() as session:
                if delete_all:
                    total = await delete_library_images_by_query(
                        session,
                        query=None,
                        headers=headers,
                        max_concurrent=max_concurrent,
                    )
                    print(f"[{target_account}] Deleted {total} images (all)")
                else:
                    total = await delete_library_images_by_query(
                        session,
                        query=query,
                        headers=headers,
                        max_concurrent=max_concurrent,
                    )
                    label = f"'{query}'" if query else "(no query)"
                    print(f"[{target_account}] Deleted {total} images matching {label}")
                remaining = await count_library_images(session, headers)
                print(
                    f"[{target_account}] Total images remaining in library: {remaining}"
                )

        safe_async_run(run(headers=headers, target_account=target_account))


# Deprecated alias: upload-to-notion
def upload_to_notion(
    history_folder: str | None = typer.Option(
        None, "--history-folder", help="Path to ChatGPT export folder"
    ),
    image_folder: str | None = typer.Option(
        None,
        "--image-folder",
        help=(
            "Path to folder containing images "
            "(default: output/images, supports absolute path)"
        ),
    ),
    db_id: str | None = typer.Option(
        None, help="Notion Database ID", callback=validate_db_id
    ),
    delete_conversation: bool = typer.Option(
        False, "--delete-conversation", help="Delete conversation after upload"
    ),
    from_history: bool = typer.Option(
        True, "--from-history", help="Use history data as source"
    ),
    all: bool = typer.Option(
        False, "--all", help="Load all data from history (not just today's)"
    ),
    no_cache: bool = typer.Option(
        False, help="Bypass SQLite cache and fetch fresh from API"
    ),
    limit: int = typer.Option(100, help="Limit number of image generations to process"),
    timezone: str | None = typer.Option(
        None, help="IANA timezone name, e.g. Asia/Jakarta"
    ),
    config: str | None = typer.Option(None, help="Path to TOML config file"),
    account: str | None = typer.Option(None, help="Named account from TOML config"),
    conversation_delay: float = typer.Option(
        2.0, "--conversation-delay", help="Delay between conversations (seconds)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-file output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Show only summary counts"),
    fail_log: Path | None = typer.Option(  # noqa: B008
        None, "--fail-log", help="Write failed items to a JSONL log file"
    ),
) -> None:
    """[DEPRECATED] Upload image generations to Notion. Use 'upload history' instead."""
    import warnings

    warnings.warn(
        "upload-to-notion is deprecated, use 'upload history' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together")
    if verbose:
        set_verbosity(VERBOSE)
    elif quiet:
        set_verbosity(QUIET)

    required_vars = [
        "NOTION_API_KEY",
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
        "CHATGPT_COOKIE_STRING",
    ]
    if db_id is None:
        required_vars.append("NOTION_DATABASE_ID")

    account_options = _build_runtime_options(
        config, account, history_folder, image_folder
    )
    total_accounts = len(account_options)

    for index, (_target_account, options) in enumerate(account_options, start=1):
        resolved = _validate_and_resolve(options, db_id, required_vars)
        effective_db_id = db_id or resolved.notion.database_id
        if effective_db_id is None:
            raise typer.BadParameter("db_id must be provided")

        effective_keep_days = None if all else 3

        print_account_log_header(
            action="Upload to Notion (deprecated)",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )

        kwargs: dict[str, Any] = {
            "image_folder": image_folder,
            "db_id": effective_db_id,
            "upload_to_notion": True,
            "delete_conversation": delete_conversation,
            "delete_library": False,
            "delete_library_all": False,
            "account": resolved.account_name,
            "check_notion_api": True,
            "from_history": from_history,
            "limit": limit,
            "keep_days": effective_keep_days,
            "timezone_name": timezone,
            "no_cache": no_cache,
            "options": options,
            "fail_log_path": fail_log,
            "conversation_delay": conversation_delay,
        }
        ok = safe_async_run(upload_sequential.upload_to_notion_sequential(**kwargs))
        if not ok:
            continue
        print_account_log_footer(
            action="Upload to Notion (deprecated)",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


def register(app: typer.Typer) -> None:
    # Subcommand group: upload
    upload_app = typer.Typer(name="upload", help="Upload and library management")
    upload_app.command("history")(upload_history)
    upload_app.command("verify")(upload_verify)
    upload_app.command("library")(upload_library)
    app.add_typer(upload_app)

    # Deprecated alias (hidden from main help)
    app.command("upload-to-notion", hidden=True)(upload_to_notion)
