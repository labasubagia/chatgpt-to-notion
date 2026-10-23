"""Upload CLI commands."""

from pathlib import Path
from typing import Any, Literal

import typer

from ...adapters.config_loader import (
    get_account_names,
    resolve_config,
    validate_runtime_config,
)
from ...domain.models import RuntimeOptions
from ...services import upload_batch, upload_sequential, upload_single
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


def register(app: typer.Typer) -> None:
    @app.command("upload-to-notion")
    def upload_to_notion_command(
        history_folder: str | None = typer.Option(
            None,
            help="Path to folder containing history data (default: output/history)",
        ),
        image_folder: str | None = typer.Option(
            None,
            help=(
                "Path to folder containing images "
                "(default: output/images, supports absolute path)"
            ),
        ),
        db_id: str | None = typer.Option(
            None, help="Notion Database ID", callback=validate_db_id
        ),
        upload_to_notion: bool = typer.Option(True, help="Whether to upload to Notion"),
        remove: bool = typer.Option(
            False, help="Whether to remove uploaded items after upload"
        ),
        remove_in_chatgpt_library: bool = typer.Option(
            False, help="Whether to remove images from ChatGPT library after upload"
        ),
        remove_all_in_chatgpt_library: bool = typer.Option(
            False,
            help="Remove ALL images from ChatGPT library after upload (ignores "
            "delete_library_queries)",
        ),
        check_notion_api: bool = typer.Option(
            True, help="Check Notion API even when uploaded_at is already set"
        ),
        from_history: bool = typer.Option(
            False, help="Use history data as source (non-uploaded items only)"
        ),
        verify_history: bool = typer.Option(
            False,
            help="Shortcut for --from-history --check-notion-api (includes uploaded)",
        ),
        all: bool = typer.Option(
            False, help="Load all data from history (not just today's)"
        ),
        no_cache: bool = typer.Option(
            False, help="Bypass SQLite cache and fetch fresh from API"
        ),
        limit: int = typer.Option(
            100, help="Limit number of image generations to process"
        ),
        timezone: str | None = typer.Option(
            None, help="IANA timezone name, e.g. Asia/Jakarta"
        ),
        config: str | None = typer.Option(None, help="Path to TOML config file"),
        account: str | None = typer.Option(None, help="Named account from TOML config"),
        max_concurrent: int = typer.Option(
            1, help="Max concurrent upload requests (1 = sequential)"
        ),
        fetch_conversation_delay: float | None = typer.Option(
            None,
            help="Delay in seconds between fetching conversation details "
            "(helps avoid 429 rate limits)",
        ),
        fetch_batch_size: int | None = typer.Option(
            None,
            help="Number of conversation detail requests per batch "
            "(0 = no batching, process all sequentially with delay)",
        ),
        mode: Literal["single", "batch", "sequential"] = typer.Option(
            "sequential",
            help=(
                "Processing mode: single (per-conversation, parallel), "
                "batch (full parallel), or sequential (one conversation at a time)"
            ),
        ),
        conversation_delay: float = typer.Option(
            2.0,
            help="Delay in seconds between conversations in sequential mode",
        ),
        verbose: bool = typer.Option(
            False, "--verbose", "-v", help="Show per-file output"
        ),
        quiet: bool = typer.Option(
            False, "--quiet", "-q", help="Show only summary counts"
        ),
        fail_log: Path | None = typer.Option(  # noqa: B008
            None, help="Write failed items to a JSONL log file"
        ),
    ) -> None:
        """Upload image generations to Notion."""
        if verbose and quiet:
            raise typer.BadParameter("Cannot use --verbose and --quiet together")
        if verbose:
            set_verbosity(VERBOSE)
        elif quiet:
            set_verbosity(QUIET)

        target_accounts = _resolve_target_accounts(account, config)
        total_accounts = len(target_accounts)
        for index, target_account in enumerate(target_accounts, start=1):
            options = RuntimeOptions(
                config_path=config,
                account=target_account,
                history_folder=history_folder,
                image_folder=image_folder,
                fetch_conversation_delay=fetch_conversation_delay,
                fetch_batch_size=fetch_batch_size,
            )
            required_vars = [
                "NOTION_API_KEY",
                "CHATGPT_AUTHORIZATION_TOKEN",
                "CHATGPT_USER_AGENT",
                "CHATGPT_COOKIE_STRING",
            ]
            if db_id is None:
                required_vars.append("NOTION_DATABASE_ID")
            validate_runtime_config(required_vars, options=options)
            resolved = resolve_config(options)
            print_account_log_header(
                action="Upload To Notion",
                account_name=resolved.account_name,
                position=index,
                total=total_accounts,
            )
            effective_db_id = db_id or resolved.notion.database_id
            if effective_db_id is None:
                raise typer.BadParameter("db_id must be provided")
            effective_from_history = from_history or verify_history
            effective_check_notion_api = check_notion_api or verify_history
            if effective_from_history and not all:
                effective_keep_days = 3
            elif all:
                effective_keep_days = None
            else:
                effective_keep_days = None

            if mode == "sequential":
                upload_fn = upload_sequential.upload_to_notion_sequential
            elif mode == "single":
                upload_fn = upload_single.upload_to_notion_single
            else:
                upload_fn = upload_batch.upload_to_notion
            kwargs: dict[str, Any] = {
                "image_folder": image_folder,
                "db_id": effective_db_id,
                "upload_to_notion": upload_to_notion,
                "remove_in_chatgpt": remove,
                "remove_in_chatgpt_library": remove_in_chatgpt_library,
                "remove_all_in_chatgpt_library": remove_all_in_chatgpt_library,
                "account": resolved.account_name,
                "check_notion_api": effective_check_notion_api,
                "from_history": effective_from_history,
                "limit": limit,
                "keep_days": effective_keep_days,
                "timezone_name": timezone,
                "no_cache": no_cache,
                "options": options,
                "fail_log_path": fail_log,
                "max_concurrent": max_concurrent,
            }
            if mode == "sequential":
                kwargs["conversation_delay"] = conversation_delay
            ok = safe_async_run(upload_fn(**kwargs))
            if not ok:
                continue
            print_account_log_footer(
                action="Upload To Notion",
                account_name=resolved.account_name,
                position=index,
                total=total_accounts,
            )
