"""Upload CLI commands."""

import asyncio
from typing import Literal

import typer

from ...adapters.config_loader import (
    get_account_names,
    resolve_config,
    validate_runtime_config,
)
from ...domain.models import RuntimeOptions
from ...services import upload_pipeline
from ...shared.console import print_account_log_footer, print_account_log_header


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
        mode: Literal["single", "batch"] = typer.Option(
            "single",
            help="Processing mode: single (per-file, resilient) or batch (parallel)",
        ),
    ) -> None:
        """Upload image generations to Notion."""
        target_accounts = _resolve_target_accounts(account, config)
        total_accounts = len(target_accounts)
        for index, target_account in enumerate(target_accounts, start=1):
            options = RuntimeOptions(
                config_path=config,
                account=target_account,
                history_folder=history_folder,
                image_folder=image_folder,
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
            assert effective_db_id is not None, "db_id must be provided"
            effective_from_history = from_history or verify_history
            effective_check_notion_api = check_notion_api or verify_history
            if effective_from_history and not all:
                effective_keep_days = 1
            elif all:
                effective_keep_days = None
            else:
                effective_keep_days = None

            upload_fn = (
                upload_pipeline.upload_to_notion_single
                if mode == "single"
                else upload_pipeline.upload_to_notion
            )
            asyncio.run(
                upload_fn(
                    image_folder=image_folder,
                    db_id=effective_db_id,
                    upload_to_notion=upload_to_notion,
                    remove_in_chatgpt=remove,
                    account=resolved.account_name,
                    check_notion_api=effective_check_notion_api,
                    from_history=effective_from_history,
                    limit=limit,
                    keep_days=effective_keep_days,
                    timezone_name=timezone,
                    no_cache=no_cache,
                    options=options,
                )
            )
            print_account_log_footer(
                action="Upload To Notion",
                account_name=resolved.account_name,
                position=index,
                total=total_accounts,
            )
