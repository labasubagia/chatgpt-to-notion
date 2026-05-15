import asyncio
from typing import Annotated

import typer
from rich.table import Table

import chatgpt
import util
from models import RuntimeOptions

app = typer.Typer()


def validate_db_id(db_id: str | None) -> str | None:
    """Validate Notion database ID"""
    if db_id is None:
        return None
    if not db_id or len(db_id) < 10:
        raise typer.BadParameter("Notion database ID must be a valid ID")
    return db_id


def _resolve_target_accounts(
    account: str | None,
    config: str | None,
) -> list[str]:
    if account:
        return [account]
    accounts = util.get_account_names(config)
    if not accounts:
        raise typer.BadParameter("No accounts found in config.")
    return accounts


def _account_dataset(account_name: str, service: str) -> str:
    return f"history/{account_name}_{service}.csv"


def _print_activity_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        util.console.print("No accounts found in config.")
        return

    columns = [
        "Account",
        "Next Wait",
        "Next Cooldown",
        "Fully Ready In",
        "Total Wait",
        "Ready Generate?",
    ]

    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(column, no_wrap=True)
    for row in rows:
        table.add_row(*(row[column] for column in columns))
    util.console.print(table)


@app.command()
def account_status(
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    timezone: Annotated[
        str | None,
        typer.Option(help="IANA timezone name, e.g. Asia/Singapore"),
    ] = None,
) -> None:
    """Show which accounts are ready to generate new data."""
    rows = util.get_account_activity_statuses(
        config_path=config,
        timezone_name=timezone,
    )
    _print_activity_table(rows)


@app.command("upload-to-notion")
def upload_to_notion(
    image_folder: Annotated[
        str, typer.Option(help="Path to the folder containing images")
    ] = "images",
    db_id: Annotated[
        str | None,
        typer.Option(help="Notion Database ID", callback=validate_db_id),
    ] = None,
    upload_to_notion: Annotated[
        bool, typer.Option(help="Whether to upload to Notion")
    ] = True,
    remove: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items after upload")
    ] = False,
    check_notion_api: Annotated[
        bool,
        typer.Option(
            help="Check Notion API even when uploaded_at is already set in CSV"
        ),
    ] = False,
    from_history: Annotated[
        bool,
        typer.Option(help="Use CSV as source (non-uploaded items only, today's data)"),
    ] = False,
    verify_history: Annotated[
        bool,
        typer.Option(
            help="Shortcut for --from-history --check-notion-api (includes uploaded)"
        ),
    ] = False,
    all: Annotated[
        bool,
        typer.Option(help="Load all data from CSV (not just today's)"),
    ] = False,
    limit: Annotated[
        int, typer.Option(help="Limit number of image generations to process")
    ] = 100,
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    account: Annotated[
        str | None, typer.Option(help="Named account from TOML config")
    ] = None,
):
    """Upload image generations to Notion"""
    target_accounts = _resolve_target_accounts(account, config)
    total_accounts = len(target_accounts)
    for index, target_account in enumerate(target_accounts, start=1):
        options = RuntimeOptions(config_path=config, account=target_account)
        required_vars = [
            "NOTION_API_KEY",
            "CHATGPT_AUTHORIZATION_TOKEN",
            "CHATGPT_USER_AGENT",
        ]
        if db_id is None:
            required_vars.append("NOTION_DATABASE_ID")
        util.validate_runtime_config(required_vars, options=options)
        resolved = util.resolve_config(options)
        util.print_account_log_header(
            action="Upload To Notion",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )
        effective_db_id = db_id or resolved.notion.database_id
        assert effective_db_id is not None, "db_id must be provided"
        account_dataset = _account_dataset(resolved.account_name, "chatgpt")
        effective_from_history = from_history or verify_history
        effective_check_notion_api = check_notion_api or verify_history
        if effective_from_history and not all:
            effective_keep_days = 1
        elif all:
            effective_keep_days = None
        else:
            effective_keep_days = None
        asyncio.run(
            chatgpt.upload_to_notion(
                image_folder=image_folder,
                db_id=effective_db_id,
                upload_to_notion=upload_to_notion,
                remove_in_chatgpt=remove,
                dataset=account_dataset,
                check_notion_api=effective_check_notion_api,
                from_history=effective_from_history,
                limit=limit,
                keep_days=effective_keep_days,
                options=options,
            )
        )
        util.print_account_log_footer(
            action="Upload To Notion",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


@app.command()
def clean_output_path():
    """Clean the output directory (except .gitkeep)"""
    print("Cleaning output path...")
    util.clean_output_path()
    print("Output path cleaned.")


if __name__ == "__main__":
    app()
