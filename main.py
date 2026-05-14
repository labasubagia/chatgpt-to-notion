import asyncio
from typing import Annotated

import typer

import chatgpt
import sora
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
        typer.echo("No accounts found in config.")
        return

    columns = [
        "Account",
        "Service",
        "Next Wait",
        "Next Cooldown",
        "Fully Ready In",
        "Total Wait",
        "Ready Generate?",
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    typer.echo(header)
    typer.echo(divider)
    for row in rows:
        typer.echo("  ".join(row[column].ljust(widths[column]) for column in columns))


@app.command()
def account_status(
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    service: Annotated[
        str,
        typer.Option(help="Service to inspect: chatgpt, sora, or all"),
    ] = "chatgpt",
    timezone: Annotated[
        str | None,
        typer.Option(help="IANA timezone name, e.g. Asia/Singapore"),
    ] = None,
) -> None:
    """Show which accounts are ready to generate new data."""
    if service not in {"chatgpt", "sora", "all"}:
        raise typer.BadParameter("Service must be chatgpt, sora, or all.")
    rows = util.get_account_activity_statuses(
        config_path=config,
        service=service,
        timezone_name=timezone,
    )
    _print_activity_table(rows)


@app.command()
def sora_upload_to_notion(
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
    trash_in_sora: Annotated[
        bool, typer.Option(help="Whether to trash uploaded items in Sora")
    ] = False,
    remove_in_sora: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items in Sora")
    ] = False,
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    account: Annotated[
        str | None, typer.Option(help="Named account from TOML config")
    ] = None,
):
    """Upload Sora generations to Notion"""
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
            action="Sora Upload To Notion",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )
        effective_db_id = db_id or resolved.notion.database_id
        assert effective_db_id is not None, "db_id must be provided"
        account_dataset = _account_dataset(resolved.account_name, "sora")
        asyncio.run(
            sora.upload_to_notion(
                image_folder,
                effective_db_id,
                upload_to_notion=upload_to_notion,
                trash_in_sora=trash_in_sora,
                remove_in_sora=remove_in_sora,
                dataset=account_dataset,
                options=options,
            )
        )
        util.print_account_log_footer(
            action="Sora Upload To Notion",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


@app.command()
def sora_cleanup_trash(
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    account: Annotated[
        str | None, typer.Option(help="Named account from TOML config")
    ] = None,
):
    """Clean up trashed Sora generations"""
    target_accounts = _resolve_target_accounts(account, config)
    total_accounts = len(target_accounts)
    for index, target_account in enumerate(target_accounts, start=1):
        options = RuntimeOptions(config_path=config, account=target_account)
        util.validate_runtime_config(
            [
                "CHATGPT_AUTHORIZATION_TOKEN",
                "CHATGPT_USER_AGENT",
            ],
            options=options,
        )
        resolved = util.resolve_config(options)
        account_dataset = _account_dataset(resolved.account_name, "sora_trash")
        util.print_account_log_header(
            action="Sora Cleanup Trash",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )
        asyncio.run(sora.cleanup_trash(dataset=account_dataset, options=options))
        util.print_account_log_footer(
            action="Sora Cleanup Trash",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


@app.command()
def sora_cleanup_tasks(
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    account: Annotated[
        str | None, typer.Option(help="Named account from TOML config")
    ] = None,
):
    """Clean up empty Sora tasks"""
    target_accounts = _resolve_target_accounts(account, config)
    total_accounts = len(target_accounts)
    for index, target_account in enumerate(target_accounts, start=1):
        options = RuntimeOptions(config_path=config, account=target_account)
        util.validate_runtime_config(
            [
                "CHATGPT_AUTHORIZATION_TOKEN",
                "CHATGPT_USER_AGENT",
            ],
            options=options,
        )
        resolved = util.resolve_config(options)
        util.print_account_log_header(
            action="Sora Cleanup Tasks",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )
        asyncio.run(sora.cleanup_tasks(options=options))
        util.print_account_log_footer(
            action="Sora Cleanup Tasks",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )


@app.command()
def chatgpt_upload_to_notion(
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
    remove_in_chatgpt: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items in ChatGPT")
    ] = False,
    limit: Annotated[
        int, typer.Option(help="Limit number of image generations to process")
    ] = 100,
    config: Annotated[str | None, typer.Option(help="Path to TOML config file")] = None,
    account: Annotated[
        str | None, typer.Option(help="Named account from TOML config")
    ] = None,
):
    """Upload ChatGPT image generations to Notion"""
    target_accounts = _resolve_target_accounts(account, config)
    total_accounts = len(target_accounts)
    for index, target_account in enumerate(target_accounts, start=1):
        options = RuntimeOptions(config_path=config, account=target_account)
        required_vars = [
            "NOTION_API_KEY",
            "CHATGPT_AUTHORIZATION_TOKEN",
            "CHATGPT_USER_AGENT",
            "CHATGPT_COOKIE_STRING_BASE64",
        ]
        if db_id is None:
            required_vars.append("NOTION_DATABASE_ID")
        util.validate_runtime_config(required_vars, options=options)
        resolved = util.resolve_config(options)
        util.print_account_log_header(
            action="ChatGPT Upload To Notion",
            account_name=resolved.account_name,
            position=index,
            total=total_accounts,
        )
        effective_db_id = db_id or resolved.notion.database_id
        assert effective_db_id is not None, "db_id must be provided"
        account_dataset = _account_dataset(resolved.account_name, "chatgpt")
        asyncio.run(
            chatgpt.upload_to_notion(
                image_folder=image_folder,
                db_id=effective_db_id,
                upload_to_notion=upload_to_notion,
                remove_in_chatgpt=remove_in_chatgpt,
                dataset=account_dataset,
                limit=limit,
                options=options,
            )
        )
        util.print_account_log_footer(
            action="ChatGPT Upload To Notion",
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
