"""Account-related CLI commands."""

from datetime import datetime

import typer
from rich.table import Table

from ...adapters import sqlite_store
from ...services.account_status_service import get_account_activity_statuses
from ...shared.console import console
from ...shared.time import resolve_timezone


def _print_activity_table(rows: list[dict[str, str]], title: str = "") -> None:
    if not rows:
        if title:
            console.print(f"[bold]{title}[/bold]")
            console.print("No accounts have data for this period.")
        else:
            console.print("No accounts found in config.")
        return

    columns = [
        "Account",
        "Next Wait",
        "Next Cooldown",
        "Fully Ready In",
        "Total Wait",
        "Ready Generate?",
    ]

    if title:
        console.print()
        console.print(f"[bold]{title}[/bold]")

    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(column, no_wrap=True)
    for row in rows:
        table.add_row(*(row[column] for column in columns))
    console.print(table)


def register(app: typer.Typer) -> None:
    @app.command()
    def account_status(
        config: str | None = typer.Option(None, help="Path to TOML config file"),
        timezone: str | None = typer.Option(
            None, help="IANA timezone name, e.g. Asia/Singapore"
        ),
    ) -> None:
        """Show which accounts are ready to generate new data."""
        sqlite_store.init_db()
        today_rows, yesterday_rows = get_account_activity_statuses(
            config_path=config,
            timezone_name=timezone,
        )
        user_tz = resolve_timezone(timezone_name=timezone)
        current_time = datetime.now().astimezone(user_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current Time: {current_time}")
        _print_activity_table(today_rows, title="Today")
        _print_activity_table(yesterday_rows, title="Yesterday")
