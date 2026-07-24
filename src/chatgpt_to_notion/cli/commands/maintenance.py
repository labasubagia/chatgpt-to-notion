"""Maintenance CLI commands."""

from pathlib import Path

import aiohttp
import typer

from ...adapters.chatgpt_api import get_headers, remove_library_images_by_query
from ...adapters.filesystem import clean_output_path
from ...adapters.sqlite_store import backup_db, restore_db
from ...domain.models import RuntimeOptions
from ...shared.console import safe_async_run


def register(app: typer.Typer) -> None:
    @app.command("clean-output-path")
    def clean_output_path_command() -> None:
        """Clean the output directory (except .gitkeep)."""
        print("Cleaning output path...")
        clean_output_path()
        print("Output path cleaned.")

    @app.command("backup-db")
    def backup_db_command(
        target: Path = typer.Argument(  # noqa: B008
            ..., help="Destination path for the backup file."
        ),
    ) -> None:
        """Copy the local SQLite database to TARGET."""
        print(f"Backing up database to {target}...")
        result = backup_db(target)
        print(f"Backup saved to {result}")

    @app.command("restore-db")
    def restore_db_command(
        source: Path = typer.Argument(  # noqa: B008
            ..., help="Path to the backup file to restore from."
        ),
    ) -> None:
        """Overwrite the local SQLite database with SOURCE."""
        print(f"Restoring database from {source}...")
        result = restore_db(source)
        print(f"Database restored from {result}")

    @app.command("remove-images")
    def remove_images_command(
        query: str | None = typer.Argument(
            None, help="Search query to filter library images (omit for all)"
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
        """Remove images from ChatGPT library, optionally filtered by query."""

        async def run() -> None:
            options = RuntimeOptions(config_path=config, account=account)
            headers = get_headers(options)
            async with aiohttp.ClientSession() as session:
                total = await remove_library_images_by_query(
                    session, query=query, headers=headers, max_concurrent=max_concurrent
                )
            label = f"'{query}'" if query else "(all images)"
            print(f"Removed {total} images matching {label}")

        safe_async_run(run())
