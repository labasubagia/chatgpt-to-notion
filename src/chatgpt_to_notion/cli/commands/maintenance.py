"""Maintenance CLI commands."""

from pathlib import Path

import typer

from ...adapters.filesystem import clean_output_path
from ...adapters.sqlite_store import backup_db, restore_db


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
