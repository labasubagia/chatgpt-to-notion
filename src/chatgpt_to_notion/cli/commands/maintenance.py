"""Maintenance CLI commands."""

import typer

from ...adapters.filesystem import clean_output_path


def register(app: typer.Typer) -> None:
    @app.command("clean-output-path")
    def clean_output_path_command() -> None:
        """Clean the output directory (except .gitkeep)."""
        print("Cleaning output path...")
        clean_output_path()
        print("Output path cleaned.")
