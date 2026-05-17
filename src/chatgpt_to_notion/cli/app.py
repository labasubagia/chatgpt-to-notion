"""Typer application entrypoint."""

import typer

from .commands import accounts, maintenance, upload

app = typer.Typer()
upload.register(app)
accounts.register(app)
maintenance.register(app)


if __name__ == "__main__":
    app()
