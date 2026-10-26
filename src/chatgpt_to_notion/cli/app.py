"""Typer application entrypoint."""

import typer

from ..adapters.config import Config
from ..adapters.filesystem import Filesystem
from ..adapters.sqlite_store import SqliteStore
from ..domain.models import ChatGPTClient
from .commands import accounts, maintenance, upload


def create_app(
    config: Config | None = None,
    store: SqliteStore | None = None,
    filesystem: Filesystem | None = None,
    chatgpt_client: ChatGPTClient | None = None,
) -> typer.Typer:
    """Build the Typer app with dependencies injected from the root."""
    config = config or Config()
    store = store or SqliteStore()
    filesystem = filesystem or Filesystem(config)
    app = typer.Typer()
    upload.register(app, config, store, filesystem)
    accounts.register(app, config, store)
    maintenance.register(app, config, store, filesystem, chatgpt_client=chatgpt_client)
    return app


app = create_app()


if __name__ == "__main__":
    app()
