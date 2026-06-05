"""Console utilities."""

import asyncio
import logging

from rich.console import Console
from rich.rule import Rule

from .http import DetailedHTTPError

logger = logging.getLogger(__name__)
console = Console()


def safe_async_run(coro) -> bool:
    try:
        asyncio.run(coro)
        return True
    except DetailedHTTPError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return False


def print_account_log_header(
    *,
    action: str,
    account_name: str,
    position: int,
    total: int,
) -> None:
    console.print()
    console.print(Rule(title=f"[{position}/{total}] {action}"))
    console.print(f"Account: {account_name}")
    console.print(Rule())


def print_account_log_footer(
    *,
    action: str,
    account_name: str,
    position: int,
    total: int,
) -> None:
    console.print(f"Finished [{position}/{total}] {action} for {account_name}")
    console.print(Rule())
