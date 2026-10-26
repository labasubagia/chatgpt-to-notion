"""Configuration loading and resolution."""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from ..domain.models import (
    AppConfig,
    ResolvedConfig,
)
from ..shared.constants import DEFAULT_CONFIG_PATH


def _load_config(config_path: str | None = None) -> AppConfig:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise ValueError(f"Missing configuration: {path} not found")
    with path.open("rb") as file_obj:
        return AppConfig.model_validate(tomllib.load(file_obj))


def _resolve_account(app: AppConfig, account_name: str) -> ResolvedConfig:
    account = app.accounts.get(account_name)
    if account is None:
        raise ValueError(f"Unknown account: {account_name}")

    notion = app.notion.model_copy(deep=True)
    if account.notion_database_id:
        notion.database_id = account.notion_database_id

    shared = app.shared
    return ResolvedConfig(
        account_name=account_name,
        account=account.model_copy(
            update={
                "user_agent": account.user_agent or shared.user_agent,
                "cookie_string_base64": (
                    account.cookie_string_base64 or shared.cookie_string_base64
                ),
            }
        ),
        notion=notion,
        history_folder=account.history_folder or shared.history_folder,
        image_folder=account.image_folder or shared.image_folder,
        delete_library_queries=(
            account.delete_library_queries or shared.delete_library_queries
        ),
        remove_all_library_images=(
            account.remove_all_library_images or shared.remove_all_library_images
        ),
    )


class Config:
    """Immutable per-account snapshot of the resolved TOML configuration."""

    def __init__(self, config_path: str | None = None) -> None:
        self._app_config = _load_config(config_path)
        if not self._app_config.accounts:
            raise ValueError(
                "Missing configuration. Create config.toml with at least one account."
            )
        accounts = {
            name: _resolve_account(self._app_config, name)
            for name in self._app_config.accounts
        }
        self._accounts: Mapping[str, ResolvedConfig] = MappingProxyType(accounts)

    @property
    def accounts(self) -> Mapping[str, ResolvedConfig]:
        return self._accounts

    def account_names(self) -> list[str]:
        return list(self._accounts)
