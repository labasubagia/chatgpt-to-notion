"""Configuration loading and resolution."""

import base64
import tomllib
from pathlib import Path

from ..domain.models import (
    AppConfig,
    NotionContext,
    ProviderContext,
    ResolvedConfig,
    RuntimeOptions,
)
from ..shared.constants import DEFAULT_CONFIG_PATH


def _load_config(config_path: str | None = None) -> AppConfig:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise ValueError(f"Missing configuration: {path} not found")
    with path.open("rb") as file_obj:
        return AppConfig.model_validate(tomllib.load(file_obj))


class Config:
    def __init__(self, config_path: str | None = None) -> None:
        self._app_config = _load_config(config_path)

    def resolve(self, options: RuntimeOptions | None = None) -> ResolvedConfig:
        runtime_options = options or RuntimeOptions()
        app_config = self._app_config

        if not app_config.accounts:
            raise ValueError(
                "Missing configuration. Create config.toml with at least one account."
            )

        account_name = runtime_options.account
        if not account_name and len(app_config.accounts) == 1:
            account_name = next(iter(app_config.accounts))
        if not account_name:
            raise ValueError("No account selected. Pass --account.")

        account = app_config.accounts.get(account_name)
        if account is None:
            raise ValueError(f"Unknown account: {account_name}")

        notion = app_config.notion.model_copy(deep=True)
        if account.notion_database_id:
            notion.database_id = account.notion_database_id

        shared = app_config.shared
        history_folder = account.history_folder or shared.history_folder
        image_folder = account.image_folder or shared.image_folder
        delete_library_queries = (
            account.delete_library_queries or shared.delete_library_queries
        )
        remove_all_library_images = (
            account.remove_all_library_images or shared.remove_all_library_images
        )
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
            history_folder=history_folder,
            image_folder=image_folder,
            delete_library_queries=delete_library_queries,
            remove_all_library_images=remove_all_library_images,
        )

    def provider_context(
        self,
        options: RuntimeOptions | None = None,
        *,
        resolved: ResolvedConfig | None = None,
    ) -> ProviderContext:
        if resolved is None:
            resolved = self.resolve(options)
        headers = {
            "Authorization": f"Bearer {resolved.account.authorization_token.strip()}",
            "User-Agent": (resolved.account.user_agent or "").strip(),
            "Content-Type": "application/json",
        }
        cookie_b64 = resolved.account.cookie_string_base64
        if cookie_b64:
            try:
                cookie_b64 = cookie_b64.strip()
                decoded_cookie = base64.b64decode(cookie_b64).decode("utf-8")
                headers["Cookie"] = decoded_cookie
            except Exception as e:
                raise ValueError(f"Invalid base64 in cookie_string_base64: {e}") from e
        return ProviderContext(
            provider="chatgpt",
            headers=headers,
            notion=resolved.notion,
            account_name=resolved.account_name,
        )

    def notion_context(
        self,
        options: RuntimeOptions | None = None,
        *,
        resolved: ResolvedConfig | None = None,
    ) -> NotionContext:
        if resolved is None:
            resolved = self.resolve(options)
        notion = resolved.notion
        return NotionContext(
            headers={
                "Authorization": f"Bearer {(notion.api_key or '').strip()}",
                "Notion-Version": notion.version,
                "Content-Type": "application/json",
            },
            database_id=notion.database_id,
            account_name=resolved.account_name,
        )

    def validate_runtime(
        self, required_vars: list[str], options: RuntimeOptions | None = None
    ) -> None:
        resolved = self.resolve(options)
        missing: list[str] = []
        for key in required_vars:
            match key:
                case "NOTION_API_KEY":
                    if not (resolved.notion.api_key or "").strip():
                        missing.append(key)
                case "NOTION_DATABASE_ID":
                    if not (resolved.notion.database_id or "").strip():
                        missing.append(key)
                case "CHATGPT_AUTHORIZATION_TOKEN":
                    if not resolved.account.authorization_token.strip():
                        missing.append(key)
                case "CHATGPT_USER_AGENT":
                    if not (resolved.account.user_agent or "").strip():
                        missing.append(key)
                case "CHATGPT_COOKIE_STRING":
                    if not (resolved.account.cookie_string_base64 or "").strip():
                        missing.append(key)
        if missing:
            raise ValueError(
                f"Missing required configuration values: {', '.join(missing)}"
            )

    def account_names(self) -> list[str]:
        return list(self._app_config.accounts.keys())


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
