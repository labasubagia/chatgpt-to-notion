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


def _load_toml_config(config_path: str | None = None) -> AppConfig | None:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return None
    with path.open("rb") as file_obj:
        return AppConfig.model_validate(tomllib.load(file_obj))


def resolve_config(options: RuntimeOptions | None = None) -> ResolvedConfig:
    if options is None:
        runtime_options = RuntimeOptions()
    elif isinstance(options, RuntimeOptions):
        runtime_options = options
    else:
        runtime_options = RuntimeOptions(
            config_path=getattr(options, "config_path", None),
            account=getattr(options, "account", None),
            history_folder=getattr(options, "history_folder", None),
            image_folder=getattr(options, "image_folder", None),
        )
    app_config = _load_toml_config(runtime_options.config_path)

    if not app_config or not app_config.accounts:
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
    )


def get_provider_context(
    provider: str = "chatgpt",
    options: RuntimeOptions | None = None,
) -> ProviderContext:
    resolved = resolve_config(options)
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


def get_notion_context(options: RuntimeOptions | None = None) -> NotionContext:
    resolved = resolve_config(options)
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


def validate_runtime_config(
    required_vars: list[str], options: RuntimeOptions | None = None
) -> None:
    resolved = resolve_config(options)
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
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")


def get_account_names(config_path: str | None = None) -> list[str]:
    app_config = _load_toml_config(config_path)
    if not app_config or not app_config.accounts:
        return []
    return list(app_config.accounts.keys())
