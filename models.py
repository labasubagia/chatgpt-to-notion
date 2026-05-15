"""Type definitions for image generation data structures and app config."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict


class ImageGeneration(Protocol):
    """Protocol for image generation data with common fields."""

    id: str
    prompt: str


class ChatGPTImageGeneration(BaseModel):
    """Image generation from ChatGPT DALL-E."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    id: str
    conversation_id: str
    message_id: str
    asset_pointer: str
    url: str
    prompt: str = ""


class NotionConfig(BaseModel):
    """Notion credentials and defaults."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    database_id: str | None = None
    version: str = "2025-09-03"


class AccountConfig(BaseModel):
    """Per-account ChatGPT credentials."""

    model_config = ConfigDict(extra="forbid")

    authorization_token: str
    user_agent: str | None = None
    notion_database_id: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None


class SharedAccountConfig(BaseModel):
    """Shared browser/session defaults across accounts."""

    model_config = ConfigDict(extra="forbid")

    user_agent: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None


class AppConfig(BaseModel):
    """Application config with named accounts."""

    model_config = ConfigDict(extra="forbid")

    notion: NotionConfig = NotionConfig()
    shared: SharedAccountConfig = SharedAccountConfig()
    accounts: dict[str, AccountConfig] = {}


class ResolvedConfig(BaseModel):
    """Fully resolved config for one run."""

    model_config = ConfigDict(extra="forbid")

    account_name: str
    account: AccountConfig
    notion: NotionConfig
    history_folder: str | None = None
    image_folder: str | None = None


class RuntimeOptions(BaseModel):
    """CLI-provided config selectors."""

    model_config = ConfigDict(extra="forbid")

    config_path: str | None = None
    account: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None


class ProviderContext(BaseModel):
    """Resolved headers and defaults for API calls."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["chatgpt"]
    headers: dict[str, str]
    notion: NotionConfig
    account_name: str


class NotionContext(BaseModel):
    """Resolved Notion headers and database selection."""

    model_config = ConfigDict(extra="forbid")

    headers: dict[str, str]
    database_id: str | None = None
    account_name: str = ""


ConfigDictType = dict[str, Any]
