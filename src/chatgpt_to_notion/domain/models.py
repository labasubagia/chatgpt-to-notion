"""Type definitions for image generation data structures and app config."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import aiohttp
from email_validator import EmailNotValidError, validate_email
from pydantic import BaseModel, ConfigDict, model_validator


class ImageGeneration(Protocol):
    """Protocol for image generation data with common fields."""

    id: str
    conversation_id: str
    message_id: str
    asset_pointer: str
    url: str
    prompt: str


class ChatGPTImageGeneration(BaseModel):
    """Image generation from ChatGPT."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    id: str
    conversation_id: str
    message_id: str
    asset_pointer: str
    url: str
    prompt: str = ""


class ChatGPTClient(Protocol):
    """Protocol for ChatGPT API operations."""

    async def get_conversation_details(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
    ) -> dict[str, Any]: ...

    async def delete_conversation(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
    ) -> dict[str, Any]: ...

    async def get_image_generations(
        self,
        session: aiohttp.ClientSession,
        limit: int = 100,
    ) -> dict[str, Any]: ...

    async def get_library_images(
        self,
        session: aiohttp.ClientSession,
        query: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]: ...

    async def get_prompt_from_image_node_in_conversation(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
        message_id: str,
        asset_pointer: str,
    ) -> str: ...

    async def delete_library_file(
        self,
        session: aiohttp.ClientSession,
        item: dict[str, Any],
    ) -> dict[str, Any]: ...

    def get_headers(self) -> dict[str, str]: ...

    async def count_library_images(
        self,
        session: aiohttp.ClientSession,
    ) -> int: ...

    async def delete_library_images_by_query(
        self,
        session: aiohttp.ClientSession,
        query: str | None = None,
        max_concurrent: int = 10,
    ) -> int: ...
    async def remove_library_images(
        self,
        resolved: ResolvedConfig,
        remove_all: bool = False,
    ) -> None: ...


class NotionClient(Protocol):
    """Protocol for Notion API operations."""

    async def is_page_exists_in_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        query: str,
    ) -> bool: ...

    async def add_page_to_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        file_path: str,
        prompt: str | None = None,
    ) -> dict[str, Any]: ...

    def get_headers(self) -> dict[str, str]: ...


class HistoryServiceProtocol(Protocol):
    """Protocol for history retrieval operations."""

    async def load_image_generations(
        self,
        account: str,
        include_uploaded: bool = False,
        keep_days: int | None = None,
        timezone_name: str | None = None,
    ) -> list[ChatGPTImageGeneration]: ...

    async def fetch_generations_for_conversation(
        self,
        client: ChatGPTClient,
        session: aiohttp.ClientSession,
        conv_id: str,
        image_items: Sequence[dict[str, Any]],
        existing_ids: set[str],
    ) -> list[ChatGPTImageGeneration]: ...

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        file_path: str,
        headers: dict[str, str] | None = None,
    ) -> None: ...

    async def get_uploaded_generation_ids(self, account: str | None) -> set[str]: ...

    async def mark_generations_uploaded(
        self, account: str | None, generation_ids: set[str]
    ) -> None: ...


class ImageServiceProtocol(Protocol):
    """Protocol for image metadata operations."""

    def add_prompt_to_image_single(
        self,
        generation: ImageGeneration,
        folder: str,
    ) -> None: ...


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
    cookie_string_base64: str | None = None
    notion_database_id: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None
    delete_library_queries: list[str] = []
    remove_all_library_images: bool = False


class SharedAccountConfig(BaseModel):
    """Shared browser/session defaults across accounts."""

    model_config = ConfigDict(extra="forbid")

    user_agent: str | None = None
    cookie_string_base64: str | None = None
    notion_database_id: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None
    delete_library_queries: list[str] = []
    remove_all_library_images: bool = False


class AppConfig(BaseModel):
    """Application config with named accounts."""

    model_config = ConfigDict(extra="forbid")

    notion: NotionConfig = NotionConfig()
    shared: SharedAccountConfig = SharedAccountConfig()
    accounts: dict[str, AccountConfig] = {}

    @model_validator(mode="after")
    def _validate_account_keys_are_emails(self) -> AppConfig:
        invalid: list[str] = []
        for name in self.accounts:
            try:
                validate_email(name, check_deliverability=False)
            except EmailNotValidError:
                invalid.append(name)
        if invalid:
            joined = ", ".join(repr(name) for name in invalid)
            raise ValueError(
                f"Account keys must be valid email addresses, got: {joined}"
            )
        return self


class ResolvedConfig(BaseModel):
    """Fully resolved config for one run."""

    model_config = ConfigDict(extra="forbid")

    account_name: str
    account: AccountConfig
    notion: NotionConfig
    history_folder: str | None = None
    image_folder: str | None = None
    delete_library_queries: list[str] = []
    remove_all_library_images: bool = False

    def provider_context(self) -> ProviderContext:
        """Build the ChatGPT provider context (headers + notion defaults)."""
        headers = {
            "Authorization": f"Bearer {self.account.authorization_token.strip()}",
            "User-Agent": (self.account.user_agent or "").strip(),
            "Content-Type": "application/json",
        }
        cookie_b64 = self.account.cookie_string_base64
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
            notion=self.notion,
            account_name=self.account_name,
        )

    def notion_context(self) -> NotionContext:
        """Build the Notion context (headers + database selection)."""
        notion = self.notion
        return NotionContext(
            headers={
                "Authorization": f"Bearer {(notion.api_key or '').strip()}",
                "Notion-Version": notion.version,
                "Content-Type": "application/json",
            },
            database_id=notion.database_id,
            account_name=self.account_name,
        )

    def validate_runtime(self, required_vars: list[str]) -> None:
        """Raise if any required configuration value is missing."""
        missing: list[str] = []
        for key in required_vars:
            match key:
                case "NOTION_API_KEY":
                    if not (self.notion.api_key or "").strip():
                        missing.append(key)
                case "NOTION_DATABASE_ID":
                    if not (self.notion.database_id or "").strip():
                        missing.append(key)
                case "CHATGPT_AUTHORIZATION_TOKEN":
                    if not self.account.authorization_token.strip():
                        missing.append(key)
                case "CHATGPT_USER_AGENT":
                    if not (self.account.user_agent or "").strip():
                        missing.append(key)
                case "CHATGPT_COOKIE_STRING":
                    if not (self.account.cookie_string_base64 or "").strip():
                        missing.append(key)
        if missing:
            raise ValueError(
                f"Missing required configuration values: {', '.join(missing)}"
            )


class RuntimeOptions(BaseModel):
    """CLI-provided config selectors."""

    model_config = ConfigDict(extra="forbid")

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
