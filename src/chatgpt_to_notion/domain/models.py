"""Type definitions for image generation data structures and app config."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
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


RemoveLibraryFunc = Callable[..., Awaitable[None]]


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
    fetch_conversation_delay: float = 0.0
    fetch_batch_size: int = 0
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
    fetch_conversation_delay: float = 0.0
    fetch_batch_size: int = 0
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
    fetch_conversation_delay: float = 0.0
    fetch_batch_size: int = 0
    remove_all_library_images: bool = False


class RuntimeOptions(BaseModel):
    """CLI-provided config selectors."""

    model_config = ConfigDict(extra="forbid")

    config_path: str | None = None
    account: str | None = None
    history_folder: str | None = None
    image_folder: str | None = None
    # Legacy rate limit fields (for config.toml, not CLI options)
    fetch_conversation_delay: float | None = None
    fetch_batch_size: int | None = None


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
