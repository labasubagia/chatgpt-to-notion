"""Shared re-exports and helpers for upload modes.

This module provides a single namespace that tests patch against.
The sequential mode implementation (upload_sequential)
imports from this module for all external dependencies.
"""

import aiohttp

from ..adapters import chatgpt_api, notion_api, sqlite_store
from ..adapters.config_loader import resolve_config
from ..domain.models import RuntimeOptions
from ..shared.http import get_http_timeout
from ..shared.logging import get_logger
from . import history_service
from .history_service import (
    fetch_generations_for_conversation,
    get_uploaded_generation_ids,
    load_image_generations,
    mark_generations_uploaded,
)
from .image_service import add_prompt_to_image_single

logger = get_logger(__name__)


# Re-export modules for downstream imports
# (accessible as shared.sqlite_store, shared.history_service)
sqlite_store = sqlite_store
history_service = history_service


# Re-export commonly patched functions for test compatibility
is_page_exists_in_db = notion_api.is_page_exists_in_db
add_page_to_db = notion_api.add_page_to_db
delete_conversation = chatgpt_api.delete_conversation
get_conversation_details = chatgpt_api.get_conversation_details
get_image_generations = chatgpt_api.get_image_generations
get_headers = chatgpt_api.get_headers


async def _remove_library_images(
    options: RuntimeOptions | None,
    remove_all: bool = False,
) -> None:
    resolved = resolve_config(options)
    effective_remove_all = remove_all or resolved.remove_all_library_images
    if not effective_remove_all and not resolved.delete_library_queries:
        return
    headers = chatgpt_api.get_headers(options)
    async with aiohttp.ClientSession(
        headers=headers, timeout=get_http_timeout()
    ) as session:
        await chatgpt_api.count_library_images(session, headers=headers)
        if effective_remove_all:
            await chatgpt_api.delete_library_images_by_query(
                session, query=None, headers=headers
            )
        else:
            for query in resolved.delete_library_queries:
                await chatgpt_api.delete_library_images_by_query(
                    session, query=query, headers=headers
                )


__all__ = [
    "load_image_generations",
    "get_uploaded_generation_ids",
    "mark_generations_uploaded",
    "is_page_exists_in_db",
    "delete_conversation",
    "add_prompt_to_image_single",
    "add_page_to_db",
    "fetch_generations_for_conversation",
    "_remove_library_images",
    "get_conversation_details",
    "get_image_generations",
    "get_headers",
    "sqlite_store",
    "history_service",
]
