"""ChatGPT API adapter."""

from typing import Any

import aiohttp

from ..domain.models import RuntimeOptions
from ..shared.http import raise_for_status_with_detail, retry_http
from .config_loader import get_provider_context

BASE_URL = "https://chatgpt.com/backend-api"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    return get_provider_context("chatgpt", options).headers


@retry_http()
async def get_conversations(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    offset: int = 0,
    limit: int = 100,
    is_archived: bool = False,
    is_starred: bool = False,
    order: str = "updated",
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/conversations",
        headers=headers or get_headers(),
        params={
            "offset": offset,
            "limit": limit,
            "order": order,
            "is_archived": str(is_archived).lower(),
            "is_starred": str(is_starred).lower(),
        },
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


@retry_http()
async def get_conversation_details(
    session: aiohttp.ClientSession,
    conversation_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers or get_headers(),
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


@retry_http()
async def delete_conversation(
    session: aiohttp.ClientSession,
    conversation_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.patch(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers or get_headers(),
        json={"is_visible": False},
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


@retry_http()
async def get_image_generations(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/my/recent/image_gen",
        headers=headers or get_headers(),
        params={"limit": limit},
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


def get_conversation_mapping_key_by_asset_pointer(
    data: dict[str, Any], asset_pointer: str
) -> str | None:
    for key, node in data.get("mapping", {}).items():
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and part.get("asset_pointer") == asset_pointer:
                return key
    return None


def get_prompt_from_image_node_in_conversation(
    data: dict[str, Any], start_node_id: str, asset_pointer: str
) -> str | None:
    mapping = data["mapping"]
    current_id: str | None = start_node_id
    if current_id not in mapping:
        current_id = get_conversation_mapping_key_by_asset_pointer(data, asset_pointer)
        if current_id is None:
            return None

    while current_id:
        node = mapping[current_id]
        message = node.get("message")

        if message:
            author = message.get("author", {}).get("role")
            if author == "user":
                inputs = message.get("content", {}).get("parts", [])
                for part in inputs:
                    if isinstance(part, str):
                        return part
        current_id = node.get("parent")

    return None
