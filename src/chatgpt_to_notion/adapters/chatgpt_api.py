"""ChatGPT API adapter."""

import asyncio
import json
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from ..domain.models import RuntimeOptions
from ..shared.http import raise_for_status_with_detail, retry_http
from .config_loader import get_provider_context

BASE_URL = "https://chatgpt.com/backend-api"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    return get_provider_context(options).headers


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
        if response.status == 404:
            body = await response.text()
            if "conversation_deleted" in body:
                return {"already_deleted": True}
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
    mapping = data.get("mapping", {})
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


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def _try_append(raw: str) -> None:
        s = raw.strip()
        if s and s != "[DONE]":
            try:
                results.append(json.loads(s))
            except json.JSONDecodeError:
                pass

    for event in body.split("\n\n"):
        data_lines: list[str] = []
        for line in event.strip().split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith(("event: ", "id: ", "retry: ")):
                continue
            else:
                data_lines.append(line)
        if data_lines:
            _try_append("".join(data_lines))
    if not results and body.strip():
        for line in body.strip().split("\n"):
            _try_append(line)
    return results


@retry_http()
async def get_library_images(
    session: aiohttp.ClientSession,
    query: str | None = None,
    headers: dict[str, str] | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, str] = {"categories": "image"}
    if query:
        params["q"] = query
    if cursor:
        params["cursor"] = cursor
    async with session.get(
        f"{BASE_URL}/files/library/nodes",
        headers=headers or get_headers(),
        params=params,
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


@retry_http()
async def delete_library_file(
    session: aiohttp.ClientSession,
    item: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    item_id = item["id"]
    params = {
        "file_id": item["file_id"],
        "parent_directory_id": item["parent_directory_id"],
        "file_name": item["name"],
        "soft_delete": "true",
    }
    async with session.post(
        f"{BASE_URL}/files/library/files/{item_id}/delete_stream",
        headers=headers or get_headers(),
        params=params,
    ) as response:
        if response.status == 404:
            return {"already_deleted": True}
        await raise_for_status_with_detail(response)
        body = await response.text()
        events = _parse_sse_events(body)
        return {"success": True, "events": events}


async def count_library_images(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
) -> int:
    headers = headers or get_headers()
    total = 0
    cursor: str | None = None

    while True:
        data = await get_library_images(
            session, query=None, headers=headers, cursor=cursor
        )
        total += len(data.get("items", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    return total


async def delete_library_images_by_query(
    session: aiohttp.ClientSession,
    query: str | None = None,
    headers: dict[str, str] | None = None,
    max_concurrent: int = 10,
) -> int:
    headers = headers or get_headers()
    total_deleted = 0
    cursor: str | None = None
    label = f"Removing images matching '{query}'" if query else "Removing all images"
    pbar = tqdm(desc=label, unit=" item")

    async def _delete_one(item: dict[str, Any], sem: asyncio.Semaphore) -> None:
        async with sem:
            try:
                await delete_library_file(session, item, headers)
            except Exception:
                raise
            finally:
                pbar.update(1)

    try:
        while True:
            data = await get_library_images(session, query, headers, cursor)
            items = data.get("items", [])
            if not items:
                break

            semaphore = asyncio.Semaphore(max_concurrent)
            await asyncio.gather(*[_delete_one(item, semaphore) for item in items])
            total_deleted += len(items)
            cursor = data.get("cursor")
            if not cursor:
                break
    finally:
        pbar.close()

    return total_deleted


class ChatGPTClientImpl:
    """Concrete implementation of ChatGPTClient protocol."""

    def __init__(self, options: RuntimeOptions | None = None):
        self._options = options

    def get_headers(self) -> dict[str, str]:
        return get_headers(self._options)

    async def get_conversation_details(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
    ) -> dict[str, Any]:
        return await get_conversation_details(
            session, conversation_id, headers=self.get_headers()
        )

    async def delete_conversation(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
    ) -> dict[str, Any]:
        return await delete_conversation(
            session, conversation_id, headers=self.get_headers()
        )

    async def get_image_generations(
        self,
        session: aiohttp.ClientSession,
        limit: int = 100,
    ) -> dict[str, Any]:
        return await get_image_generations(
            session, headers=self.get_headers(), limit=limit
        )

    async def get_library_images(
        self,
        session: aiohttp.ClientSession,
        query: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return await get_library_images(
            session, query=query, headers=self.get_headers(), cursor=cursor
        )

    async def get_prompt_from_image_node_in_conversation(
        self,
        session: aiohttp.ClientSession,
        conversation_id: str,
        message_id: str,
        asset_pointer: str,
    ) -> str:
        # Get conversation details first
        conv_data = await get_conversation_details(
            session, conversation_id, headers=self.get_headers()
        )
        # Then extract prompt
        prompt = get_prompt_from_image_node_in_conversation(
            conv_data, message_id, asset_pointer
        )
        return prompt or ""

    async def delete_library_file(
        self,
        session: aiohttp.ClientSession,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        return await delete_library_file(session, item, headers=self.get_headers())


async def remove_library_images(
    options: RuntimeOptions | None = None,
    remove_all: bool = False,
) -> None:
    """Default implementation for remove_library_func protocol.

    Deletes library images based on configured queries
    or all images.
    """
    from ..shared.http import get_http_timeout
    from .config_loader import get_provider_context, resolve_config

    # Get queries from options or resolve from config
    if remove_all:
        queries = None  # Not needed when remove_all=True
    else:
        queries = getattr(options, "delete_library_queries", None) if options else None
        if not queries:
            # Try to resolve from config
            try:
                resolved = resolve_config(options)
                queries = resolved.delete_library_queries
                remove_all = resolved.remove_all_library_images
            except Exception:
                return

    if not remove_all and not queries:
        return

    # Get headers from provider context
    provider_ctx = get_provider_context(options)
    headers = provider_ctx.headers

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:
        # Count total images first (for progress reporting)
        await count_library_images(session, headers=headers)
        if remove_all:
            # Delete all images
            await delete_library_images_by_query(session, query=None, headers=headers)
        else:
            # Delete images matching each query
            for query in queries or []:
                await delete_library_images_by_query(
                    session, query=query, headers=headers
                )
