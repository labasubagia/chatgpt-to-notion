"""History retrieval and download orchestration."""

import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import aiohttp

from ..adapters import chatgpt_api
from ..adapters.sqlite_store import sqlite_store
from ..domain.models import ChatGPTClient, ChatGPTImageGeneration, RuntimeOptions
from ..shared.http import (
    raise_for_status_with_detail,
    retry_http,
)
from ..shared.logging import get_logger

logger = get_logger("history")

get_prompt_from_image_node_in_conversation = (
    chatgpt_api.get_prompt_from_image_node_in_conversation
)


async def get_uploaded_generation_ids(account: str | None) -> set[str]:
    if not account:
        return set()
    return sqlite_store.get_uploaded_ids(account)


async def mark_generations_uploaded(
    account: str | None, generation_ids: set[str]
) -> None:
    if not account or not generation_ids:
        return
    sqlite_store.mark_uploaded(account, generation_ids)


@retry_http()
async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    file_path: str,
    headers: dict[str, str] | None = None,
) -> None:
    dir_path = os.path.dirname(file_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_path)
    try:
        async with session.get(url, headers=headers or {}) as response:
            await raise_for_status_with_detail(response)
            with os.fdopen(fd, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def load_image_generations(
    account: str,
    include_uploaded: bool = False,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    options: RuntimeOptions | None = None,
) -> list[ChatGPTImageGeneration]:
    del options
    return sqlite_store.get_generations(
        account,
        include_uploaded=include_uploaded,
        keep_days=keep_days,
        timezone_name=timezone_name,
    )


async def fetch_generations_for_conversation(
    client: ChatGPTClient,
    session: aiohttp.ClientSession,
    conv_id: str,
    image_items: Sequence[dict[str, Any]],
    existing_ids: set[str],
) -> list[ChatGPTImageGeneration]:
    """Build generations for one conversation.

    Fetches the conversation detail at most once, and only when the
    conversation contains items not yet stored locally.
    """
    new_items = [item for item in image_items if str(item["id"]) not in existing_ids]
    if not new_items:
        return []
    detail = await client.get_conversation_details(session, conv_id)
    generations: list[ChatGPTImageGeneration] = []
    for image_generation in new_items:
        prompt = get_prompt_from_image_node_in_conversation(
            detail,
            str(image_generation["message_id"]),
            str(image_generation["asset_pointer"]),
        )
        created_at_value = image_generation["created_at"]
        assert isinstance(created_at_value, int | float)
        created_at = datetime.fromtimestamp(
            created_at_value,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds")
        generations.append(
            ChatGPTImageGeneration(
                created_at=created_at,
                id=str(image_generation["id"]),
                conversation_id=conv_id,
                message_id=str(image_generation["message_id"]),
                asset_pointer=str(image_generation["asset_pointer"]),
                url=str(image_generation["url"]),
                prompt=prompt or "",
            )
        )
    return generations
