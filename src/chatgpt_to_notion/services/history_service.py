"""History retrieval and download orchestration."""

import asyncio
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from ..adapters import chatgpt_api, sqlite_store
from ..adapters.config_loader import get_provider_context, resolve_config
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import (
    MAX_CONCURRENT_DOWNLOADS,
    image_ext_from_url,
)
from ..shared.http import (
    exc_detail,
    get_download_timeout,
    get_http_timeout,
    raise_for_status_with_detail,
    retry_http,
)
from ..shared.logging import get_logger
from ..shared.verbosity import (
    StageCounter,
    is_verbose,
    log_service_error,
    write_fail_log,
)

logger = get_logger("history")

get_image_generations = chatgpt_api.get_image_generations
get_conversation_details = chatgpt_api.get_conversation_details
get_prompt_from_image_node_in_conversation = (
    chatgpt_api.get_prompt_from_image_node_in_conversation
)

def get_uploaded_generation_ids(account: str | None) -> set[str]:
    if not account:
        return set()
    return sqlite_store.get_uploaded_ids(account)

def mark_generations_uploaded(account: str | None, generation_ids: set[str]) -> None:
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

def load_image_generations(
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
    session: aiohttp.ClientSession,
    conv_id: str,
    image_items: Sequence[dict[str, Any]],
    headers: dict[str, str],
    existing_ids: set[str],
) -> list[ChatGPTImageGeneration]:
    """Build generations for one conversation.

    Fetches the conversation detail at most once, and only when the
    conversation contains items not yet stored locally.
    """
    new_items = [item for item in image_items if str(item["id"]) not in existing_ids]
    if not new_items:
        return []

    detail = await chatgpt_api.get_conversation_details(session, conv_id, headers)
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
