"""Shared re-exports and helpers for upload modes.

This module provides a single namespace that tests patch against.
The three mode implementations (upload_batch, upload_single, upload_sequential)
import from this module for all external dependencies.
"""

import asyncio
from collections import defaultdict

import aiohttp
from tqdm.asyncio import tqdm

from ..adapters import chatgpt_api, notion_api, sqlite_store
from ..adapters.config_loader import resolve_config
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import MAX_CONCURRENT_REQUESTS, image_ext_from_url
from ..shared.http import exc_detail, get_http_timeout
from ..shared.logging import get_logger
from ..shared.verbosity import (
    StageCounter,
    is_verbose,
    log_service_error,
)
from . import history_service
from .history_service import (
    download_all_images,
    fetch_generations_for_conversation,
    fetch_image_generations,
    get_uploaded_generation_ids,
    load_image_generations,
    mark_generations_uploaded,
    save_generations,
)
from .image_service import add_prompt_to_image_single, add_prompt_to_images

logger = get_logger(__name__)


# Re-export modules for downstream imports
# (accessible as shared.sqlite_store, shared.history_service)
sqlite_store = sqlite_store
history_service = history_service


# Re-export commonly patched functions for test compatibility
is_page_exists_in_db = notion_api.is_page_exists_in_db
add_page_to_db = notion_api.add_page_to_db
upload_all_images_to_notion = notion_api.upload_all_images_to_notion
delete_conversation = chatgpt_api.delete_conversation
get_conversations = chatgpt_api.get_conversations
get_conversation_details = chatgpt_api.get_conversation_details
get_conversation_details_batch = chatgpt_api.get_conversation_details_batch
get_conversation_mapping_key_by_asset_pointer = (
    chatgpt_api.get_conversation_mapping_key_by_asset_pointer
)
get_prompt_from_image_node_in_conversation = (
    chatgpt_api.get_prompt_from_image_node_in_conversation
)
get_image_generations = chatgpt_api.get_image_generations
get_headers = chatgpt_api.get_headers


# Lazy wrappers avoid circular imports at module load time.
async def upload_to_notion(*args, **kwargs):
    from .upload_batch import upload_to_notion as _impl

    return await _impl(*args, **kwargs)


async def upload_to_notion_single(*args, **kwargs):
    from .upload_single import upload_to_notion_single as _impl

    return await _impl(*args, **kwargs)


async def upload_to_notion_sequential(*args, **kwargs):
    from .upload_sequential import upload_to_notion_sequential as _impl

    return await _impl(*args, **kwargs)


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
            await chatgpt_api.remove_library_images_by_query(
                session, query=None, headers=headers
            )
        else:
            for query in resolved.delete_library_queries:
                await chatgpt_api.remove_library_images_by_query(
                    session, query=query, headers=headers
                )


async def delete_conversation_of_image_generation_uploaded_to_notion(
    generations: list[ChatGPTImageGeneration],
    db_id: str,
    options: RuntimeOptions | None = None,
) -> None:
    """Delete conversations whose images are all uploaded to Notion."""
    conversation_map: dict[str, set[str]] = defaultdict(set)
    generation_urls: dict[str, str] = {}
    for generation in generations:
        conversation_map[generation.conversation_id].add(generation.id)
        generation_urls[generation.id] = generation.url

    if not conversation_map:
        return

    resolved = resolve_config(options)
    fetch_delay = resolved.fetch_conversation_delay
    fetch_batch_size = resolved.fetch_batch_size
    fetch_batch_pause = 5.0

    pbar = tqdm(total=len(conversation_map), desc="Deleting conversations")
    counter = StageCounter("Deleted")

    async with aiohttp.ClientSession(
        headers=chatgpt_api.get_headers(options), timeout=get_http_timeout()
    ) as session:
        conv_ids = list(conversation_map.keys())
        conv_details_map = await chatgpt_api.get_conversation_details_batch(
            session,
            conv_ids,
            headers=chatgpt_api.get_headers(options),
            delay=fetch_delay,
            batch_size=fetch_batch_size,
            batch_pause=fetch_batch_pause,
        )

        async def delete_by_id(conversation_id: str, image_ids: set[str]):
            if conversation_id not in conv_details_map:
                counter.add("failed")
                if is_verbose():
                    pbar.write(
                        f"❌ Conversation ID {conversation_id} failed: "
                        "details not fetched"
                    )
                pbar.update(1)
                return

            try:
                for image_id in image_ids:
                    ext = image_ext_from_url(generation_urls.get(image_id, ""))
                    file_name = f"{image_id}{ext}"
                    exists = await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    )
                    if not exists:
                        counter.add("skipped")
                        if is_verbose():
                            pbar.write(f"⏭️  {file_name} not found in Notion, skipped")
                        return

                await delete_conversation(
                    session,
                    conversation_id,
                    headers=chatgpt_api.get_headers(options),
                )
                counter.add("success")
                if is_verbose():
                    pbar.write(f"✅ Conversation ID {conversation_id}")
            except Exception as exc:
                counter.add("failed")
                if is_verbose():
                    pbar.write(f"❌ Conversation ID {conversation_id} failed: {exc}")
                log_service_error(
                    logger,
                    "Failed to delete conversation " + conversation_id,
                    exc_detail(exc),
                )
            finally:
                pbar.update(1)

        await asyncio.gather(
            *[delete_by_id(conv_id, ids) for conv_id, ids in conversation_map.items()],
            return_exceptions=True,
        )

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    print()


async def delete_conversations_after_upload(
    generations: list[ChatGPTImageGeneration],
    db_id: str,
    account: str | None,
    uploaded_ids: set[str],
    check_notion_api: bool = False,
    options: RuntimeOptions | None = None,
) -> None:
    del account
    conversation_map: dict[str, set[str]] = defaultdict(set)
    generation_urls: dict[str, str] = {}
    for generation in generations:
        conversation_map[generation.conversation_id].add(generation.id)
        generation_urls[generation.id] = generation.url

    if not conversation_map:
        return

    pbar = tqdm(total=len(conversation_map), desc="Deleting conversations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    counter = StageCounter("Deleted")

    async with aiohttp.ClientSession(
        headers=chatgpt_api.get_headers(options), timeout=get_http_timeout()
    ) as chatgpt_session:

        async def delete_conv(conv_id: str):
            async with semaphore:
                image_ids = conversation_map[conv_id]
                remaining = image_ids - uploaded_ids
                if remaining:
                    counter.add("skipped")
                    if is_verbose():
                        pbar.write(
                            f"⏭️  Conversation {conv_id} skipped, "
                            f"{len(remaining)} image(s) not uploaded"
                        )
                    pbar.update(1)
                    return
                try:
                    if check_notion_api:
                        for image_id in image_ids:
                            ext = image_ext_from_url(generation_urls.get(image_id, ""))
                            file_name = f"{image_id}{ext}"
                            exists = await is_page_exists_in_db(
                                chatgpt_session,
                                db_id,
                                file_name,
                                options=options,
                            )
                            if not exists:
                                counter.add("skipped")
                                if is_verbose():
                                    pbar.write(
                                        f"⏭️  Conversation {conv_id} skipped, "
                                        f"{file_name} not found in Notion"
                                    )
                                pbar.update(1)
                                return
                    await delete_conversation(
                        chatgpt_session,
                        conv_id,
                        headers=chatgpt_api.get_headers(options),
                    )
                    counter.add("success")
                    if is_verbose():
                        pbar.write(f"✅ Conversation {conv_id}")
                except Exception as exc:
                    counter.add("failed")
                    if is_verbose():
                        pbar.write(f"❌ Conversation {conv_id} failed: {exc}")
                    log_service_error(
                        logger,
                        "Failed to delete conversation " + conv_id,
                        exc_detail(exc),
                    )
                finally:
                    pbar.update(1)

        await asyncio.gather(
            *[delete_conv(conv_id) for conv_id in conversation_map],
            return_exceptions=True,
        )

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    print()


__all__ = [
    "upload_to_notion",
    "upload_to_notion_single",
    "upload_to_notion_sequential",
    "fetch_image_generations",
    "download_all_images",
    "add_prompt_to_images",
    "upload_all_images_to_notion",
    "save_generations",
    "load_image_generations",
    "get_uploaded_generation_ids",
    "mark_generations_uploaded",
    "is_page_exists_in_db",
    "delete_conversation",
    "delete_conversation_of_image_generation_uploaded_to_notion",
    "delete_conversations_after_upload",
    "add_prompt_to_image_single",
    "add_page_to_db",
    "fetch_generations_for_conversation",
    "_remove_library_images",
    "get_conversations",
    "get_conversation_details",
    "get_conversation_details_batch",
    "get_conversation_mapping_key_by_asset_pointer",
    "get_prompt_from_image_node_in_conversation",
    "get_image_generations",
    "get_headers",
    "sqlite_store",
    "history_service",
]
