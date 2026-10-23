"""Sequential upload mode: one conversation at a time end-to-end."""

import aiohttp
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from . import upload_pipeline as shared
from ..adapters.config_loader import resolve_config
from ..adapters.filesystem import resolve_image_folder
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import image_ext_from_url
from ..shared.http import exc_detail, get_http_timeout
from ..shared.logging import get_logger
from ..shared.verbosity import (
    StageCounter,
    is_verbose,
    log_service_error,
    write_fail_log,
)

logger = get_logger(__name__)


async def upload_to_notion_sequential(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    remove_in_chatgpt_library: bool = True,
    remove_all_in_chatgpt_library: bool = False,
    add_prompt_to_image: bool = True,
    account: str | None = None,
    check_notion_api: bool = False,
    from_history: bool = False,
    limit: int = 100,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    no_cache: bool = False,
    options: RuntimeOptions | None = None,
    fail_log_path: Path | None = None,
    max_concurrent: int = 1,
    conversation_delay: float = 2.0,
) -> None:
    """Process conversations sequentially (one at a time) to minimize 429 errors.

    Each conversation is fully processed (download images, add prompts,
    upload to Notion) before moving to the next conversation.
    A delay is added between conversations.
    """
    if not db_id:
        raise ValueError("db_id must not be empty")
    if not account:
        raise ValueError("account is required")

    resolved_image_folder = resolve_image_folder(image_folder, options)

    uploaded_ids = shared.get_uploaded_generation_ids(account)
    counter = StageCounter("Processed")

    async with aiohttp.ClientSession(
        headers=shared.get_headers(options), timeout=get_http_timeout()
    ) as chatgpt_session:
        async with aiohttp.ClientSession(timeout=get_http_timeout()) as notion_session:
            # Build the per-conversation work list up front.
            # from_history: prompts already in the local DB.
            # otherwise: only the lightweight image list is fetched here;
            # each conversation's detail is fetched lazily inside the loop.
            conv_payloads: list[tuple[str, Any]]
            if from_history:
                generations = shared.load_image_generations(
                    account=account,
                    include_uploaded=check_notion_api,
                    keep_days=keep_days,
                    timezone_name=timezone_name,
                    options=options,
                )
                conversation_map: dict[str, list[ChatGPTImageGeneration]] = defaultdict(
                    list
                )
                for g in generations:
                    conversation_map[g.conversation_id].append(g)
                conv_payloads = list(conversation_map.items())
                existing_ids: set[str] = set()
            else:
                data = await shared.get_image_generations(
                    chatgpt_session,
                    headers=shared.get_headers(options),
                    limit=limit,
                )
                items = data.get("items", [])
                api_ids = {str(item["id"]) for item in items}
                existing_ids = shared.sqlite_store.get_existing_ids(account, api_ids)
                conv_items_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for item in items:
                    conv_items_map[str(item["conversation_id"])].append(item)
                conv_payloads = list(conv_items_map.items())

            if not conv_payloads:
                print("No generations found.")
                if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
                    await shared._remove_library_images(
                        options, remove_all=remove_all_in_chatgpt_library
                    )
                return

            pbar = tqdm(
                total=len(conv_payloads),
                desc="Processing conversations sequentially",
            )

            async def process_one(generation: ChatGPTImageGeneration) -> bool:
                ext = image_ext_from_url(generation.url)
                file_name = f"{generation.id}{ext}"
                try:
                    if generation.id in uploaded_ids and not check_notion_api:
                        if is_verbose():
                            pbar.write(f"⏭️  {file_name} skipped, already uploaded")
                        return True

                    if await shared.is_page_exists_in_db(
                        notion_session, db_id, file_name, options=options
                    ):
                        shared.mark_generations_uploaded(account, {generation.id})
                        uploaded_ids.add(generation.id)
                        if is_verbose():
                            pbar.write(f"⏭️  {file_name} skipped, already in Notion")
                        return True

                    file_path = Path(resolved_image_folder) / file_name
                    if not file_path.exists():
                        await shared.history_service.download_image(
                            chatgpt_session,
                            generation.url,
                            str(file_path),
                            headers=shared.get_headers(options),
                        )

                    if add_prompt_to_image:
                        shared.add_prompt_to_image_single(
                            generation,
                            str(resolved_image_folder),
                        )

                    if upload_to_notion:
                        await shared.add_page_to_db(
                            notion_session,
                            db_id,
                            str(file_path),
                            generation.prompt,
                            options=options,
                        )

                    shared.mark_generations_uploaded(account, {generation.id})
                    uploaded_ids.add(generation.id)
                    if is_verbose():
                        pbar.write(f"✅ {file_name} uploaded")
                    return True
                except Exception as exc:
                    log_service_error(
                        logger,
                        "Failed to process " + file_name,
                        exc_detail(exc),
                    )
                    if fail_log_path:
                        write_fail_log(
                            fail_log_path,
                            {
                                "id": generation.id,
                                "conversation_id": generation.conversation_id,
                                "url": generation.url,
                                "prompt": generation.prompt,
                                "stage": "upload",
                                "file": file_name,
                                "error": str(exc),
                            },
                        )
                    return False

            for idx, (conv_id, payload) in enumerate(conv_payloads):
                if from_history:
                    conv_generations: list[ChatGPTImageGeneration] = payload
                else:
                    new_generations = await shared.fetch_generations_for_conversation(
                        chatgpt_session,
                        conv_id,
                        payload,
                        shared.get_headers(options),
                        existing_ids,
                    )
                    if new_generations:
                        await shared.sqlite_store.async_upsert_generations(
                            account, new_generations
                        )
                        existing_ids.update(g.id for g in new_generations)
                    conv_ids = {str(item["id"]) for item in payload}
                    conv_generations = shared.sqlite_store.get_generations(
                        account,
                        include_uploaded=True,
                        ids_filter=conv_ids,
                    )

                results = []
                for g in conv_generations:
                    results.append(await process_one(g))

                all_ok = all(r is True for r in results)
                if all_ok:
                    # In API mode every image in the conversation came from
                    # this run, so the conversation is eligible for deletion.
                    if remove_in_chatgpt and not from_history:
                        try:
                            await shared.delete_conversation(
                                chatgpt_session,
                                conv_id,
                                headers=shared.get_headers(options),
                            )
                            if is_verbose():
                                pbar.write(f"✅ Conversation {conv_id} deleted")
                        except Exception as exc:
                            log_service_error(
                                logger,
                                "Failed to delete conversation " + conv_id,
                                exc_detail(exc),
                            )
                    counter.add("success")
                else:
                    failed = sum(1 for r in results if r is not True)
                    if is_verbose():
                        pbar.write(
                            f"⚠️  Conversation {conv_id} skipped, "
                            f"{failed} image(s) failed"
                        )
                    counter.add("failed")
                pbar.update(1)

                # Delay between conversations to avoid rate limits
                if conversation_delay > 0 and idx < len(conv_payloads) - 1:
                    await asyncio.sleep(conversation_delay)

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    print()

    if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
        await shared._remove_library_images(
            options, remove_all=remove_all_in_chatgpt_library
        )
