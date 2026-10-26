"""Sequential upload mode: one conversation at a time end-to-end."""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from ..adapters.config_loader import config_loader
from ..adapters.filesystem import Filesystem
from ..adapters.filesystem import filesystem as default_fs
from ..adapters.sqlite_store import SqliteStore
from ..adapters.sqlite_store import sqlite_store as default_store
from ..domain.models import (
    ChatGPTClient,
    ChatGPTImageGeneration,
    HistoryServiceProtocol,
    ImageServiceProtocol,
    NotionClient,
    RemoveLibraryFunc,
    RuntimeOptions,
)
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
    delete_conversation: bool = False,
    delete_library: bool = False,
    delete_library_all: bool = False,
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
    chatgpt_client: ChatGPTClient | None = None,
    notion_client: NotionClient | None = None,
    history_service: HistoryServiceProtocol | None = None,
    image_service: ImageServiceProtocol | None = None,
    remove_library_func: RemoveLibraryFunc | None = None,
    sqlite_store: SqliteStore | None = None,
    filesystem: Filesystem | None = None,
) -> None:
    """Process conversations sequentially (one at a time) to minimize 429 errors.

    Each conversation is fully processed (download images, add prompts,
    upload to Notion) before moving to the next conversation.
    A delay is added between conversations.
    """
    # Import concrete implementations for defaults
    from typing import cast

    from ..adapters.chatgpt_api import ChatGPTClientImpl, remove_library_images
    from ..adapters.notion_api import NotionClientImpl
    from ..services import history_service as hs
    from ..services import image_service as img_svc

    # Use injected or default implementations
    provider_ctx = config_loader.provider_context(options)
    notion_ctx = config_loader.notion_context(options)
    chatgpt_client_impl: ChatGPTClient = chatgpt_client or ChatGPTClientImpl(
        provider_ctx
    )
    notion_client_impl: NotionClient = cast(
        NotionClient, notion_client or NotionClientImpl(notion_ctx)
    )
    history_service_impl: HistoryServiceProtocol = history_service or hs
    image_service_impl: ImageServiceProtocol = image_service or img_svc
    remove_library_func_impl: RemoveLibraryFunc = (
        remove_library_func or remove_library_images
    )
    store = sqlite_store or default_store
    fs = filesystem or default_fs

    if not db_id:
        raise ValueError("db_id must not be empty")
    if not account:
        raise ValueError("account is required")

    resolved_image_folder = fs.resolve_image_folder(image_folder, options)

    store.init_db()

    uploaded_ids = await history_service_impl.get_uploaded_generation_ids(account)
    counter = StageCounter("Processed")

    async with aiohttp.ClientSession(
        headers=chatgpt_client_impl.get_headers(), timeout=get_http_timeout()
    ) as chatgpt_session:
        async with aiohttp.ClientSession(timeout=get_http_timeout()) as notion_session:
            # Build the per-conversation work list up front.
            # from_history: prompts already in the local DB.
            # otherwise: only the lightweight image list is fetched here;
            # each conversation's detail is fetched lazily inside the loop.
            conv_payloads: list[tuple[str, Any]]
            if from_history:
                generations = await history_service_impl.load_image_generations(
                    account=account,
                    include_uploaded=check_notion_api,
                    keep_days=keep_days,
                    timezone_name=timezone_name,
                )
                conversation_map: dict[str, list[ChatGPTImageGeneration]] = defaultdict(
                    list
                )
                for g in generations:
                    conversation_map[g.conversation_id].append(g)
                conv_payloads = list(conversation_map.items())
                existing_ids: set[str] = set()
            else:
                data = await chatgpt_client_impl.get_image_generations(
                    chatgpt_session,
                    limit=limit,
                )
                items = data.get("items", [])
                api_ids = {str(item["id"]) for item in items}
                existing_ids = store.get_existing_ids(account, api_ids)
                conv_items_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for item in items:
                    conv_items_map[str(item["conversation_id"])].append(item)
                conv_payloads = list(conv_items_map.items())
            if not conv_payloads:
                print("No generations found.")
                if delete_library or delete_library_all:
                    await remove_library_func_impl(
                        options, remove_all=delete_library_all
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

                    if await notion_client_impl.is_page_exists_in_db(
                        notion_session, db_id, file_name
                    ):
                        await history_service_impl.mark_generations_uploaded(
                            account, {generation.id}
                        )
                        uploaded_ids.add(generation.id)
                        if is_verbose():
                            pbar.write(f"⏭️  {file_name} skipped, already in Notion")
                        return True

                    file_path = Path(resolved_image_folder) / file_name
                    if not file_path.exists():
                        await history_service_impl.download_image(
                            chatgpt_session,
                            generation.url,
                            str(file_path),
                            headers=chatgpt_client_impl.get_headers(),
                        )

                    if add_prompt_to_image:
                        image_service_impl.add_prompt_to_image_single(
                            generation,
                            str(resolved_image_folder),
                        )

                    if upload_to_notion:
                        await notion_client_impl.add_page_to_db(
                            notion_session, db_id, str(file_path), generation.prompt
                        )

                    await history_service_impl.mark_generations_uploaded(
                        account, {generation.id}
                    )
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
                    func = history_service_impl.fetch_generations_for_conversation
                    new_generations = await func(
                        chatgpt_client_impl,
                        chatgpt_session,
                        conv_id,
                        payload,
                        existing_ids,
                    )
                    if new_generations:
                        await store.async_upsert_generations(
                            account, new_generations
                        )
                        existing_ids.update(g.id for g in new_generations)
                    conv_ids = {str(item["id"]) for item in payload}
                    conv_generations = store.get_generations(
                        account,
                        include_uploaded=True,
                        ids_filter=conv_ids,
                    )
                    conv_generations.sort(key=lambda g: g.created_at)

                results = []
                for g in conv_generations:
                    results.append(await process_one(g))

                all_ok = all(r is True for r in results)
                if all_ok:
                    # In API mode every image in the conversation came from
                    # this run, so the conversation is eligible for deletion.
                    if delete_conversation and not from_history:
                        try:
                            await chatgpt_client_impl.delete_conversation(
                                chatgpt_session, conv_id
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

    if delete_library or delete_library_all:
        await remove_library_func_impl(options, remove_all=delete_library_all)
