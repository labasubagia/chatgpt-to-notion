"""Main upload workflow orchestration."""

import asyncio
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

from ..adapters import chatgpt_api, notion_api, sqlite_store
from ..adapters.filesystem import resolve_image_folder
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import MAX_CONCURRENT_REQUESTS, image_ext_from_url
from ..shared.http import get_http_timeout
from ..shared.logging import get_logger
from ..shared.verbosity import StageCounter, is_verbose, write_fail_log
from . import history_service
from .history_service import (
    download_all_images,
    fetch_image_generations,
    get_uploaded_generation_ids,
    load_image_generations,
    mark_generations_uploaded,
    save_generations,
)
from .image_service import add_prompt_to_image_single, add_prompt_to_images

logger = get_logger("upload_pipeline")

get_conversations = chatgpt_api.get_conversations
get_conversation_details = chatgpt_api.get_conversation_details
get_headers = chatgpt_api.get_headers
get_conversation_mapping_key_by_asset_pointer = (
    chatgpt_api.get_conversation_mapping_key_by_asset_pointer
)
get_prompt_from_image_node_in_conversation = (
    chatgpt_api.get_prompt_from_image_node_in_conversation
)
get_image_generations = chatgpt_api.get_image_generations
delete_conversation = chatgpt_api.delete_conversation
is_page_exists_in_db = notion_api.is_page_exists_in_db
upload_all_images_to_notion = notion_api.upload_all_images_to_notion
add_page_to_db = notion_api.add_page_to_db


async def delete_conversation_of_image_generation_uploaded_to_notion(
    generations: list[ChatGPTImageGeneration],
    db_id: str,
    options: RuntimeOptions | None = None,
) -> None:
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
    ) as session:

        async def delete_by_id(conversation_id: str, image_ids: set[str]):
            async with semaphore:
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
                                pbar.write(
                                    f"⏭️  {file_name} not found in Notion, skipped"
                                )
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
                        pbar.write(
                            f"❌ Conversation ID {conversation_id} failed: {exc}"
                        )
                    logger.exception(
                        "Failed to delete conversation %s", conversation_id
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
                    logger.exception("Failed to delete conversation %s", conv_id)
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


async def upload_to_notion(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
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
) -> None:
    if not db_id:
        raise ValueError("db_id must not be empty")

    resolved_image_folder = resolve_image_folder(image_folder, options)
    sqlite_store.init_db()

    remote_generations: list[ChatGPTImageGeneration] = []
    if from_history:
        if not account:
            raise ValueError("account is required when from_history=True")
        generations = load_image_generations(
            account=account,
            include_uploaded=check_notion_api,
            keep_days=keep_days,
            timezone_name=timezone_name,
            options=options,
        )
    else:
        remote_generations, generations = await fetch_image_generations(
            limit=limit,
            options=options,
            no_cache=no_cache,
        )

    if not generations:
        print("No generations found.")
        return

    if account and not from_history:
        save_generations(account=account, data=generations)

    await download_all_images(
        generations=generations,
        download_folder=str(resolved_image_folder),
        options=options,
        fail_log_path=fail_log_path,
    )

    if add_prompt_to_image:
        add_prompt_to_images(generations=generations, folder=str(resolved_image_folder))

    if upload_to_notion:
        await upload_all_images_to_notion(
            generations=generations,
            db_id=db_id,
            image_folder=str(resolved_image_folder),
            account=account,
            check_notion_api=check_notion_api,
            options=options,
            fail_log_path=fail_log_path,
        )

    if len(remote_generations) <= 0:
        return

    if remove_in_chatgpt:
        await delete_conversation_of_image_generation_uploaded_to_notion(
            generations=remote_generations,
            db_id=db_id,
            options=options,
        )


async def upload_to_notion_single(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
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
) -> None:
    if not account:
        raise ValueError("account is required")
    if not db_id:
        raise ValueError("db_id must not be empty")

    resolved_image_folder = resolve_image_folder(image_folder, options)
    sqlite_store.init_db()

    remote_generations: list[ChatGPTImageGeneration] = []
    if from_history:
        generations = load_image_generations(
            account=account,
            include_uploaded=check_notion_api,
            keep_days=keep_days,
            timezone_name=timezone_name,
            options=options,
        )
    else:
        remote_generations, generations = await fetch_image_generations(
            limit=limit,
            options=options,
            no_cache=no_cache,
        )

    if not generations:
        print("No generations found.")
        return

    if account and not from_history:
        save_generations(account=account, data=generations)

    uploaded_ids = get_uploaded_generation_ids(account)
    db_lock = asyncio.Lock()
    pbar = tqdm(total=len(generations), desc="Processing files")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    counter = StageCounter("Processed")

    async with aiohttp.ClientSession(
        headers=chatgpt_api.get_headers(options), timeout=get_http_timeout()
    ) as chatgpt_session:
        async with aiohttp.ClientSession(timeout=get_http_timeout()) as notion_session:

            async def process_one(generation: ChatGPTImageGeneration):
                async with semaphore:
                    ext = image_ext_from_url(generation.url)
                    file_name = f"{generation.id}{ext}"
                    try:
                        file_path = Path(resolved_image_folder) / file_name
                        if not file_path.exists():
                            await history_service.download_image(
                                chatgpt_session,
                                generation.url,
                                str(file_path),
                                headers=chatgpt_api.get_headers(options),
                            )

                        # Claim the ID under lock to prevent duplicate uploads
                        async with db_lock:
                            if generation.id in uploaded_ids and not check_notion_api:
                                counter.add("skipped")
                                if is_verbose():
                                    pbar.write(
                                        f"⏭️  {file_name} skipped, already uploaded"
                                    )
                                return
                            uploaded_ids.add(generation.id)

                        if await is_page_exists_in_db(
                            notion_session, db_id, file_name, options=options
                        ):
                            async with db_lock:
                                mark_generations_uploaded(account, {generation.id})
                            counter.add("skipped")
                            if is_verbose():
                                pbar.write(f"⏭️  {file_name} skipped, already in Notion")
                            return

                        if add_prompt_to_image:
                            add_prompt_to_image_single(
                                generation,
                                str(resolved_image_folder),
                            )

                        if upload_to_notion:
                            await add_page_to_db(
                                notion_session,
                                db_id,
                                str(file_path),
                                generation.prompt,
                                options=options,
                            )

                        async with db_lock:
                            mark_generations_uploaded(account, {generation.id})
                        counter.add("uploaded")
                        if is_verbose():
                            pbar.write(f"✅ {file_name} uploaded")
                    except Exception as exc:
                        # Release the claim so another attempt can retry
                        async with db_lock:
                            uploaded_ids.discard(generation.id)
                        counter.add("failed")
                        if is_verbose():
                            pbar.write(f"❌ {file_name} failed: {exc}")
                        logger.exception("Failed to process %s", file_name)
                        if fail_log_path:
                            write_fail_log(
                                fail_log_path,
                                {
                                    "stage": "upload",
                                    "file": file_name,
                                    "error": str(exc),
                                },
                            )
                    finally:
                        pbar.update(1)

            await asyncio.gather(
                *[process_one(generation) for generation in generations],
                return_exceptions=True,
            )

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    print()

    if len(remote_generations) <= 0:
        return

    if remove_in_chatgpt:
        await delete_conversations_after_upload(
            generations=remote_generations,
            db_id=db_id,
            account=account,
            uploaded_ids=uploaded_ids,
            check_notion_api=check_notion_api,
            options=options,
        )
