"""Main upload workflow orchestration."""

import asyncio
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

from ..adapters import chatgpt_api, notion_api, sqlite_store
from ..adapters.config_loader import resolve_config
from ..adapters.filesystem import resolve_image_folder
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import MAX_CONCURRENT_REQUESTS, image_ext_from_url
from ..shared.http import exc_detail, get_http_timeout
from ..shared.logging import get_logger
from ..shared.verbosity import (
    StageCounter,
    is_verbose,
    log_service_error,
    write_fail_log,
)
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


async def _remove_library_images(options: RuntimeOptions | None) -> None:
    resolved = resolve_config(options)
    if not resolved.delete_library_queries:
        return
    headers = chatgpt_api.get_headers(options)
    async with aiohttp.ClientSession(
        headers=headers, timeout=get_http_timeout()
    ) as session:
        for query in resolved.delete_library_queries:
            await chatgpt_api.remove_library_images_by_query(
                session, query=query, headers=headers
            )
        remaining = await chatgpt_api.count_library_images(session, headers)
        print(f"Total images remaining in library: {remaining}")


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


async def upload_to_notion(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    remove_in_chatgpt_library: bool = True,
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
        if remove_in_chatgpt_library:
            await _remove_library_images(options)
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
        if remove_in_chatgpt_library:
            await _remove_library_images(options)
        return

    if remove_in_chatgpt:
        await delete_conversation_of_image_generation_uploaded_to_notion(
            generations=remote_generations,
            db_id=db_id,
            options=options,
        )

    if remove_in_chatgpt_library:
        await _remove_library_images(options)


async def upload_to_notion_single(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    remove_in_chatgpt_library: bool = True,
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
        if remove_in_chatgpt_library:
            await _remove_library_images(options)
        return

    if account and not from_history:
        save_generations(account=account, data=generations)

    # Group generations by conversation
    conversation_map: dict[str, list[ChatGPTImageGeneration]] = defaultdict(list)
    for g in generations:
        conversation_map[g.conversation_id].append(g)

    # Conversations eligible for deletion: all images came from API this run
    eligible_convos: set[str] = set()
    if remove_in_chatgpt and remote_generations:
        remote_gen_ids = {g.id for g in remote_generations}
        for conv_id, images in conversation_map.items():
            if all(g.id in remote_gen_ids for g in images):
                eligible_convos.add(conv_id)

    uploaded_ids = get_uploaded_generation_ids(account)
    pbar = tqdm(total=len(conversation_map), desc="Processing conversations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    counter = StageCounter("Processed")

    async with aiohttp.ClientSession(
        headers=chatgpt_api.get_headers(options), timeout=get_http_timeout()
    ) as chatgpt_session:
        async with aiohttp.ClientSession(timeout=get_http_timeout()) as notion_session:

            async def process_one(generation: ChatGPTImageGeneration) -> bool:
                ext = image_ext_from_url(generation.url)
                file_name = f"{generation.id}{ext}"
                try:
                    async with semaphore:
                        if generation.id in uploaded_ids and not check_notion_api:
                            if is_verbose():
                                pbar.write(f"⏭️  {file_name} skipped, already uploaded")
                            return True

                        if await is_page_exists_in_db(
                            notion_session, db_id, file_name, options=options
                        ):
                            mark_generations_uploaded(account, {generation.id})
                            uploaded_ids.add(generation.id)
                            if is_verbose():
                                pbar.write(f"⏭️  {file_name} skipped, already in Notion")
                            return True

                        file_path = Path(resolved_image_folder) / file_name
                        if not file_path.exists():
                            await history_service.download_image(
                                chatgpt_session,
                                generation.url,
                                str(file_path),
                                headers=chatgpt_api.get_headers(options),
                            )

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

                        mark_generations_uploaded(account, {generation.id})
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
                                "stage": "upload",
                                "file": file_name,
                                "error": str(exc),
                            },
                        )
                    return False

            async def process_conversation(
                conv_id: str, images: list[ChatGPTImageGeneration]
            ) -> None:
                results = await asyncio.gather(
                    *[process_one(g) for g in images],
                    return_exceptions=True,
                )
                all_ok = all(r is True for r in results)
                if all_ok:
                    if remove_in_chatgpt and conv_id in eligible_convos:
                        try:
                            await delete_conversation(
                                chatgpt_session,
                                conv_id,
                                headers=chatgpt_api.get_headers(options),
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

            await asyncio.gather(
                *[
                    process_conversation(conv_id, images)
                    for conv_id, images in conversation_map.items()
                ],
                return_exceptions=True,
            )

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    print()

    if remove_in_chatgpt_library:
        await _remove_library_images(options)
