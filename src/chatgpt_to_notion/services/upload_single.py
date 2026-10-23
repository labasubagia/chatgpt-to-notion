"""Single upload mode: per-conversation parallel, conversations parallel."""

import asyncio
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

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
from . import upload_pipeline as shared

logger = get_logger(__name__)


async def upload_to_notion_single(
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
) -> None:
    if not account:
        raise ValueError("account is required")
    if not db_id:
        raise ValueError("db_id must not be empty")

    resolved_image_folder = resolve_image_folder(image_folder, options)

    remote_generations: list[ChatGPTImageGeneration] = []
    if from_history:
        generations = shared.load_image_generations(
            account=account,
            include_uploaded=check_notion_api,
            keep_days=keep_days,
            timezone_name=timezone_name,
            options=options,
        )
    else:
        remote_generations, generations = await shared.fetch_image_generations(
            limit=limit,
            options=options,
            no_cache=no_cache,
        )

    if not generations:
        print("No generations found.")
        if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
            await shared._remove_library_images(
                options, remove_all=remove_all_in_chatgpt_library
            )
        return

    if account and not from_history:
        shared.save_generations(account=account, data=generations)

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

    uploaded_ids = shared.get_uploaded_generation_ids(account)
    pbar = tqdm(total=len(conversation_map), desc="Processing conversations")
    semaphore = asyncio.Semaphore(max_concurrent)
    counter = StageCounter("Processed")

    async with aiohttp.ClientSession(
        headers=shared.get_headers(options), timeout=get_http_timeout()
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

    if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
        await shared._remove_library_images(
            options, remove_all=remove_all_in_chatgpt_library
        )
