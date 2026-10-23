"""Batch upload mode: fetch all, download all, upload all (max parallelism)."""

from pathlib import Path

from . import upload_pipeline as shared
from ..adapters.filesystem import resolve_image_folder
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.logging import get_logger

logger = get_logger(__name__)


async def upload_to_notion(
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
    """Batch mode: fetch all generations, download all images, upload all to Notion."""
    if not db_id:
        raise ValueError("db_id must not be empty")

    resolved_image_folder = resolve_image_folder(image_folder, options)

    remote_generations: list[ChatGPTImageGeneration] = []
    if from_history:
        if not account:
            raise ValueError("account is required when from_history=True")
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

    await shared.download_all_images(
        generations=generations,
        download_folder=str(resolved_image_folder),
        options=options,
        fail_log_path=fail_log_path,
    )

    if add_prompt_to_image:
        shared.add_prompt_to_images(
            generations=generations, folder=str(resolved_image_folder)
        )

    if upload_to_notion:
        await shared.upload_all_images_to_notion(
            generations=generations,
            db_id=db_id,
            image_folder=str(resolved_image_folder),
            account=account,
            check_notion_api=check_notion_api,
            options=options,
            fail_log_path=fail_log_path,
            max_concurrent=max_concurrent,
        )

    if len(remote_generations) <= 0:
        if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
            await shared._remove_library_images(
                options, remove_all=remove_all_in_chatgpt_library
            )
        return

    if remove_in_chatgpt:
        await shared.delete_conversation_of_image_generation_uploaded_to_notion(
            generations=remote_generations,
            db_id=db_id,
            options=options,
        )

    if remove_in_chatgpt_library or remove_all_in_chatgpt_library:
        await shared._remove_library_images(options, remove_all=remove_all_in_chatgpt_library)