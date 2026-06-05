"""History retrieval and download orchestration."""

import asyncio
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from ..adapters import chatgpt_api, sqlite_store
from ..adapters.config_loader import get_provider_context
from ..domain.models import ChatGPTImageGeneration, RuntimeOptions
from ..shared.constants import (
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
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
from ..shared.verbosity import StageCounter, is_verbose, write_fail_log

logger = get_logger("history")

get_image_generations = chatgpt_api.get_image_generations
get_conversation_details = chatgpt_api.get_conversation_details
get_prompt_from_image_node_in_conversation = (
    chatgpt_api.get_prompt_from_image_node_in_conversation
)


def save_generations(
    account: str,
    data: Sequence[dict] | Sequence[BaseModel],
    keep_days: int = 3,
    display_days: int = 1,
) -> None:
    if len(data) == 0:
        print("No generations to save.")
        return

    generations: list[ChatGPTImageGeneration] = []
    for item in data:
        if isinstance(item, ChatGPTImageGeneration):
            generations.append(item)
        elif isinstance(item, BaseModel):
            generations.append(ChatGPTImageGeneration(**item.model_dump()))
        elif isinstance(item, dict):
            generations.append(ChatGPTImageGeneration(**item))
    sqlite_store.upsert_generations(account, generations)
    sqlite_store.delete_old_generations(account, keep_days=keep_days)

    cutoff = datetime.now(timezone.utc) - timedelta(days=display_days)
    today_count = sqlite_store.count_recent_generations(account, cutoff.isoformat())
    print(
        f"✅ Saved generations for account '{account}' (Total Today: {today_count})\n"
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


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    return get_provider_context("chatgpt", options).headers


async def fetch_image_generations(
    limit: int = 100,
    options: RuntimeOptions | None = None,
    no_cache: bool = False,
) -> tuple[list[ChatGPTImageGeneration], list[ChatGPTImageGeneration]]:
    del no_cache
    account = options.account if options and options.account else "default"
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    headers = get_headers(options)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:
        data = await get_image_generations(session, headers=headers, limit=limit)

        items = data.get("items", [])
        api_ids = {item["id"] for item in items}
        existing_ids = sqlite_store.get_existing_ids(account, api_ids)
        new_items = [item for item in items if item["id"] not in existing_ids]

        if new_items:
            pbar = tqdm(total=len(new_items), desc="Fetching new generation details")
            counter = StageCounter("Fetched")

            async def fetch_generation_details(
                image_generation: dict[str, object],
            ) -> ChatGPTImageGeneration | None:
                async with semaphore:
                    try:
                        detail = await get_conversation_details(
                            session,
                            str(image_generation["conversation_id"]),
                            headers=headers,
                        )
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
                        if is_verbose():
                            pbar.write(f"✅ img ID {image_generation['id']}")
                        counter.add("success")
                        return ChatGPTImageGeneration(
                            created_at=created_at,
                            id=str(image_generation["id"]),
                            conversation_id=str(image_generation["conversation_id"]),
                            message_id=str(image_generation["message_id"]),
                            asset_pointer=str(image_generation["asset_pointer"]),
                            url=str(image_generation["url"]),
                            prompt=prompt or "",
                        )
                    except Exception as exc:
                        counter.add("failed")
                        if is_verbose():
                            pbar.write(
                                f"❌ img ID {image_generation['id']} failed: {exc}"
                            )
                        detail = exc_detail(exc)
                        if is_verbose():
                            logger.exception(
                                "Failed to fetch image generation %s\n%s",
                                image_generation["id"],
                                detail,
                            )
                        else:
                            logger.error(
                                "Failed to fetch image generation %s: %s",
                                image_generation["id"],
                                detail,
                            )
                        return None
                    finally:
                        pbar.update(1)

            results = await asyncio.gather(
                *[fetch_generation_details(item) for item in new_items],
                return_exceptions=True,
            )
            pbar.close()
            if not is_verbose():
                print(counter.summary_line())
            print()

            new_generations = [
                generation
                for generation in results
                if generation is not None
                and not isinstance(generation, Exception)
                and isinstance(generation, ChatGPTImageGeneration)
            ]
            await sqlite_store.async_upsert_generations(account, new_generations)

        generations = sqlite_store.get_generations(
            account,
            include_uploaded=True,
            ids_filter=api_ids,
        )
        remote_generations = [gen for gen in generations if gen.id in api_ids]
        return remote_generations, generations


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


async def download_all_images(
    generations: list[ChatGPTImageGeneration],
    download_folder: str,
    options: RuntimeOptions | None = None,
    fail_log_path: Path | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")
    counter = StageCounter("Downloaded")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    download_path = Path(download_folder)

    results: list[str] = []

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_download_timeout()
    ) as session:

        async def download(row: ChatGPTImageGeneration) -> str:
            async with semaphore:
                ext = image_ext_from_url(row.url)
                file_name = f"{row.id}{ext}"
                file_path = download_path / file_name

                if os.path.exists(file_path):
                    counter.add("skipped")
                    if is_verbose():
                        pbar.write(f"⏭️  {file_name} skipped, already exists")
                    pbar.update(1)
                    return "skipped"

                try:
                    await download_image(
                        session, row.url, str(file_path), headers=get_headers(options)
                    )
                    counter.add("success")
                    if is_verbose():
                        pbar.write(f"✅ {file_name}")
                    pbar.update(1)
                    return "success"
                except Exception as exc:
                    counter.add("failed")
                    if is_verbose():
                        pbar.write(f"❌ {file_name} failed: {exc}")
                    detail = exc_detail(exc)
                    if is_verbose():
                        logger.exception(
                            "Failed to download %s\n%s",
                            file_name,
                            detail,
                        )
                    else:
                        logger.error("Failed to download %s: %s", file_name, detail)
                    if fail_log_path:
                        write_fail_log(
                            fail_log_path,
                            {"stage": "download", "file": file_name, "error": str(exc)},
                        )
                    pbar.update(1)
                    return "failed"

        results = await asyncio.gather(*[download(row) for row in generations])

    pbar.close()
    if not is_verbose():
        print(counter.summary_line())
    logger.info(
        "Download summary: %d success, %d skipped, %d failed",
        results.count("success"),
        results.count("skipped"),
        results.count("failed"),
    )
    print()
