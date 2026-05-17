"""Image metadata helpers."""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

from ..domain.models import ImageGeneration
from ..shared.constants import MAX_RETRIES


def edit_png_info(
    file_path: str, payload: dict[str, str], overwrite: bool = True
) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with Image.open(file_path) as image:
        metadata = PngInfo()
        for key, value in image.info.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, str):
                metadata.add_text(key, value)
            elif isinstance(value, int):
                metadata.add_text(key, str(value))
        for key, value in payload.items():
            if overwrite or key not in image.info:
                metadata.add_text(key, value)
        image.save(file_path, pnginfo=metadata)


def get_png_prompt(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with Image.open(file_path) as image:
        prompt = image.info.get("Prompt")
        return prompt if isinstance(prompt, str) else None


def add_prompt_to_image_single(generation: ImageGeneration, folder: str) -> None:
    file_path = Path(folder) / f"{generation.id}.png"
    if not file_path.exists():
        return
    if get_png_prompt(str(file_path)) == generation.prompt:
        return
    edit_png_info(str(file_path), payload={"Prompt": generation.prompt})


def add_prompt_to_images(
    generations: Sequence[ImageGeneration], folder: str, max_workers: int = 10
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Adding prompts to images")

    def add_prompt(row: ImageGeneration):
        file_path = Path(folder) / f"{row.id}.png"
        if not os.path.exists(file_path):
            pbar.write(f"⚠️  {file_path} not found, skipped")
            pbar.update(1)
            return

        for _ in range(MAX_RETRIES):
            try:
                if get_png_prompt(str(file_path)) == row.prompt:
                    pbar.write(f"⏭️  {file_path} skipped, prompt unchanged")
                    pbar.update(1)
                    break
                edit_png_info(str(file_path), payload={"Prompt": row.prompt})
                pbar.write(f"✅ {file_path}")
                pbar.update(1)
                break
            except Exception as exc:
                pbar.write(f"⚠️  {file_path} edit error: {exc}, retrying...")
        else:
            pbar.write(f"❌ {file_path} edit failed after {MAX_RETRIES} retries")
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(add_prompt, row) for row in generations]
        for future in as_completed(futures):
            future.result()

    pbar.close()
    print()
