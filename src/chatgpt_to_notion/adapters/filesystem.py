"""Filesystem and path helpers."""

import os
import shutil
from pathlib import Path

from ..domain.models import RuntimeOptions
from ..shared import constants
from .config_loader import resolve_config


def get_output_path(input_path_str: str, is_dir: bool = False) -> Path:
    input_path = Path(input_path_str)
    if input_path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {input_path_str}")

    base_dir = Path(constants.OUTPUT_PATH).resolve()
    final_path = (base_dir / input_path).resolve()
    if base_dir not in final_path.parents:
        raise ValueError("Path attempts to escape the output directory!")

    if is_dir:
        final_path.mkdir(parents=True, exist_ok=True)
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return final_path


def _resolve_folder_path(folder_str: str | None, default_subpath: str) -> Path:
    if not folder_str:
        return Path(constants.OUTPUT_PATH).resolve() / default_subpath
    path = Path(os.path.expanduser(folder_str))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_history_folder(options: RuntimeOptions | None = None) -> Path:
    options = options or RuntimeOptions()
    if options.history_folder:
        return _resolve_folder_path(options.history_folder, "")
    if options.config_path or options.account:
        try:
            resolved = resolve_config(options)
            if resolved.history_folder:
                return _resolve_folder_path(resolved.history_folder, "")
        except ValueError:
            pass
    return Path(constants.OUTPUT_PATH).resolve() / "history"


def get_image_folder(options: RuntimeOptions | None = None) -> Path:
    options = options or RuntimeOptions()
    if options.image_folder:
        return _resolve_folder_path(options.image_folder, "")
    if options.config_path or options.account:
        try:
            resolved = resolve_config(options)
            if resolved.image_folder:
                return _resolve_folder_path(resolved.image_folder, "")
        except ValueError:
            pass
    return Path(constants.OUTPUT_PATH).resolve() / "images"


def resolve_image_folder(
    image_folder: str | None,
    options: RuntimeOptions | None = None,
) -> Path:
    if image_folder is None:
        return get_image_folder(options)
    path = Path(image_folder)
    if path.is_absolute():
        return path
    return get_output_path(image_folder)


def clean_output_path() -> None:
    base_dir = Path(constants.OUTPUT_PATH).resolve()
    if not base_dir.exists():
        return
    for item in base_dir.iterdir():
        if item.is_file() and item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
