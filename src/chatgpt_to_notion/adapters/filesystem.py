"""Filesystem and path helpers."""

import os
import shutil
from pathlib import Path

from ..domain.models import RuntimeOptions
from ..shared import constants
from .config_loader import ConfigLoader
from .config_loader import config_loader as default_config_loader


class Filesystem:
    def __init__(self, config_loader: ConfigLoader | None = None):
        self._config_loader = config_loader or default_config_loader

    def output_path(self, input_path_str: str, is_dir: bool = False) -> Path:
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

    def _resolve_folder_path(
        self, folder_str: str | None, default_subpath: str
    ) -> Path:
        if not folder_str:
            return Path(constants.OUTPUT_PATH).resolve() / default_subpath
        path = Path(os.path.expanduser(folder_str))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def image_folder(self, options: RuntimeOptions | None = None) -> Path:
        options = options or RuntimeOptions()
        if options.image_folder:
            return self._resolve_folder_path(options.image_folder, "")
        if options.config_path or options.account:
            try:
                resolved = self._config_loader.resolve(options)
                if resolved.image_folder:
                    return self._resolve_folder_path(resolved.image_folder, "")
            except ValueError:
                pass
        path = Path(constants.OUTPUT_PATH).resolve() / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_image_folder(
        self, image_folder: str | None, options: RuntimeOptions | None = None
    ) -> Path:
        if image_folder is None:
            return self.image_folder(options)
        path = Path(image_folder)
        if path.is_absolute():
            return path
        return self.output_path(image_folder)

    def clean_output_path(self) -> None:
        base_dir = Path(constants.OUTPUT_PATH).resolve()
        if not base_dir.exists():
            return
        # Preserve the SQLite database, which lives inside the output dir
        # (see sqlite_store._get_db_path). Cleaning it would destroy all
        # upload history and the API cache.
        db_path = base_dir / "chatgpt.db"
        for item in base_dir.iterdir():
            if item == db_path:
                continue
            if item.is_file() and item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


filesystem = Filesystem()
