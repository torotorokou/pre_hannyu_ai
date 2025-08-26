# app/config/loader/main_path_resolver.py

from pathlib import Path
import yaml
import os
from typing import Optional, Union


class MainPath:
    def __init__(self, config_path: str = "config/main_paths.yaml"):
        self.base_dir = BaseDirProvider().get_base_dir()
        config_dict = YamlLoader(self.base_dir).load(config_path)
        self.resolver = MainPathResolver(config_dict, self.base_dir)

    def get_path(
        self, keys: Union[str, list[str]], section: Optional[str] = None
    ) -> Path:
        return self.resolver.get_path(keys, section)

    def get_config(self) -> dict:
        return self.resolver.config_data


class BaseDirProvider:
    def __init__(self, default_path: str = "/work/app"):
        self.base_dir = Path(os.getenv("BASE_DIR", default_path))

    def get_base_dir(self) -> Path:
        return self.base_dir


class YamlLoader:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load(self, relative_path: str) -> dict:
        full_path = self.base_dir / relative_path
        with open(full_path, encoding="utf-8") as f:
            return yaml.safe_load(f)


class MainPathResolver:
    def __init__(self, config_data: dict, base_dir: Path):
        self.config_data = config_data
        self.base_dir = base_dir

    def get_path(
        self, keys: Union[str, list[str]], section: Optional[str] = None
    ) -> Path:
        target = self.config_data
        if section:
            target = target.get(section, {})
            if target is None:
                raise KeyError(f"セクション '{section}' が見つかりません")

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            target = target.get(key)
            if target is None:
                raise KeyError(f"キー '{'.'.join(keys)}' が見つかりません")

        return self.base_dir / target
