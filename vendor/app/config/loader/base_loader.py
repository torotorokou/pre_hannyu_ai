from abc import ABC, abstractmethod
from typing import Any


class ConfigLoaderInterface(ABC):
    @abstractmethod
    def load(self) -> Any:
        """設定情報を読み込んで返す

        Returns:
            Any: yaml、csvなど
        """
