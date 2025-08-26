from abc import ABC, abstractmethod
from logic.sanbo_navi.scr.loader import get_resource_paths
from dotenv import load_dotenv
import os
from openai import OpenAI


class AIConfigBase(ABC):
    @abstractmethod
    def get_client(self):
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass


class OpenAIConfig(AIConfigBase):
    def __init__(self):
        env_url = get_resource_paths().get("ENV_PATH")
        load_dotenv(dotenv_path=env_url)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = self._create_client()

    def _create_client(self) -> OpenAI | None:
        if not self.api_key or not (
            self.api_key.startswith("sk-") or self.api_key.startswith("sk-proj-")
        ):
            return None
        return OpenAI(api_key=self.api_key)

    def get_client(self):
        return self.client

    def is_valid(self) -> bool:
        if not self.api_key:
            self.error_message = "OPENAI_API_KEY が設定されていません。"
            return False

        if not (self.api_key.startswith("sk-") or self.api_key.startswith("sk-proj-")):
            self.error_message = "OPENAI_API_KEY の形式が無効です（'sk-' または 'sk-proj-' で始まっている必要があります）。"
            return False

        if self.client is None:
            self.error_message = "OpenAI クライアントの作成に失敗しました。"
            return False

        self.error_message = None
        return True


# --- AI クライアントの読み込み ---
def load_ai(config_class=OpenAIConfig):
    config = config_class()
    client = config.get_client()
    return client if config.is_valid() else None
