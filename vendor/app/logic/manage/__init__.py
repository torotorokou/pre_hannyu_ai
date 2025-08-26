# processors/__init__.py
import sys
import os
import importlib
from utils.config_loader import get_template_config
from utils.logger import app_logger


# /work/logic を import path に追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = app_logger()
template_configs = dict(
    list(get_template_config().items())[:5]
)  # 最初の5つのテンプレートのみを取得

template_processors = {}

for key in template_configs.keys():
    try:
        module = importlib.import_module(
            f".{key}", package=__name__
        )  # ← ここがポイント
        func = getattr(module, "process")
        template_processors[key] = func
        # logger.info(f"✅ {key}.py の process 関数を登録しました")
    except Exception as e:
        logger.warning(f"❌ 処理関数の読み込みエラー: {key} → {type(e).__name__}: {e}")
