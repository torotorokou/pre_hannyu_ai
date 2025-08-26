import os

# --- 共通設定 ---

# アプリのベースディレクトリ
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /work/app/

# よく使うパス
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Streamlit共通設定
DEFAULT_PORT = 8501  # 基本ポート
