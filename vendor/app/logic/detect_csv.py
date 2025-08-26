from utils.logger import app_logger  # ← Streamlit環境用のロガー取得
from utils.file_loader import read_csv
from utils.config_loader import receive_header_definition
import pandas as pd
from pathlib import Path

# import io
# import pandas as pd
# from utils.logger import app_logger
# from logic.controllers.header_loader import receive_header_definition, load_template_signatures
# from utils.readers import read_csv  # 独自のread_csvを使っている場合


def load_template_signatures(df) -> dict:
    """
    CSVファイルからテンプレートのカラム情報を辞書として読み込む。
    """
    templates = {}

    for _, row in df.iterrows():
        name = row["template_name"]
        cols = [row[f"column{i}"] for i in range(1, 6) if pd.notna(row[f"column{i}"])]
        templates[name] = cols

    return templates


def detect_csv_type(file) -> str:

    logger = app_logger()

    try:
        logger.info("📥 detect_csv_type(): 開始")

        # 判別ルールの読み込み
        df_csv = receive_header_definition()
        signatures = load_template_signatures(df_csv)

        # ✅ ファイル形式に応じて開く
        if isinstance(file, str) or isinstance(file, Path):
            # ファイルパスの場合（staging環境など）
            with open(file, "r", encoding="utf-8") as f:
                df = read_csv(f, nrows=1)
        else:
            # UploadedFileなどのBytesIO系（dev環境）
            file.seek(0)
            df = read_csv(file, nrows=1)

        cols = list(df.columns)[:5]
        logger.info(f"📊 アップロードCSVの先頭列: {cols}")

        for name, expected in signatures.items():
            if cols[: len(expected)] == expected:
                logger.info(f"✅ 種別判定成功: {name}")
                return name

        logger.warning("⚠️ 種別が一致しませんでした → 不明な形式")
        return "不明な形式"

    except Exception as e:
        logger.error(f"❌ 読み込みエラー: {e}", exc_info=True)
        return f"読み込みエラー: {e}"
