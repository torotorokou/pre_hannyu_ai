import streamlit as st
from logic.detect_csv import detect_csv_type
from components.ui_message import show_warning_bubble
from utils.config_loader import get_csv_label_map
from utils.logger import app_logger  # ロガーのインポート


def handle_uploaded_files(required_keys):
    """
    アップロードされたCSVファイルの整合性を確認し、正しいもののみを返す関数。

    Parameters:
    - required_keys: アップロードが必要なファイルのキー一覧（例：["receive", "shipping"]）
    - csv_label_map: 各キーに対応するCSV種別名（例：{"receive": "受入データ"}）

    Returns:
    - uploaded_files: キーに対応するアップロード済みファイルの辞書（整合性エラーならNoneが入る）
    """
    logger = app_logger()
    uploaded_files = {}
    csv_label_map = get_csv_label_map()

    # アップロード対象のキー（例: "receive", "shipping"）ごとに確認
    for key in required_keys:
        # Streamlitのセッションステートからアップロード済みファイルを取得
        uploaded = st.session_state.get(f"uploaded_{key}")
        # logger.info(f"checking: {key}, uploaded: {bool(uploaded)}")

        if uploaded:
            # このキーに期待されるCSV種別名を取得（例: "shipping" → "出荷データ"）
            expected_name = csv_label_map.get(key, key)

            # アップロードされたCSVファイルの種別名を検出
            detected_name = detect_csv_type(uploaded)
            # logger.info(f"{key} → expected: {expected_name}, detected: {detected_name}")

            # 検出された種別名と期待値が一致しない場合、警告を出して除外
            if detected_name != expected_name:
                logger.warning(
                    f"{key} mismatch: expected {expected_name}, got {detected_name}"
                )
                show_warning_bubble(expected_name, detected_name)  # UIに警告表示
                st.session_state[f"uploaded_{key}"] = None  # アップロードを無効化
                uploaded_files[key] = None
            else:
                # 整合性が取れていれば登録
                uploaded_files[key] = uploaded
        else:
            # ファイルがアップロードされていない場合
            # logger.warning(f"No file uploaded for: {key}")
            uploaded_files[key] = None

    return uploaded_files
