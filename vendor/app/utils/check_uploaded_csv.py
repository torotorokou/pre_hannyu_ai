from app_pages.manage.view import render_upload_header
import tempfile
import streamlit as st


from utils.config_loader import (
    get_csv_label_map,
)


import streamlit as st
import tempfile


def render_csv_upload_section(upload_key: str):
    """
    任意のCSVファイルアップロードUIを表示し、テンポラリ保存＋セッションに登録する。

    Args:
        upload_key (str): ファイル種別キー（例: "shipping", "yard", "receive"）
    """
    # ラベル取得マップ（必要なら外部に切り出してもOK）
    csv_label_map = get_csv_label_map()
    label = csv_label_map.get(upload_key, f"{upload_key}ファイル")

    st.markdown(f"### 📂 {label}のアップロード")

    render_upload_header(label)
    uploaded_file = st.file_uploader(
        label, type="csv", key=upload_key, label_visibility="collapsed"
    )

    session_key = f"uploaded_{upload_key}"
    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            st.session_state[session_key] = tmp_path
        except Exception as e:
            st.error(f"{label}の保存に失敗しました: {e}")
            st.session_state[session_key] = None
    else:
        st.session_state[session_key] = None


def check_single_file_uploaded(
    uploaded_file: str | None, required_key: str
) -> tuple[bool, str | None]:
    """
    単一ファイルがアップロードされているかをチェックする

    Args:
        uploaded_file (str | None): 一時ファイルパスまたは None
        required_key (str): 対象のファイルキー名（例: 'shipping'）

    Returns:
        is_uploaded (bool): ファイルがアップロードされているか
        missing_key (str | None): 未アップロードの場合はキー名、それ以外は None
    """
    is_uploaded = uploaded_file is not None
    missing_key = None if is_uploaded else required_key
    return is_uploaded, missing_key
