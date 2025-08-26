import streamlit as st
from io import BytesIO


def centered_button(label: str, key: str = None, disabled: bool = False) -> bool:
    """
    中央寄せでスタイル付きのボタンを表示する関数。

    Parameters:
        label (str): ボタンに表示するテキスト
        key (str): ボタンのキー（任意）
        disabled (bool): ボタンを無効化するかどうか（デフォルトは有効）

    Returns:
        bool: ボタンが押されたかどうか
    """

    # スタイルを一度だけ適用
    if "_custom_centered_button" not in st.session_state:
        st.markdown(
            """
        <style>
        div.stButton > button {
            background-color: #fbbc04;
            color: #111;
            font-weight: 600;
            font-size: 16px;
            padding: 0.6rem 1.5rem;
            border-radius: 6px;
            border: none;
            min-width: 160px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            background-color: #f9a825;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.session_state["_custom_centered_button"] = True

    # 中央寄せレイアウト
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        return st.button(label, key=key, disabled=disabled)


def centered_download_button(
    label: str,
    data: BytesIO,
    file_name: str,
    mime: str = "application/octet-stream",
    key: str = None,
) -> bool:
    """
    中央寄せでスタイル付きのダウンロードボタンを表示する関数。

    Parameters:
        label (str): ボタンに表示するテキスト
        data (BytesIO): ダウンロードするデータ（バイナリ）
        file_name (str): ダウンロードファイル名
        mime (str): MIMEタイプ（デフォルトはバイナリ）
        key (str): ボタンのキー（任意）

    Returns:
        bool: ボタンが押されたかどうか
    """

    # スタイルを一度だけ適用
    if "_custom_centered_download_button" not in st.session_state:
        st.markdown(
            """
        <style>
        div.stDownloadButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
            font-size: 16px;
            padding: 0.6rem 1.5rem;
            border-radius: 6px;
            border: none;
            min-width: 180px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }
        div.stDownloadButton > button:hover {
            background-color: #43a047;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.session_state["_custom_centered_download_button"] = True

    # 中央寄せレイアウト
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        return st.download_button(
            label=label, data=data, file_name=file_name, mime=mime, key=key
        )
