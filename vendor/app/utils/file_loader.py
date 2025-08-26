import pandas as pd
import streamlit as st


@st.cache_data
def read_csv(file, nrows=None):
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", nrows=nrows)


def load_uploaded_csv_files(uploaded_files: dict) -> dict:
    """
    アップロードされた複数のCSVファイルを読み込み、辞書形式で返す。

    Parameters:
        uploaded_files (dict): {"file_key": UploadedFile オブジェクト} の辞書

    Returns:
        dict: {"file_key": DataFrame} の辞書
    """
    dfs = {}
    for file_key, uploaded_file in uploaded_files.items():
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dfs[file_key] = df
            except Exception as e:
                st.error(f"❌ {file_key} の読み込みでエラーが発生しました: {e}")
    return dfs
