import streamlit as st
import pandas as pd
from datetime import date


def render_download_button(start_date: date, end_date: date) -> None:
    """
    搬入量予測結果のCSVをダウンロードできるボタンを表示する。

    Parameters:
        start_date (date): 予測対象の開始日
        end_date (date): 予測対象の終了日

    Returns:
        None
    """
    df = st.session_state["df_import_prediction"]
    df_csv = _convert_to_csv(df)
    _inject_download_button_style()

    filename = _generate_filename(start_date, end_date)
    st.download_button(
        label="📥 CSVダウンロード",
        data=df_csv,
        file_name=filename,
        mime="text/csv; charset=shift_jis",
    )


def _convert_to_csv(df: pd.DataFrame) -> str:
    """
    ダウンロード対象のDataFrameを加工してCSV文字列に変換する。

    Parameters:
        df (pd.DataFrame): 搬入量予測結果

    Returns:
        str: shift_jis形式のCSV文字列
    """
    df_download = df[
        [
            "日付",
            "曜日",
            "予測値",
            "補正後予測",
            "下限95CI",
            "上限95CI",
            "判定ラベル",
            "未満確率",
        ]
    ].copy()
    return df_download.to_csv(index=False, encoding="shift_jis")


def _generate_filename(start_date: date, end_date: date) -> str:
    """
    予測対象期間をもとにCSVファイル名を生成する。

    Returns:
        str: ファイル名（例: 20250501_20250506_予測結果.csv）
    """
    return f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_予測結果.csv"


def _inject_download_button_style() -> None:
    """
    StreamlitのダウンロードボタンにCSSスタイルを適用する。
    """
    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(to right, #fddb3a, #f6b93b);
            color: black;
            border: none;
            font-weight: bold;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
