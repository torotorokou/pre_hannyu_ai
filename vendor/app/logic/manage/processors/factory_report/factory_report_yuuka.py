import pandas as pd
import streamlit as st
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.excel_tools import add_label_rows_and_restore_sum
from logic.manage.processors.factory_report.summary import summary_apply_by_sheet
from logic.manage.utils.summary_tools import (
    summarize_value_by_cell_with_label,
)
from utils.debug_tools import DevTools


def process_yuuka(df_yard: pd.DataFrame, df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷有価パートの帳票生成処理。

    Parameters:
        df_yard : pd.DataFrame
            ヤード一覧のDataFrame
        df_shipping : pd.DataFrame
            出荷一覧のDataFrame

    Returns:
        pd.DataFrame
            整形された出荷有価帳票のDataFrame
    """
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["factory_report"]
    master_path = config["master_csv_path"]["yuuka"]
    master_csv = load_master_and_template(master_path)

    # --- ② 有価の値集計処理（df_yard + df_shippingを使用） ---
    updated_master_csv = apply_yuuka_summary(master_csv, df_yard, df_shipping)

    # --- ③ 品目単位で有価名をマージし、合計を計算 ---
    updated_with_sum = summarize_value_by_cell_with_label(
        updated_master_csv, cell_col="有価名", label_col="セル"
    )

    # フォーマット修正
    final_df = format_table(updated_with_sum)

    logger.info("✅ 出荷有価の帳票生成が完了しました。")
    return final_df


def apply_yuuka_summary(master_csv, df_yard, df_shipping):
    devtools = DevTools()
    logger = app_logger()
    df_map = {"ヤード": df_yard, "出荷": df_shipping}

    sheet_key_pairs = [
        ("ヤード", ["品名"]),
        ("出荷", ["品名"]),
        ("出荷", ["業者名", "品名"]),
        ("出荷", ["現場名", "業者名", "品名"]),
    ]

    master_csv_updated = master_csv.copy()

    for sheet_name, key_cols in sheet_key_pairs:
        data_df = df_map[sheet_name]

        master_csv_updated = summary_apply_by_sheet(
            master_csv=master_csv_updated,
            data_df=data_df,
            sheet_name=sheet_name,
            key_cols=key_cols,
        )

    return master_csv_updated


def format_table(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    出荷処分のマスターCSVから必要な列を整形し、カテゴリを付与する。

    Parameters:
        master_csv : pd.DataFrame
            出荷処分の帳票CSV（"業者名", "セル", "値" を含む）

    Returns:
        pd.DataFrame : 整形後の出荷処分データ
    """
    # 必要列を抽出
    format_df = master_csv[["有価名", "セル", "値", "セルロック", "順番"]].copy()

    # 置換
    format_df.rename(columns={"有価名": "大項目"}, inplace=True)

    # カテゴリ列を追加
    format_df["カテゴリ"] = "有価"

    return format_df
