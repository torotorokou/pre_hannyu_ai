import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.excel_tools import add_label_rows_and_restore_sum
from logic.manage.utils.summary_tools import (
    summarize_value_by_cell_with_label,
)

from logic.manage.processors.factory_report.summary import summary_apply_by_sheet


def process_yard(df_yard: pd.DataFrame, df_shipping: pd.DataFrame) -> pd.DataFrame:
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["factory_report"]
    master_path = config["master_csv_path"]["yard"]
    master_csv = load_master_and_template(master_path)

    # --- ② ヤードの値集計処理（df_yard + df_shippingを使用） ---
    updated_master_csv = apply_yard_summary(master_csv, df_yard, df_shipping)
    updated_master_csv = negate_template_values(updated_master_csv)

    # # --- ③ 品目名単位でマージし、合計を計算 ---
    updated_with_sum = summarize_value_by_cell_with_label(
        updated_master_csv, cell_col="品目名", label_col="セル"
    )

    # # # --- ④ 合計行などを追加集計 ---
    # target_keys = ["品目名"]
    # target_values = ["合計_ヤード"]
    # updated_with_sum2 = write_sum_to_target_cell(
    #     updated_with_sum, target_keys, target_values
    # )

    # フォーマット修正
    final_df = format_table(updated_with_sum)

    logger.info("✅ 出荷ヤードの帳票生成が完了しました。")
    return final_df


def apply_yard_summary(master_csv, df_yard, df_shipping):
    df_map = {"ヤード": df_yard, "出荷": df_shipping}

    sheet_key_pairs = [
        ("ヤード", ["種類名"]),
        ("ヤード", ["種類名", "品名"]),
        ("出荷", ["業者名", "品名"]),
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
    format_df = master_csv[["品目名", "セル", "値", "セルロック", "順番"]].copy()

    # 列の置換
    format_df.rename(columns={"品目名": "大項目"}, inplace=True)

    # カテゴリ列を追加
    format_df["カテゴリ"] = "ヤード"

    return format_df


def negate_template_values(master_csv: pd.DataFrame) -> pd.DataFrame:
    # --- 条件フィルター：対象は「品目名=その他」かつ「種類名=処分費」
    condition = (master_csv["品目名"] == "その他") & (master_csv["種類名"] == "処分費")

    # --- 品名ごとにマイナス処理
    mask_sentubetsu = condition & (master_csv["品名"] == "選別")
    mask_kinko = condition & (master_csv["品名"] == "金庫")
    mask_gc = condition & (master_csv["品名"] == "GC軽鉄・ｽﾁｰﾙ類")

    # --- 対象の値をマイナスに（符号反転）
    master_csv.loc[mask_sentubetsu, "値"] = -master_csv.loc[mask_sentubetsu, "値"]
    master_csv.loc[mask_kinko, "値"] = -master_csv.loc[mask_kinko, "値"]
    master_csv.loc[mask_gc, "値"] = -master_csv.loc[mask_gc, "値"]

    return master_csv
