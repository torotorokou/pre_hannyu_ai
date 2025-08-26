import pandas as pd
import numpy as np
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import (
    get_template_config,
    get_expected_dtypes_by_template,
    get_required_columns_definition,
)
from logic.manage.utils.load_template import load_master_and_template


def processor_func(dfs: dict) -> pd.DataFrame:
    """
    データ処理を行う関数

    Args:
        dfs (dict): 各種データフレームを含む辞書
            - shipping: 出荷データ
            - yard: ヤードデータ
            - receive: 受入データ

    Returns:
        pd.DataFrame: 処理結果のデータフレーム
    """
    # 出荷一覧のdfを取得
    shipping_df = make_csv(dfs)

    # マスターCSVの読込
    master_csv = process1(shipping_df)

    # df_shippingからデータを取得
    df_after = process2(master_csv, shipping_df)

    # master_csvにマージ
    master_csv = make_merge_df(master_csv, df_after)

    # 単価計算
    master_csv = process3(master_csv)

    # カラムの最終調整
    master_csv = process4(master_csv)

    # ダウンロードボタンの表示
    # process5(master_csv)
    return master_csv


def make_csv(dfs):
    # 各データフレームを取得
    shipping_df = dfs.get("shipping", pd.DataFrame())

    # 必要なカラムのリストを取得
    cul_list = get_required_shipping_columns()

    # 必要なカラムが存在するか確認
    missing_columns = [col for col in cul_list if col not in shipping_df.columns]
    if missing_columns:
        raise ValueError(
            f"以下の必要なカラムが出荷データに存在しません: {missing_columns}"
        )

    # 必要なカラムのみに絞る
    shipping_df = shipping_df[cul_list]

    # 空白除去
    shipping_df = strip_whitespace(shipping_df)

    # テンプレートに基づく型定義を取得
    expected_dtypes = get_expected_dtypes_by_template("balance_management_table")
    shipping_dtypes = expected_dtypes.get("shipping", {})

    # データ型を適用
    if shipping_dtypes:
        shipping_df = enforce_dtypes(shipping_df, shipping_dtypes)

    return shipping_df


def process1(df: pd.DataFrame) -> pd.DataFrame:
    # マスターCSVの読込
    config = get_template_config()["balance_management_table"]
    master_path = config["master_csv_path"]["balance_management_table"]
    master_csv = load_master_and_template(master_path)

    # テンプレートに基づく型定義を取得
    expected_dtypes = get_expected_dtypes_by_template("balance_management_table")
    master_dtypes = expected_dtypes.get("master", {})

    # 空白除去
    master_csv = strip_whitespace(master_csv)

    # データ型を適用
    if master_dtypes:
        master_csv = enforce_dtypes(master_csv, master_dtypes)
    else:
        # マスター定義がない場合のデフォルト処理
        for col in master_csv.columns:
            if col in ["大項目", "中項目"]:
                master_csv[col] = (
                    master_csv[col].astype(str).where(master_csv[col].notna(), "")
                )
            else:
                master_csv[col] = (
                    master_csv[col].astype(str).where(master_csv[col].notna(), np.nan)
                )

    return master_csv


import pandas as pd
from typing import List


def process2(master_df: pd.DataFrame, shipping_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_match: List[str] = [
        col for col in master_df.columns if col not in ["大項目", "中項目"]
    ]

    # --- 結果を保持するリスト
    result_list = []

    for _, row in master_df.iterrows():
        condition = shipping_df[columns_to_match[0]] == row[columns_to_match[0]]
        for col in columns_to_match[1:]:
            val = row[col]
            if pd.notna(val) and str(val) not in ["0", "nan", "NaN"]:
                condition &= shipping_df[col] == val

        filtered = shipping_df[condition].copy()

        # 💡 条件を保存（row から大項目・中項目なども含めて持ってくる）
        for col in master_df.columns:
            filtered[col] = row[col]

        result_list.append(filtered)

    # --- 結合
    final_result = pd.concat(result_list, ignore_index=True)

    # --- グループキー（大項目・中項目など）
    group_columns = master_df.columns.tolist()

    # --- 集計処理
    sum_df = final_result.groupby(group_columns)[["正味重量", "金額"]].sum()

    # ✅ 件数（行数）を追加
    count_series = final_result.groupby(group_columns).size()
    sum_df["台数"] = count_series

    # リセットインデックスして整形
    final_result = sum_df.reset_index()

    return final_result


def get_required_shipping_columns() -> list:
    """
    balance_management_tableに必要な出荷データのカラムを取得する

    Returns:
        list: 必要なカラムのリスト
    """
    # required_columns_definition.yamlから必要なカラムを取得
    required_cols = get_required_columns_definition("balance_management_table")

    # shippingのカラムを取得（存在しない場合は空のリストを返す）
    shipping_cols = required_cols.get("shipping", [])

    return shipping_cols


def make_merge_df(master_csv, df_after):
    # カラム名を変更
    df_after = df_after.rename(columns={"正味重量": "合計正味重量", "金額": "合計金額"})

    # マージ
    master_csv = master_csv.merge(df_after, on=["大項目", "中項目"], how="left")

    # 必要なカラムだけをピックアップ
    selected_columns = ["大項目", "中項目", "合計正味重量", "合計金額", "台数"]
    master_csv = master_csv[selected_columns]
    return master_csv


def process3(master_csv):
    # 合計正味重量が0の行は単価を0に、それ以外は通常の割り算
    master_csv["単価"] = master_csv.apply(
        lambda row: (
            row["合計金額"] / row["合計正味重量"] if row["合計正味重量"] != 0 else 0
        ),
        axis=1,
    )
    return master_csv


def process4(master_csv: pd.DataFrame) -> pd.DataFrame:
    selected_columns = ["大項目", "中項目", "合計正味重量", "合計金額", "単価", "台数"]

    # NaNを補完してから型変換（int列は0、float列は0.0など）
    master_csv["合計正味重量"] = master_csv["合計正味重量"].fillna(0)
    master_csv["合計金額"] = master_csv["合計金額"].fillna(0.0)
    master_csv["単価"] = master_csv["単価"].fillna(0.0)
    master_csv["台数"] = master_csv["台数"].fillna(0)

    # 型変換
    master_csv = master_csv.astype(
        {
            "大項目": str,
            "中項目": str,
            "合計正味重量": int,
            "合計金額": float,
            "単価": float,
            "台数": int,
        }
    )

    # 並び替え
    master_csv = master_csv[selected_columns]

    return master_csv


# from components.custom_button import centered_download_button
# from io import BytesIO

# def process5(master_csv: pd.DataFrame):
#     excel_bytes = convert_df_to_excel_bytes(master_csv)

#     centered_download_button(
#         label="📥 Excelファイルをダウンロード",
#         data=excel_bytes,
#         file_name="管理表.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )
