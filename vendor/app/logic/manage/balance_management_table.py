import pandas as pd
from utils.config_loader import get_template_config, get_required_columns_definition


def get_condition_columns(master_df: pd.DataFrame) -> list:
    """
    マスターデータから条件として使用するカラムを抽出する

    Args:
        master_df (pd.DataFrame): マスターデータ

    Returns:
        list: 条件として使用するカラムのリスト
    """
    # '大項目'と'中項目'以外のカラムを取得
    condition_columns = [
        col for col in master_df.columns if col not in ["大項目", "中項目"]
    ]
    return condition_columns


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


def process(shipping_df: pd.DataFrame, master_csv: pd.DataFrame = None) -> pd.DataFrame:
    """
    shipping_dfから balance_management_table.csv の条件に基づいて集計を行う

    Args:
        shipping_df (pd.DataFrame): 出荷データ
        master_csv (pd.DataFrame, optional): マスターデータ。Noneの場合はデフォルトのCSVを読み込む

    Returns:
        pd.DataFrame: 集計結果
    """
    # 必要なカラムの取得
    required_columns = get_required_shipping_columns()

    # 必要なカラムが存在するか確認
    missing_columns = [
        col for col in required_columns if col not in shipping_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"以下の必要なカラムが出荷データに存在しません: {missing_columns}"
        )

    # マスターデータの読み込み
    if master_csv is None:
        config = get_template_config()
        master_path = config["balance_management_table"]["master_csv_path"][
            "balance_management_table"
        ]
        master_df = pd.read_csv(master_path)
    else:
        master_df = master_csv

    # 条件として使用するカラムを取得
    columns_to_check = get_condition_columns(master_df)

    # 結果を格納するための空のDataFrame
    results = []

    # 各条件ごとに集計
    for _, row in master_df.iterrows():
        # 条件に基づくフィルタリング
        mask = pd.Series(True, index=shipping_df.index)

        # 各カラムの条件を適用
        for col in columns_to_check:
            if pd.notna(row[col]) and col in shipping_df.columns:
                mask &= shipping_df[col].astype(str) == str(row[col])

        # フィルタリングされたデータを取得
        filtered_df = shipping_df[mask]

        if not filtered_df.empty:
            # 集計（例：数量と金額の合計）
            summary = {
                "大項目": row["大項目"],
                "中項目": row["中項目"],
                "件数": len(filtered_df),
                "数量": (
                    filtered_df["数量"].sum() if "数量" in filtered_df.columns else 0
                ),
                "金額": (
                    filtered_df["金額"].sum() if "金額" in filtered_df.columns else 0
                ),
            }
            results.append(summary)

    # 結果をDataFrameに変換
    result_df = pd.DataFrame(results)

    # 大項目と中項目でソート
    if not result_df.empty:
        result_df = result_df.sort_values(["大項目", "中項目"])

    return result_df
