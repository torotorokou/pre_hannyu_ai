import pandas as pd
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.summary_tools import summary_apply
from utils.config_loader import get_unit_price_table_csv
from logic.manage.utils.multiply_tools import multiply_columns
from utils.config_loader import get_unit_price_table_csv
from logic.manage.utils.multiply_tools import multiply_columns


# 業者別の処分費
def calculate_total_disposal_cost(
    df_yard: pd.DataFrame,
    df_shipping: pd.DataFrame,
) -> int:
    """
    出荷データ・ヤードデータをもとに処分費の総額を計算する。

    各種処分費を以下のように集計し、合算する：
    - 出荷データに基づく処分費（業者別）
    - 出荷データに基づく金庫対象処分費
    - ヤードデータに基づく金庫対象処分費

    Parameters
    ----------
    df_shipping : pd.DataFrame
        出荷一覧データ（業者CD、金額、正味重量などを含む）

    df_yard : pd.DataFrame
        ヤード出荷データ（種類名、品名、正味重量などを含む）

    Returns
    -------
    int
        集計された処分費の合計金額
    """

    # --- ① 業者別処分費 ---
    cost_by_vendor = int(calculate_disposal_costs(df_shipping)["値"].sum())

    # --- ② 金庫（出荷データ） ---
    cost_safe_shipping = int(calculate_safe_disposal_costs(df_shipping)["値"].sum())

    # --- ③ 金庫（ヤードデータ） ---
    cost_safe_yard = int(calculate_yard_disposal_costs(df_yard)["値"].sum())

    # --- ④ 総合計 ---
    total_cost = cost_by_vendor + cost_safe_shipping + cost_safe_yard

    return total_cost


def calculate_disposal_costs(df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    処分費（shobun_cost）の業者別金額をマスターCSVに反映させる。

    Parameters
    ----------
    df_shipping : pd.DataFrame
        出荷データ（業者CD, 金額を含む）

    Returns
    -------
    pd.DataFrame
        金額が反映された処分費用マスターDataFrame
    """

    # --- マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["shobun_cost"]
    master_csv_cost = load_master_and_template(master_path)

    # --- 業者CDの型を統一（str） ---
    df_shipping["業者CD"] = df_shipping["業者CD"].astype(str)
    master_csv_cost["業者CD"] = master_csv_cost["業者CD"].astype(str)

    # --- 金額集計と反映（業者CD単位） ---
    key_cols = ["業者CD"]
    source_col = "金額"
    master_csv_cost = summary_apply(master_csv_cost, df_shipping, key_cols, source_col)

    return master_csv_cost


#  処分費_金庫
def calculate_safe_disposal_costs(df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    処分費（金庫対象）を出荷データと単価マスタから計算し、マスターCSVに反映する。

    処理概要：
    - 出荷データから「業者名×品名」ごとの正味重量を集計し、
    - 単価テーブルから設定単価を取得、
    - 両者を掛け合わせて「値（＝コスト）」列を算出する。

    Parameters
    ----------
    df_shipping : pd.DataFrame
        出荷データ（業者名、品名、正味重量を含む）

    Returns
    -------
    pd.DataFrame
        「設定単価」「正味重量」「値（単価×重量）」を含むマスターDataFrame
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["syobun_cost_kinko"]
    master_df = load_master_and_template(master_path)

    # --- ② 出荷データから「正味重量」を集計してマスターに反映 ---
    key_cols = ["業者名", "品名"]
    master_with_weight = summary_apply(
        master_df,
        df_shipping,
        key_cols=key_cols,
        source_col="正味重量",
        target_col="正味重量",  # 上書きする形で反映
    )

    # --- ③ 単価テーブルから「設定単価」を取得してマスターに反映 ---
    unit_price_df = get_unit_price_table_csv()
    master_with_price = summary_apply(
        master_with_weight,
        unit_price_df,
        key_cols=key_cols,
        source_col="設定単価",
        target_col="設定単価",
    )

    # --- ④ 正味重量 × 設定単価 = 処分費（値） を計算 ---
    master_csv_kinko = multiply_columns(
        master_with_price, col1="設定単価", col2="正味重量", result_col="値"
    )

    return master_csv_kinko


def calculate_yard_disposal_costs(yard_df: pd.DataFrame) -> pd.DataFrame:
    """
    ヤードデータから処分費（金庫分）を計算し、マスターに反映する。

    処理ステップ:
    1. 「種類名 × 品名」ごとに yard_df の正味重量を集計し、マスターに適用
    2. 単価表から設定単価を適用
    3. 単価 × 正味重量 を計算し、「値」列に出力

    Parameters
    ----------
    yard_df : pd.DataFrame
        ヤード出荷データ（種類名、品名、正味重量を含む）

    Returns
    -------
    pd.DataFrame
        「正味重量」「設定単価」「値（コスト）」を反映したマスターDataFrame
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["syobun_cost_kinko_yard"]
    master_df = load_master_and_template(master_path)

    # --- ② 正味重量の集計適用（種類名 × 品名） ---
    key_cols = ["種類名", "品名"]
    master_with_weight = summary_apply(
        master_df,
        yard_df,
        key_cols=key_cols,
        source_col="正味重量",
        target_col="正味重量",
    )

    # --- ③ 単価の適用 ---
    unit_price_df = get_unit_price_table_csv()
    master_with_price = summary_apply(
        master_with_weight,
        unit_price_df,
        key_cols=key_cols,
        source_col="設定単価",
        target_col="設定単価",
    )

    # --- ④ 処分費の計算（単価 × 正味重量） ---
    master_csv_kinko_yard = multiply_columns(
        master_with_price, col1="設定単価", col2="正味重量", result_col="値"
    )

    return master_csv_kinko_yard
