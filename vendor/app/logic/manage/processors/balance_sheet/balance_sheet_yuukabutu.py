import pandas as pd
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.summary_tools import summary_apply
from utils.config_loader import get_unit_price_table_csv
from logic.manage.utils.multiply_tools import multiply_columns
from utils.config_loader import get_unit_price_table_csv
from logic.manage.utils.multiply_tools import multiply_columns


def calculate_total_valuable_material_cost(
    df_yard: pd.DataFrame,
    df_shipping: pd.DataFrame,
) -> int:
    """
    出荷データとヤードデータをもとに、有価物の合計金額を算出する。

    処理内容：
    - 出荷データから業者別に有価金額を集計
    - ヤードデータから品目別に有価金額を集計
    - それぞれの「値」を合算し、総額を返す

    Parameters
    ----------
    df_shipping : pd.DataFrame
        出荷一覧（業者名、金額を含む）

    df_yard : pd.DataFrame
        ヤード出荷データ（品名、数量を含む）

    Returns
    -------
    int
        出荷分 + ヤード分の有価物合計金額
    """

    # --- 有価_出荷 ---
    shipping_summary_df = aggregate_valuable_material_by_vendor(df_shipping)
    sum_shipping = shipping_summary_df["値"].sum()

    # --- 有価_ヤード ---
    yard_summary_df = calculate_valuable_material_cost_by_item(df_yard)
    sum_yard = yard_summary_df["値"].sum()

    # --- 合計 ---
    total_value = int(sum_shipping + sum_yard)
    return total_value


def aggregate_valuable_material_by_vendor(shipping_df: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データをもとに、有価物（業者別）の金額を集計してマスターに反映する。

    Parameters
    ----------
    shipping_df : pd.DataFrame
        出荷データ（業者名、金額を含む）

    Returns
    -------
    pd.DataFrame
        「業者名」ごとの金額を「値」列に反映したマスターDataFrame
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["yuka_shipping"]
    master_df = load_master_and_template(master_path)

    # --- ② 業者別に金額を集計し、「値」列に反映 ---
    key_cols = ["業者名"]
    aggregated_df = summary_apply(
        master_df,
        shipping_df,
        key_cols=key_cols,
        source_col="金額",
        target_col="値",
    )

    return aggregated_df


### 有価ヤード


def calculate_valuable_material_cost_by_item(df_yard: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データと単価表から、有価物（品名別）の金額を算出する。

    処理ステップ:
    1. 出荷データから品名ごとの数量をマスターに集計適用
    2. 単価テーブルから品名ごとの設定単価を適用
    3. 数量 × 単価で金額（値）を計算し、大項目列に品名を転記

    Parameters
    ----------
    df_yard : pd.DataFrame
        ヤードデータ（品名、数量を含む）

    Returns
    -------
    pd.DataFrame
        「品名（→大項目）」「数量」「設定単価」「値（金額）」を含む集計済みデータ
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["yuka_yard"]
    master_df = load_master_and_template(master_path)

    # --- ② 品名ごとの数量を集計して反映 ---
    master_with_quantity = summary_apply(
        master_df,
        df_yard,
        key_cols=["品名"],
        source_col="数量",
        target_col="数量",
    )

    # --- ③ 単価テーブルから設定単価を反映（有価物のみ） ---
    unit_price_df = get_unit_price_table_csv()
    unit_price_df = unit_price_df[unit_price_df["必要項目"] == "有価物"]

    master_with_unit_price = summary_apply(
        master_with_quantity,
        unit_price_df,
        key_cols=["品名"],
        source_col="設定単価",
        target_col="設定単価",
    )

    # --- ④ 数量 × 単価 → 値 を計算 ---
    result_df = multiply_columns(
        master_with_unit_price, col1="設定単価", col2="数量", result_col="値"
    )

    # --- ⑤ 品名列 → 大項目列へコピー（用途別での出力整形） ---
    result_df = result_df.rename(columns={"品名": "大項目"})

    return result_df
