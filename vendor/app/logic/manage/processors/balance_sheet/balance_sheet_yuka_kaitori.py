import pandas as pd
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.summary_tools import summary_apply
from utils.config_loader import get_unit_price_table_csv
from logic.manage.utils.multiply_tools import multiply_columns


def calculate_purchase_value_of_valuable_items(receive_df: pd.DataFrame) -> int:
    """
    有価物（買取）の買取金額合計を算出する。

    処理ステップ：
    1. 「品名 × 単価」ごとの数量を受入データから集計
    2. 単価マスタ（有価買取）を反映
    3. 単価 × 数量 → 値（金額）を算出
    4. 「支払」の伝票区分行に、受入データからの実金額を反映
    5. 全体の合計金額を返す

    Parameters
    ----------
    receive_df : pd.DataFrame
        受入データ（品名、単価、数量、金額、伝票区分名などを含む）

    Returns
    -------
    int
        有価物買取の総額（金額の合計）
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["uriage_yuka_kaitori"]
    master_df = load_master_and_template(master_path)

    # --- ② 「品名 × 伝票区分」で数量を集計して反映 ---
    master_with_quantity = summary_apply(
        master_df,
        receive_df,
        key_cols=["品名", "伝票区分名"],
        source_col="数量",
        target_col="数量",
    )

    # --- ③ 単価マスタ（有価買取）の適用 ---
    unit_price_df = get_unit_price_table_csv()
    unit_price_df = unit_price_df[unit_price_df["必要項目"] == "有価買取"]

    master_with_prices = summary_apply(
        master_with_quantity,
        unit_price_df,
        key_cols=["品名"],
        source_col="設定単価",
        target_col="設定単価",
    )

    # --- ④ 単価 × 数量 を計算して金額（値）を算出 ---
    result_df = multiply_columns(
        master_with_prices, col1="設定単価", col2="数量", result_col="値"
    )

    # # --- ⑤ 「支払」行に実際の金額を反映 ---
    # payment_summary = summary_apply(
    #     master_with_prices,
    #     receive_df,
    #     key_cols=["伝票区分名"],
    #     source_col="金額",
    #     target_col="値",
    # )

    # if not payment_summary[payment_summary["伝票区分名"] == "支払"].empty:
    #     payment_value = payment_summary.loc[
    #         payment_summary["伝票区分名"] == "支払", "値"
    #     ].values[0]
    #     result_df.loc[result_df["伝票区分名"] == "支払", "値"] = payment_value

    # --- ⑥ 合計金額を返す ---
    total = int(result_df["値"].sum())
    return total
