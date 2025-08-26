import pandas as pd
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.summary_tools import summary_apply


def calculate_honest_sales_by_unit(df_receive: pd.DataFrame) -> tuple[int, int]:
    """
    「オネストkg」「オネストm3」の売上金額を受入データから計算する。

    処理の流れ：
    1. 「伝票区分名 × 単位名」で金額を集計し、オネストm3の値を算出
    2. 「伝票区分名」で金額を集計し、オネストkgの値を取得
    3. オネストkg から m3 を引いた差分を「kg の純粋売上」として返す

    Parameters
    ----------
    df_receive : pd.DataFrame
        受入データ。伝票区分名・単位名・金額列を含む。

    Returns
    -------
    tuple[int, int]
        (オネストkg の総額, オネストkg - オネストm3 の差額)
    """

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["uriage"]
    master_df = load_master_and_template(master_path)

    # --- ② 「伝票区分名 × 単位名」で金額を集計（m3） ---
    summary_m3 = summary_apply(
        master_df,
        df_receive,
        key_cols=["伝票区分名", "単位名"],
        source_col="金額",
        target_col="金額",
    )

    # --- ③ 「伝票区分名」のみで金額を集計（kg） ---
    summary_kg = summary_apply(
        master_df,
        df_receive,
        key_cols=["伝票区分名"],
        source_col="金額",
        target_col="金額",
    )

    # --- ④ m3 売上金額を取得 ---
    honest_row_m3 = summary_m3[summary_m3["項目"] == "オネストm3"]
    honest_m3_value = honest_row_m3["金額"].values[0] if not honest_row_m3.empty else 0

    # --- ⑤ kg 売上金額を取得 ---
    honest_row_kg = summary_kg[summary_kg["項目"] == "オネストkg"]
    honest_kg_value = honest_row_kg["金額"].values[0] if not honest_row_kg.empty else 0

    # kg - m3 の差分を計算（kg純粋）
    honest_kg_total = honest_kg_value - honest_m3_value

    return honest_kg_total, honest_m3_value
