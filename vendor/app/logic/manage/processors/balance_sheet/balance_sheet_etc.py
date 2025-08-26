import pandas as pd
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from utils.date_tools import to_reiwa_format
from logic.manage.utils.summary_tools import set_value_fast_safe


def calculate_misc_summary_rows(
    master_csv: pd.DataFrame, first_invoice_date: pd.Timestamp
) -> pd.DataFrame:
    """
    売上・仕入・損益の補足行（「月日」「売上計」「仕入計」「損益」）を計算し、マスターに追加する。
    """

    def safe_int(val, default=0):
        try:
            if pd.isna(val):
                return default
            return int(val)
        except Exception:
            return default

    # --- ① etcマスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["etc"]
    etc_df = load_master_and_template(master_path)

    # --- ② 日付を和暦で記載（最初の受入日） ---
    reiwa_date = to_reiwa_format(first_invoice_date)
    etc_df = set_value_fast_safe(
        df=etc_df,
        match_columns=["大項目"],
        match_values=["月日"],
        value=reiwa_date,
        value_col="値",
    )

    # --- ③ 売上合計（オネストkg + オネストm3 - 有価買取） ---
    honest_kg = safe_int(
        master_csv.loc[master_csv["大項目"] == "オネストkg", "値"].values[0]
    )
    honest_m3 = safe_int(
        master_csv.loc[master_csv["大項目"] == "オネストm3", "値"].values[0]
    )
    yuka_kaitori = safe_int(
        master_csv.loc[master_csv["大項目"] == "有価買取", "値"].values[0]
    )
    sales_total = honest_kg + honest_m3 - yuka_kaitori
    etc_df = set_value_fast_safe(etc_df, ["大項目"], ["売上計"], sales_total, "値")

    # --- ④ 仕入合計（処分費 - 有価物） ---
    shobun_cost = safe_int(
        master_csv.loc[master_csv["大項目"] == "処分費", "値"].values[0]
    )
    yuka_cost = safe_int(
        master_csv.loc[master_csv["大項目"] == "有価物", "値"].values[0]
    )
    cost_total = shobun_cost - yuka_cost
    etc_df = set_value_fast_safe(etc_df, ["大項目"], ["仕入計"], cost_total, "値")

    # --- ⑤ 損益（売上計 - 仕入計） ---
    profit_total = sales_total - cost_total
    etc_df = set_value_fast_safe(etc_df, ["大項目"], ["損益"], profit_total, "値")

    # --- ⑥ summary_df に etc_df を結合して返す ---
    result_df = pd.concat([master_csv, etc_df], axis=0, ignore_index=True)

    return result_df
