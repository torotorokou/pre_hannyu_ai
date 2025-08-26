import pandas as pd
from utils.logger import app_logger
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_template_config
from logic.manage.processors.balance_sheet.balance_sheet_fact import (
    process_factory_report,
)
from logic.manage.processors.balance_sheet.balance_sheet_syobun import (
    calculate_total_disposal_cost,
)
from logic.manage.processors.balance_sheet.balance_sheet_yuukabutu import (
    calculate_total_valuable_material_cost,
)
from logic.manage.processors.balance_sheet.balance_sheet_inbound_truck_count import (
    inbound_truck_count,
)
from logic.manage.processors.balance_sheet.balacne_sheet_inbound_weight import (
    inbound_weight,
)
from logic.manage.processors.balance_sheet.balance_sheet_honest import (
    calculate_honest_sales_by_unit,
)
from logic.manage.processors.balance_sheet.balance_sheet_yuka_kaitori import (
    calculate_purchase_value_of_valuable_items,
)
from logic.manage.processors.balance_sheet.balance_sheet_etc import (
    calculate_misc_summary_rows,
)


def process(dfs: dict) -> pd.DataFrame:
    """
    複数のCSVから業務帳票データ（balance_sheet）を統合的に処理し、最終的なDataFrameを返す。

    Parameters
    ----------
    dfs : dict
        ファイル名（キー）と読み込まれたDataFrameの辞書

    Returns
    -------
    pd.DataFrame
        処理後のマスター帳票データ（master_csv）
    """
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["balance_sheet"]
    master_path = config["master_csv_path"]["factory"]
    master_csv = load_master_and_template(master_path)

    # --- ② テンプレート設定と対象CSVの読み込み ---
    template_key = "balance_sheet"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]

    # ✅ 必須ファイル＋任意ファイルの両方を取得
    required_keys = template_config.get("required_files", [])
    optional_keys = template_config.get("optional_files", [])
    csv_keys = required_keys + optional_keys

    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- データ読み込み（dfsは事前に構築済とする） ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)

    # --- 各ファイルを取得（Noneの可能性もある） ---
    df_receive = df_dict.get("receive")
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    target_day = pd.to_datetime(df_shipping["伝票日付"].values[0]).date()

    # --- ③ 各処理の適用 ---

    # 仕入処理
    logger.info("▶️ 搬出量データ処理開始")
    master_csv = process_factory_report(dfs, master_csv)

    logger.info("▶️ 処分費データ処理開始")
    master_csv.loc[master_csv["大項目"] == "処分費", "値"] = (
        calculate_total_disposal_cost(df_yard, df_shipping)
    )

    logger.info("▶️ 有価物データ処理開始")
    master_csv.loc[master_csv["大項目"] == "有価物", "値"] = (
        calculate_total_valuable_material_cost(df_yard, df_shipping)
    )

    # 売上ページ:receiveが空なら処理を飛ばす
    if df_receive is not None:
        logger.info("▶️ 搬入台数データ処理開始")
        master_csv.loc[master_csv["大項目"] == "搬入台数", "値"] = inbound_truck_count(
            df_receive
        )

        logger.info("▶️ 搬入量データ処理開始")
        master_csv.loc[master_csv["大項目"] == "搬入量", "値"] = inbound_weight(
            df_receive
        )

        logger.info("▶️ オネストkg / m3 データ処理開始")
        honest_kg, honest_m3 = calculate_honest_sales_by_unit(df_receive)
        master_csv.loc[master_csv["大項目"] == "オネストkg", "値"] = honest_kg
        master_csv.loc[master_csv["大項目"] == "オネストm3", "値"] = honest_m3

        logger.info("▶️ 有価買取データ処理開始")
        master_csv.loc[master_csv["大項目"] == "有価買取", "値"] = (
            calculate_purchase_value_of_valuable_items(df_receive)
        )

    # 最終処理
    logger.info("▶️ 売上・仕入・損益まとめ処理開始")
    master_csv = calculate_misc_summary_rows(master_csv, target_day)

    return master_csv
