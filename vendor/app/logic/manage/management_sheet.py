import pandas as pd
from utils.logger import app_logger
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_template_config
from utils.value_setter import set_value_fast_safe
from logic.manage.processors.management_sheet.factory_report import (
    update_from_factory_report,
)
from logic.manage.processors.management_sheet.balance_sheet import (
    update_from_balance_sheet,
)
from logic.manage.processors.management_sheet.average_sheet import (
    update_from_average_sheet,
)
from logic.manage.processors.management_sheet.sukurappu_senbetu import scrap_senbetsu

from logic.manage.processors.management_sheet.manage_etc import manage_etc


def process(dfs: dict) -> pd.DataFrame:
    """
    管理表テンプレート用のメイン処理関数。
    各種CSVデータを読み込み、工場日報・搬出入・ABC・スクラップ等の処理を適用し、
    最終的な管理表データフレームを返します。
    Parameters
    ----------
    dfs : dict
        各CSVのデータフレーム辞書
    Returns
    -------
    pd.DataFrame
        統合・加工済みの管理表データ
    """
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["management_sheet"]
    master_path = config["master_csv_path"]["management_sheet"]
    master_csv = load_master_and_template(master_path)

    # --- ② テンプレート設定と対象CSVの読み込み ---
    template_key = "management_sheet"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_receive = df_dict.get("receive")
    # df_shipping = df_dict.get("shipping")
    # df_yard = df_dict.get("yard")

    # --- ③ 各処理の適用 ---
    logger.info("▶️ 管理表_工場日報からの読込")
    master_csv = update_from_factory_report(dfs, master_csv)

    logger.info("▶️ 管理表_搬出入からの読込")
    master_csv = update_from_balance_sheet(dfs, master_csv)

    logger.info("▶️ 管理表_ABCからの読込")
    master_csv = update_from_average_sheet(dfs, master_csv)

    logger.info("▶️ スクラップ・選別")
    master_csv = scrap_senbetsu(df_receive, master_csv)

    logger.info("▶️ 日付・その他")
    etc_df = manage_etc(df_receive)

    logger.info("▶️ 結合")
    df_final = pd.concat([master_csv, etc_df], axis=0, ignore_index=True)

    return df_final
