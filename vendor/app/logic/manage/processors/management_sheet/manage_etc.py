import pandas as pd
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_template_config
from utils.value_setter import set_value_fast_safe
from utils.date_tools import get_title_from_date


def manage_etc(df_receive):

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["management_sheet"]
    master_path = config["master_csv_path"]["etc"]
    master_csv = load_master_and_template(master_path)

    today = df_receive["伝票日付"][0]

    # 日付を記入
    master_csv = set_value_fast_safe(master_csv, ["大項目"], ["日"], today.day)

    # タイトルを記入
    title = get_title_from_date(today)
    master_csv = set_value_fast_safe(master_csv, ["大項目"], ["タイトル"], title)

    return master_csv
