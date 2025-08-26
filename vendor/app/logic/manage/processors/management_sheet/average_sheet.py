from utils.value_setter import set_value_fast_safe
from logic.manage.average_sheet import process as process_ave
from logic.manage.utils.dataframe_tools import (
    apply_summary_all_items,
)


def update_from_average_sheet(dfs, master_csv):
    csv_ave = process_ave(dfs)

    # 検索ワード作成
    csv_ave["大項目"] = csv_ave["大項目"] + csv_ave["品目_台数他"]
    # 搬出入からの読込
    master_csv = apply_summary_all_items(master_csv, csv_ave)

    return master_csv
