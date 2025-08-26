from utils.value_setter import set_value_fast_safe
from logic.manage.factory_report import process as process_fact
from logic.manage.utils.dataframe_tools import apply_summary_all_items


# 工場日報からの読込
def update_from_factory_report(dfs, master_csv):
    csv_fac = process_fact(dfs)

    # 工場日報からの読込
    master_csv = apply_summary_all_items(master_csv, csv_fac)

    return master_csv
