from utils.logger import app_logger
from utils.value_setter import set_value_fast_safe


def reflect_total_from_factory(master_csv, df_factory):

    total_sum = df_factory.loc[df_factory["大項目"] == "総合計", "値"].squeeze()

    match_columns = ["大項目"]
    match_value = ["搬出量"]
    master_csv = set_value_fast_safe(master_csv, match_columns, match_value, total_sum)

    return master_csv


def process_factory_report(dfs, master_csv):
    logger = app_logger()
    from logic.manage.factory_report import process

    # 工場日報からdfを読込
    df_factory = process(dfs)

    # 搬出量を抜出
    after_master_csv = reflect_total_from_factory(master_csv, df_factory)

    return after_master_csv
