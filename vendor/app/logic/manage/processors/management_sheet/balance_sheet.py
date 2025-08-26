from utils.value_setter import set_value_fast_safe
from logic.manage.balance_sheet import process as process_bal
from logic.manage.utils.dataframe_tools import (
    apply_summary_all_items,
    apply_division_result_to_master,
    apply_subtraction_result_to_master,
)


def update_from_balance_sheet(dfs, master_csv):
    csv_bal = process_bal(dfs)

    # 搬出入からの読込
    master_csv = apply_summary_all_items(master_csv, csv_bal)

    # 売上平均単価
    master_csv = apply_division_result_to_master(
        master_csv,
        numerator_item="売上金額",
        denominator_item="搬入量",
        target_item="売上平均単価",
    )

    # 仕入平均単価
    master_csv = apply_division_result_to_master(
        master_csv,
        numerator_item="仕入金額",
        denominator_item="搬出量",
        target_item="仕入平均単価",
    )

    # 粗利
    master_csv = apply_subtraction_result_to_master(
        master_csv,
        minuend_item="売上金額",
        subtrahend_item="仕入金額",
        target_item="粗利",
    )

    # 粗利単価当日
    master_csv = apply_subtraction_result_to_master(
        master_csv,
        minuend_item="売上平均単価",
        subtrahend_item="仕入平均単価",
        target_item="粗利単価（当日）",
    )

    return master_csv
