from logic.manage.utils.excel_tools import (
    add_label_rows_and_restore_sum,
    add_label_rows,
)


def make_label(df):
    # 有価ラベル
    final_df = add_label_rows(df, label_source_col="大項目", offset=-1)

    # # 処分ラベル
    # final_df = add_label_rows(df, label_col="業者名", offset=-1)

    # final_df = add_label_rows_and_restore_sum(
    #     final_df, label_col="業者名", offset=-1
    # )

    # # ヤードラベル
    # final_df = add_label_rows_and_restore_sum(
    #     final_df, label_col="品目名", offset=-1
    # )

    return final_df
