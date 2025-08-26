import pandas as pd
from utils.value_setter import set_value_fast, set_value_fast_safe
from utils.logger import app_logger
from logic.manage.utils.summary_tools import safe_merge_by_keys


def summary_add_column_if_notna(
    df: pd.DataFrame, from_col: str, to_col: str
) -> pd.DataFrame:
    if from_col not in df.columns or to_col not in df.columns:
        return df

    df = df.copy()
    df[from_col] = pd.to_numeric(df[from_col], errors="coerce").fillna(0)
    df[to_col] = pd.to_numeric(df[to_col], errors="coerce").fillna(0)

    df[to_col] = df[to_col] + df[from_col]
    return df


def apply_column_addition_by_keys(
    base_df: pd.DataFrame,
    addition_df: pd.DataFrame,
    join_keys: list[str],
    value_col_to_add: str = "加算",
    update_target_col: str = "単価",
) -> pd.DataFrame:
    logger = app_logger()
    logger.info(
        f"▶️ カラム加算処理（重複除外）: キー={join_keys}, 加算列={value_col_to_add} ➕ 対象列={update_target_col}"
    )

    # 🔁 同じ列名の場合は退避名を使う（列名衝突を防ぐ）
    temp_add_col = (
        f"__temp_add_{value_col_to_add}"
        if value_col_to_add == update_target_col
        else value_col_to_add
    )

    # ① 重複を除いた加算対象データの作成
    unique_add_df = addition_df.drop_duplicates(subset=join_keys)[
        join_keys + [value_col_to_add]
    ].rename(columns={value_col_to_add: temp_add_col})

    # ✅ ② base_df を join_keys に存在するものだけにフィルタ
    valid_keys = unique_add_df[join_keys].drop_duplicates()
    filtered_base_df = base_df.merge(valid_keys, on=join_keys, how="inner")

    # ③ マージして加算対象列を結合
    merged_df = safe_merge_by_keys(
        master_df=filtered_base_df, data_df=unique_add_df, key_cols=join_keys
    )

    # ④ 加算処理（NaNは0として扱う）
    updated_df = summary_add_column_if_notna(
        merged_df, from_col=temp_add_col, to_col=update_target_col
    )

    # ⑤ 加算用の一時列は削除
    if temp_add_col in updated_df.columns:
        updated_df.drop(columns=[temp_add_col], inplace=True)

    return updated_df
