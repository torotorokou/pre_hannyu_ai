import pandas as pd
from utils.logger import app_logger
from logic.manage.utils.summary_tools import (
    safe_merge_by_keys,
    summary_update_column_if_notna,
)


# --- utils/filters.py ---
def apply_negation_filters(
    df: pd.DataFrame, match_df: pd.DataFrame, key_cols: list[str], logger=None
) -> pd.DataFrame:
    """
    match_df の key_cols に `Not値` または `NOT値` があれば、その値を除外するフィルタを df に適用。
    """
    filter_conditions = {}
    for col in key_cols:
        if col not in df.columns:
            if logger:
                logger.warning(f"⚠️ データに列 '{col}' が存在しません。スキップします。")
            continue

        unique_vals = match_df[col].dropna().unique()
        neg_vals = [
            v[3:]
            for v in unique_vals
            if isinstance(v, str) and v.lower().startswith("not")
        ]
        if neg_vals:
            filter_conditions[col] = neg_vals
            if logger:
                logger.info(
                    f"🚫 '{col}' に対して否定フィルタ: {', '.join(neg_vals)} を適用しました"
                )

    for col, ng_values in filter_conditions.items():
        df = df[~df[col].isin(ng_values)]

    return df


# --- processors/summary.py ---
def process_sheet_partition(
    master_csv: pd.DataFrame, sheet_name: str, expected_level: int, logger=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    指定シートから key_level 一致行と不一致行を分離。
    """
    sheet_df = master_csv[master_csv["CSVシート名"] == sheet_name].copy()

    if "key_level" not in sheet_df.columns:
        if logger:
            logger.warning("❌ key_level列が存在しません。スキップします。")
        return pd.DataFrame(), pd.DataFrame()

    try:
        match_df = sheet_df[sheet_df["key_level"].astype(int) == expected_level].copy()
        remain_df = sheet_df[sheet_df["key_level"].astype(int) != expected_level].copy()
        return match_df, remain_df
    except Exception as e:
        if logger:
            logger.error(f"❌ key_level変換エラー: {e}")
        return pd.DataFrame(), pd.DataFrame()


def summary_apply_by_sheet(
    master_csv: pd.DataFrame,
    data_df: pd.DataFrame,
    sheet_name: str,
    key_cols: list[str],
    source_col: str = "正味重量",
    target_col: str = "値",
) -> pd.DataFrame:
    """
    master_csv に対し、data_df を key_cols で groupby & sum してマージする。
    master_csv の key_level によるフィルタ、および `Not値` による not検索をサポート。
    `Not値` を含む列はマージキーから除外する。
    """
    logger = app_logger()
    logger.info(f"▶️ シート: {sheet_name}, キー: {key_cols}, 集計列: {source_col}")

    # --- 該当シートの key_level フィルタ ---
    expected_level = len(key_cols)
    match_df, remain_df = process_sheet_partition(
        master_csv, sheet_name, expected_level, logger
    )

    if match_df.empty:
        logger.info(
            f"⚠️ key_level={expected_level} に一致する行がありません。スキップします。"
        )
        return master_csv

    # --- not検索を適用（Not値のある行を除外） ---
    filtered_data_df = apply_negation_filters(
        data_df.copy(), match_df, key_cols, logger
    )

    # --- マージ用 key を再定義（Not〇〇を含む列を除外） ---
    merge_key_cols = []
    for col in key_cols:
        if col in match_df.columns:
            has_neg = any(
                isinstance(val, str) and val.lower().startswith("not")
                for val in match_df[col].dropna().unique()
            )
            if not has_neg:
                merge_key_cols.append(col)
            else:
                logger.info(f"⚠️ '{col}' に 'Not' 指定があるためマージキーから除外")

    if not merge_key_cols:
        logger.warning("❌ 有効なマージキーが存在しません。処理をスキップします。")
        return master_csv

    # --- 集計 ---
    agg_df = filtered_data_df.groupby(merge_key_cols, as_index=False)[
        [source_col]
    ].sum()

    # --- マージ ---
    merged_df = safe_merge_by_keys(match_df, agg_df, merge_key_cols)
    merged_df = summary_update_column_if_notna(merged_df, source_col, target_col)

    # --- 正味重量の削除 ---
    if source_col in merged_df.columns:
        merged_df.drop(columns=[source_col], inplace=True)

    # --- 最終結合（元データの他シート + 残余 + マージ結果）---
    master_others = master_csv[master_csv["CSVシート名"] != sheet_name]
    final_df = pd.concat([master_others, remain_df, merged_df], ignore_index=True)

    return final_df
