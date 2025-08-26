import pandas as pd
from utils.value_setter import set_value_fast, set_value_fast_safe
from utils.logger import app_logger


def write_sum_to_target_cell(
    df: pd.DataFrame,
    target_keys: list[str],
    target_values: list,
    value_column: str = "値",
) -> pd.DataFrame:
    """
    テンプレートの「合計」セルに値を書き込む。
    指定したキー構成に対応するセルに、合計値を設定する汎用関数。

    Parameters:
        df : pd.DataFrame
            対象データフレーム（通常はマスターCSV）
        target_keys : list[str]
            キー列名のリスト（例: ["業者CD", "業者名", "品名"]）
        target_values : list
            条件として一致させたい値のリスト（例: ["合計", None, None]）
        value_column : str
            合計を算出する対象の値列（デフォルト: "値"）

    Returns:
        pd.DataFrame : 合計が反映されたDataFrame
    """
    total = pd.to_numeric(df[value_column], errors="coerce").sum()

    df = set_value_fast_safe(
        df, target_keys, target_values, total, value_col=value_column
    )

    return df


# 工場日報のみ使用
def summarize_value_by_cell_with_label(
    df: pd.DataFrame,
    value_col: str = "値",
    cell_col: str = "セル",
    label_col: str = "有価名",
    fill_missing_cells: bool = False,
) -> pd.DataFrame:
    """
    セルごとに数値を合計し、ラベル・セルロック・順番を付けて返す。
    セルがNaNの行はgroupbyから自動的に除外されます。
    """

    # NaNセル処理（任意）
    if fill_missing_cells:
        df[cell_col] = df[cell_col].fillna("未設定")

    # 数値変換（NaNはそのまま）
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # セルごとに値を合計
    grouped = df.groupby(cell_col, as_index=False)[value_col].sum()

    # 補足情報（ラベル・セルロック・順番）を取り出し、重複除去
    cols_to_preserve = [cell_col, label_col, "セルロック", "順番"]
    cell_info = df[cols_to_preserve].drop_duplicates(subset=cell_col)

    # マージして返す
    grouped_named = pd.merge(grouped, cell_info, on=cell_col, how="left")

    return grouped_named


def summary_apply(
    master_csv: pd.DataFrame,
    data_df: pd.DataFrame,
    key_cols: list[str],
    source_col: str = "正味重量",
    target_col: str = "値",
) -> pd.DataFrame:
    """
    インポートCSVをgroupby＆sumし、マスターCSVにマージ＆更新する汎用関数（シート名なし版）。
    """
    logger = app_logger()
    logger.info(
        f"▶️ マスター更新処理: キー={key_cols}, 集計列={source_col} ➡ 書き込み列={target_col}"
    )

    # ① groupbyで合計
    agg_df = data_df.groupby(key_cols, as_index=False)[[source_col]].sum()

    # ② 安全にマージ
    merged_df = safe_merge_by_keys(
        master_df=master_csv.copy(), data_df=agg_df, key_cols=key_cols
    )

    # ③ 値を書き込み（NaN以外）
    updated_df = summary_update_column_if_notna(merged_df, source_col, target_col)

    # ④ source_colとtarget_colが違う場合だけ削除
    if source_col != target_col:
        updated_df.drop(columns=[source_col], inplace=True)

    return updated_df


def safe_merge_by_keys(
    master_df: pd.DataFrame,
    data_df: pd.DataFrame,
    key_cols: list[str],
    how: str = "left",
) -> pd.DataFrame:
    """
    指定した複数のキー列を使って、安全にマージする関数。
    マスター側のキー列に欠損値（NaN）がある行はマージ対象から除外される。

    Parameters
    ----------
    master_df : pd.DataFrame
        マージ元（テンプレート）
    data_df : pd.DataFrame
        マージ対象のデータ
    key_cols : list[str]
        結合に使用するキー列（1〜3列想定）
    how : str
        マージ方法（デフォルト: "left"）

    Returns
    -------
    pd.DataFrame
        マージ済みのDataFrame（未マージ行も含まれる）
    """

    # ① キーに空欄がある行を除外してマージ
    master_valid = master_df.dropna(subset=key_cols)
    data_valid = data_df.dropna(subset=key_cols)

    merged = pd.merge(master_valid, data_valid, on=key_cols, how=how)

    # ② キーが不完全（NaN含む）な行を保持して復元
    master_skipped = master_df[master_df[key_cols].isna().any(axis=1)]

    # ③ マージしたものと未マージのものを結合して返す
    final_df = pd.concat([merged, master_skipped], ignore_index=True)

    return final_df


def summary_update_column_if_notna(
    df, source_col: str, target_col: str
) -> pd.DataFrame:
    mask = df[source_col].notna()
    df.loc[mask, target_col] = df.loc[mask, source_col]
    return df
