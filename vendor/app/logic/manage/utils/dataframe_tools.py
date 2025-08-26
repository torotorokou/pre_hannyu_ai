import pandas as pd
from utils.value_setter import set_value_fast_safe


def apply_summary_all_items(
    master_csv: pd.DataFrame,
    csv_fac: pd.DataFrame,
    search_col: str = "検索ワード",
    fac_key_col: str = "大項目",
    value_col: str = "値",
) -> pd.DataFrame:
    """
    master_csv の検索ワード列と csv_fac の大項目を照合し、一致するものがあれば
    csv_fac の値を master_csv の値列に反映する。

    Parameters:
        master_csv: 値を書き込むテンプレート（検索ワード列を含む）
        csv_fac: 実績データ（大項目と値を含む）
        search_col: master_csv 側の照合に使う列（例: "検索ワード1"）
        fac_key_col: csv_fac 側の照合に使う列（例: "大項目"）
        value_col: 値の列（両方で共通。例: "値"）

    Returns:
        pd.DataFrame: 値を反映した master_csv
    """
    updated_csv = master_csv.copy()

    for i, row in updated_csv.iterrows():
        search_word = row.get(search_col)

        # 実績データ内に検索ワードが存在すれば、その値を取得
        value_series = csv_fac.loc[csv_fac[fac_key_col] == search_word, value_col]

        if not value_series.empty:
            value = value_series.values[0]
            updated_csv.at[i, value_col] = value

    return updated_csv


def apply_division_result_to_master(
    df: pd.DataFrame,
    numerator_item: str,
    denominator_item: str,
    target_item: str,
    key_col: str = "大項目",
    value_col: str = "値",
) -> pd.DataFrame:
    """
    指定された項目同士で割り算を行い、その結果をマスターに反映する。
    オブジェクト型でも安全に数値変換して演算する。
    """
    # 対象列を数値型に変換
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    numerator = df.loc[df[key_col] == numerator_item, value_col]
    denominator = df.loc[df[key_col] == denominator_item, value_col]

    if numerator.empty or denominator.empty:
        return df  # 値が見つからない場合はスキップ

    num = numerator.values[0]
    denom = denominator.values[0]

    if pd.isna(num) or pd.isna(denom) or denom == 0:
        return df  # NaN や 0除算を防止

    result = num / denom
    df.loc[df[key_col] == target_item, value_col] = result
    return df


def apply_subtraction_result_to_master(
    df: pd.DataFrame,
    minuend_item: str,
    subtrahend_item: str,
    target_item: str,
    key_col: str = "大項目",
    value_col: str = "値",
) -> pd.DataFrame:
    """
    指定された2項目で引き算を行い、結果を別の項目に反映。
    オブジェクト型でも安全に数値変換して演算する。
    """
    # 対象列を数値型に変換
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    minuend = df.loc[df[key_col] == minuend_item, value_col]
    subtrahend = df.loc[df[key_col] == subtrahend_item, value_col]

    if minuend.empty or subtrahend.empty:
        return df  # 値が見つからない場合はスキップ

    a = minuend.values[0]
    b = subtrahend.values[0]

    if pd.isna(a) or pd.isna(b):
        return df  # NaNのまま演算しない

    result = a - b
    df.loc[df[key_col] == target_item, value_col] = result
    return df
