import pandas as pd


def round_value_column_generic(
    df: pd.DataFrame, key_columns: list[str], value_column: str = "値"
) -> pd.DataFrame:
    """
    任意のキー列群に「単価」が含まれていればその行を単価行とみなし、
    指定された値列に対して丸め処理を行う。

    Parameters:
        df (pd.DataFrame): 入力DataFrame
        key_columns (list[str]): 「単価」キーワードをチェックする列名リスト
        value_column (str): 丸め対象の列名（デフォルト: "値"）

    Returns:
        pd.DataFrame: 丸め後のDataFrame（破壊的変更あり）
    """

    # --- 単価判定用フラグ列（複数列に "単価" を含むかどうか） ---
    is_tanka = pd.Series(False, index=df.index)
    for col in key_columns:
        is_tanka |= df[col].astype(str).str.contains("単価", na=False)

    # --- 値を数値に変換（文字列・日付は除外） ---
    numeric_vals = pd.to_numeric(df[value_column], errors="coerce")
    is_numeric = ~numeric_vals.isna()

    # --- 値の初期化 ---
    rounded = df[value_column].copy()

    # --- 単価行：小数第2位で丸め ---
    mask_tanka = is_tanka & is_numeric & (numeric_vals != 0)
    rounded.loc[mask_tanka] = numeric_vals.loc[mask_tanka].round(2)

    # --- その他：整数で丸め ---
    mask_non_tanka = ~is_tanka & is_numeric
    rounded.loc[mask_non_tanka] = (
        numeric_vals.loc[mask_non_tanka].round(0).astype("Int64")
    )

    # --- 結果を反映 ---
    df[value_column] = rounded

    return df
