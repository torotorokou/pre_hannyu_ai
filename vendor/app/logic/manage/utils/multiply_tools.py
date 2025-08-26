import pandas as pd


def multiply_columns(
    df: pd.DataFrame, col1: str, col2: str, result_col: str = "値"
) -> pd.DataFrame:
    """
    指定された2列を掛け算して新しい列に保存する汎用関数。

    Parameters
    ----------
    df : pd.DataFrame
        対象のDataFrame
    col1 : str
        掛け算する1つ目の列名
    col2 : str
        掛け算する2つ目の列名
    result_col : str, default "値"
        計算結果を格納する列名

    Returns
    -------
    pd.DataFrame
        掛け算列を追加したDataFrame
    """
    df = df.copy()

    df[col1] = pd.to_numeric(df[col1], errors="coerce")
    df[col2] = pd.to_numeric(df[col2], errors="coerce")

    df[result_col] = df[col1] * df[col2]

    return df
