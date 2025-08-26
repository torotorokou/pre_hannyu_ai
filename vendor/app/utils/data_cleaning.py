import pandas as pd


def clean_cd_column(df: pd.DataFrame, col: str = "業者CD") -> pd.DataFrame:
    valid = df[col].notna()

    # ① 一旦文字列として変換 → ② intに変換 → ③ Series全体に代入（dtypeを明示）
    cleaned = df.loc[valid, col].apply(lambda x: int(float(x)))
    df.loc[valid, col] = cleaned.astype("Int64")  # ← Nullable Int 型（Pandas公式推奨）
    return df
