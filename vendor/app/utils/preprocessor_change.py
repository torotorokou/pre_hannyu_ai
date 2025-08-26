import pandas as pd


def process_csv_by_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    CSVデータを日付カラムでソートし、最小日付のデータのみ抽出する。
    曜日 (例: "(月)") が含まれていても処理できるように対応。

    Parameters:
        df (pd.DataFrame): 処理対象のDataFrame
        date_column (str): 日付列の名前（例："伝票日付"）

    Returns:
        pd.DataFrame: ソート＆フィルタ済みのDataFrame
    """
    # 曜日などの文字を除去（例："2024/04/01(月)" → "2024/04/01"）
    df[date_column] = (
        df[date_column].astype(str).str.replace(r"\(.*?\)", "", regex=True).str.strip()
    )

    # 日付変換
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # 有効な日付だけに絞る
    df = df.dropna(subset=[date_column])

    # 日付で昇順ソート
    df = df.sort_values(by=date_column)

    # 最小日付を取得
    min_date = df[date_column].min()

    # 最小日付の行のみ抽出
    filtered_df = df[df[date_column] == min_date]

    return filtered_df


def check_date_alignment(dfs: dict, date_columns: dict) -> bool:
    import pandas as pd
    import streamlit as st

    """
    各DataFrameにおける日付のユニーク値が揃っているか確認。
    """
    date_sets = {}
    for key, df in dfs.items():
        date_col = date_columns.get(key)

        if not date_col or date_col not in df.columns:
            st.warning(f"⚠️ {key} に日付カラム {date_col} が見つかりません。")
            return False

        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date
        date_sets[key] = set(dates)

    keys = list(date_sets.keys())
    base_dates = date_sets[keys[0]]
    all_match = True

    for k in keys[1:]:
        if date_sets[k] != base_dates:
            st.error(f"❌ `{keys[0]}` と `{k}` の日付セットが一致していません。")
            st.write(f"- `{keys[0]}`: {sorted(base_dates)}")
            st.write(f"- `{k}`: {sorted(date_sets[k])}")
            all_match = False

    if all_match:
        st.success(f"✅ すべてのCSVで日付が一致しています：{sorted(base_dates)}")

    return all_match
