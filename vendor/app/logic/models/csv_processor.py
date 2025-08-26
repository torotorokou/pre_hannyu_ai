import pandas as pd


def process_csv_by_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df[date_column] = (
        df[date_column].astype(str).str.replace(r"\(.*?\)", "", regex=True).str.strip()
    )
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df = df.sort_values(by=date_column)
    min_date = df[date_column].min()
    return df[df[date_column] == min_date]


def check_date_alignment(dfs: dict, date_columns: dict) -> dict:
    date_sets = {}
    for key, df in dfs.items():
        date_col = date_columns.get(key)
        if date_col not in df.columns:
            return {
                "status": False,
                "error": f"{key} に日付カラム {date_col} が見つかりません。",
            }
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date
        date_sets[key] = set(dates)

    keys = list(date_sets.keys())
    base_dates = date_sets[keys[0]]
    for k in keys[1:]:
        if date_sets[k] != base_dates:
            return {
                "status": False,
                "error": f"`{keys[0]}` と `{k}` の日付セットが一致していません。CSVの日付を確認して、再実行して下さい",
                "details": {keys[0]: sorted(base_dates), k: sorted(date_sets[k])},
            }

    return {"status": True, "dates": sorted(base_dates)}
