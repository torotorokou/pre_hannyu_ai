import pandas as pd
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.sql import save_df_to_sqlite_unique
from utils.config_loader import get_path_from_yaml
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import get_expected_dtypes_by_template

# from logic.factory_manage.predict_model_ver2 import get_df


def make_sql_old(df_raw: pd.DataFrame = None) -> pd.DataFrame:
    """
    過去の複数年分のCSVファイルと最新データを読み込み、
    日付や数値整形、不要データの除外、祝日フラグ付与を行い、
    SQLiteに統合保存するメイン関数。
    """
    base_dir = get_path_from_yaml("input", section="directories")

    # データ読込
    # df_raw = read_csv_hannnyuu_old()

    # --- 祝日フラグ付与 ---
    start_date = df_raw["伝票日付"].min().date()
    end_date = df_raw["伝票日付"].max().date()
    print(f"🔍 df2_min_max: {start_date} ～ {end_date}")
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)

    df_raw["祝日フラグ"] = df_raw["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )
    print("🎌 祝日フラグ付与完了")

    # --- SQLite保存 ---
    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(df=df_raw, db_path=db_path, table_name="ukeire")
        print("✅ SQLite保存完了")
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")

    return df_raw


def make_sql_db(df: pd.DataFrame):
    """
    与えられたデータフレームから無効データを削除し、
    整形・祝日付与をしてSQLiteに保存する。

    Args:
        df (pd.DataFrame): 元データ
    """
    print(f"🔍 元データの行数: {len(df)}")
    df = df.copy()
    df["伝票日付"] = df["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")

    dropped_rows = df[df[["伝票日付", "正味重量"]].isna().any(axis=1)]
    dropped_rows.to_csv("dropped_rows.csv", index=False)

    df = df.dropna(subset=["正味重量", "伝票日付"])
    print(f"🔍 dropna後の行数: {len(df)}")

    df = df[df["正味重量"] != 0]
    print(f"🔍 正味重量≠0 の行数: {len(df)}")

    start_date = df["伝票日付"].min().date()
    end_date = df["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)
    df["祝日フラグ"] = df["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )

    df = df.loc[:, ["伝票日付", "正味重量", "品名", "祝日フラグ"]]
    print(f"🔍 整形後の行数: {len(df)}")

    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(
            df=df,
            db_path=db_path,
            table_name="ukeire",
        )
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


def make_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    テンプレートに基づいて受入データの空白除去・型変換を行う。

    Args:
        df (pd.DataFrame): 元データ

    Returns:
        pd.DataFrame: 整形済データ
    """
    df = strip_whitespace(df)
    expected_dtypes = get_expected_dtypes_by_template("inbound_volume")
    dtypes = expected_dtypes.get("receive", {})

    if dtypes:
        df = enforce_dtypes(df, dtypes)

    return df


def read_csv_hannnyuu():
    """
    搬入量予測に必要なCSVデータを読み込んで統合・整形する関数。

    Returns
    -------
    pd.DataFrame
        整形済みの搬入データ（列: 伝票日付・品名・正味重量）
    """
    # --- データ取得 ---
    base_dir = get_path_from_yaml("input", section="directories")

    # --- 新データ（2024～2025） ---
    df_new = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")[
        ["伝票日付", "正味重量", "品名"]
    ]
    df_new["伝票日付"] = df_new["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_new["伝票日付"] = pd.to_datetime(df_new["伝票日付"], errors="coerce")

    # --- 旧データ（2020〜2023） ---
    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2022 = pd.read_csv(f"{base_dir}/2022顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv", low_memory=False)[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_old = pd.concat([df_2020, df_2021, df_2022, df_2023], ignore_index=True)
    df_old.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    df_old["伝票日付"] = pd.to_datetime(df_old["伝票日付"], errors="coerce")

    # --- 結合とクリーニング ---
    df_all = pd.concat([df_new, df_old], ignore_index=True)
    df_all["正味重量"] = pd.to_numeric(df_all["正味重量"], errors="coerce")
    df_all = df_all.dropna(subset=["正味重量", "伝票日付"])

    # --- 確認出力 ---
    start_date = df_all["伝票日付"].min().date()
    end_date = df_all["伝票日付"].max().date()
    print(f"🔍 df1_min_max: {start_date} ～ {end_date}")

    return df_all


def read_csv_hannnyuu_old():
    """
    旧形式の搬入量CSVデータを読み込んで整形・統合する関数。

    Returns
    -------
    pd.DataFrame
        整形済みの搬入データ
    """
    base_dir = get_path_from_yaml("input", section="directories")
    df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")
    df_raw = df_raw[["伝票日付", "正味重量", "品名"]]
    df_raw["伝票日付"] = df_raw["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")

    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")

    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")

    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv")

    df_2020 = df_2020[["伝票日付", "商品", "正味重量_明細"]]

    df_2021 = df_2021[["伝票日付", "商品", "正味重量_明細"]]

    df_2023 = df_2023[["伝票日付", "商品", "正味重量_明細"]]

    df_all = pd.concat([df_2020, df_2021, df_2023])

    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"])

    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)

    df_raw = pd.concat([df_raw, df_all])

    return df_all


# --- 実行 ---
if __name__ == "__main__":
    # df = get_df()
    make_sql_old(df)
