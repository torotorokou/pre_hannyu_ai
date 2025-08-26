import pandas as pd
from sqlalchemy import create_engine
from utils.config_loader import get_path_from_yaml
from utils.get_holydays import get_japanese_holidays
from datetime import datetime


def check_sql():
    """
    SQLiteデータベースから祝日フラグ付き伝票日付を抽出し、表示するデバッグ用関数。
    Returns
    -------
    なし
    """
    # --- SQLiteデータベース接続 ---
    db_path = get_path_from_yaml("weight_data", section="sql_database")
    engine = create_engine(f"sqlite:///{db_path}")

    # --- クエリ実行（祝日フラグが1のレコードだけ取得。日付重複なし）---
    query = """
        SELECT DISTINCT 伝票日付, 祝日フラグ
        FROM ukeire
        ORDER BY 伝票日付
    """
    df = pd.read_sql(query, engine)
    # --- 結果表示 ---
    print(df)


def check_holidays():
    """
    2020-2025年の祝日リストを取得し、表示するデバッグ用関数。
    Returns
    -------
    なし
    """
    # --- 祝日フラグ追加 ---
    start_date = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    end_date = datetime.strptime("2025-12-31", "%Y-%m-%d").date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    print(holidays)


if __name__ == "__main__":
    check_sql()
    # check_holidays()
