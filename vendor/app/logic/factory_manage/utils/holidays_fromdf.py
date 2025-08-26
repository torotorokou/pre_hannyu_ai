def get_date_holidays(df):
    """
    df内の祝日フラグ=1の日付を一意に取得し、start_date～end_dateの範囲内で返す

    Args:
        df (pd.DataFrame): データ（'伝票日付'、'祝日フラグ'カラムが含まれていること）

    Returns:
        list[str]: 祝日の日付（YYYY-MM-DD 形式）のリスト
    """

    start_date = df["伝票日付"].min().date()
    end_date = df["伝票日付"].max().date()

    # print(f"🔍 祝日抽出範囲: {start_date} ～ {end_date}")

    # --- 祝日フラグが1の行のみ抽出 ---
    mask = df["祝日フラグ"] == 1
    holidays_series = df.loc[mask, "伝票日付"]

    # --- 重複除去 & 日付範囲内で絞り込み ---
    holidays = holidays_series.drop_duplicates()
    holidays = holidays[
        (holidays.dt.date >= start_date) & (holidays.dt.date <= end_date)
    ]

    # --- 日付型を文字列（YYYY-MM-DD）に変換してリスト化 ---
    holidays_list = holidays.dt.strftime("%Y-%m-%d").tolist()

    return holidays_list
