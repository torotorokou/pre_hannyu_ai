from datetime import datetime, date
import pandas as pd
import datetime


def get_weekday_japanese(date_input):
    """日付（strまたはdatetime）から日本語の曜日を返す"""
    weekdays_ja = ["日", "月", "火", "水", "木", "金", "土"]

    if isinstance(date_input, str):
        # 区切り文字に応じてパース
        if "/" in date_input:
            date_input = datetime.strptime(date_input, "%Y/%m/%d")
        elif "-" in date_input:
            date_input = datetime.strptime(date_input, "%Y-%m-%d")
        else:
            raise ValueError(
                "日付形式は 'YYYY-MM-DD' または 'YYYY/MM/DD' にしてください"
            )

    weekday_index = (date_input.weekday() + 1) % 7
    return weekdays_ja[weekday_index]


def extract_first_date(df: pd.DataFrame, col: str = "伝票日付") -> pd.Timestamp:
    """データから最初の日付を返す（責務①）"""
    return pd.to_datetime(df[col].dropna().iloc[0])


def to_japanese_era(dt) -> str:
    """
    西暦の日付を「令和7年」などの和暦表記に変換する関数。
    pandas.Timestamp にも対応。
    """
    if hasattr(dt, "date"):
        dt = dt.date()

    if dt >= datetime.date(2019, 5, 1):
        year = dt.year - 2018
        era = "令和"
    elif dt >= datetime.date(1989, 1, 8):
        year = dt.year - 1988
        era = "平成"
    elif dt >= datetime.date(1926, 12, 25):
        year = dt.year - 1925
        era = "昭和"
    else:
        return f"{dt.year}年"

    return f"{era}元年" if year == 1 else f"{era}{year}年"


def to_japanese_month_day(dt) -> str:
    """
    任意の日付オブジェクトから「3月1日」の形式に変換する。
    pandas.Timestamp / datetime.date 両対応。

    Parameters
    ----------
    dt : datetime.date, datetime.datetime, pandas.Timestamp
        日付オブジェクト

    Returns
    -------
    str : 例 "3月1日"
    """
    if hasattr(dt, "date"):
        dt = dt.date()
    return f"{dt.month}月{dt.day}日"


def to_reiwa_format(d) -> str:
    """
    日付を「R7年〇月〇日」のような令和表記に変換する。

    Parameters
    ----------
    d : datetime.date, datetime.datetime, pandas.Timestamp など
        対象の日付オブジェクト（型を問わず対応）

    Returns
    -------
    str
        令和表記の文字列（例: R7年5月8日）
    """
    # 令和開始日（比較しやすいように pd.Timestamp に統一）
    reiwa_start = pd.Timestamp("2019-05-01")

    if pd.isna(d):
        return ""

    # 型が文字列などの場合にも対応
    d = pd.to_datetime(d)

    if d < reiwa_start:
        raise ValueError("令和以前の日付は対応していません")

    reiwa_year = d.year - 2018  # R1年 = 2019年
    return f"R{reiwa_year}年{d.month}月{d.day}日"


def get_title_from_date(d: date) -> str:
    reiwa_year = d.year - 2018  # 令和元年は2019年
    return f"管理票　R{reiwa_year} .{d.month}"
