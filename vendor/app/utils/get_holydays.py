from datetime import datetime, timedelta, date
import jpholiday
from typing import Union, List


def get_japanese_holidays(
    start: Union[str, date], end: Union[str, date], as_str: bool = True
) -> Union[List[str], List[date]]:
    """
    指定した期間の日本の祝日を取得する関数。

    Args:
        start (str or date): 開始日（"YYYY-MM-DD" または date型）
        end (str or date): 終了日（"YYYY-MM-DD" または date型）
        as_str (bool): Trueなら"YYYY-MM-DD"形式、Falseならdate型

    Returns:
        Union[List[str], List[date]]: 祝日のリスト（文字列またはdate型）
    """
    # --- 日付型でなければ変換 ---
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    # --- 日付範囲の祝日抽出 ---
    holidays = [
        d
        for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
        if jpholiday.is_holiday(d)
    ]

    return [d.strftime("%Y-%m-%d") for d in holidays] if as_str else holidays
