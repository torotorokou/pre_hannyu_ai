# カレンダー用のイベント作成処理
import pandas as pd


def df_to_calendar_events(df, date_col, title_prefix):
    events = []
    for d in pd.to_datetime(df[date_col]):
        events.append(
            {
                "id": f"{title_prefix}-{d}",
                "title": f"{title_prefix} 読み込み済",
                "start": d.strftime("%Y-%m-%d"),
            }
        )
    return events
