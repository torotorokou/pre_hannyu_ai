import streamlit as st
import pandas as pd
import sqlite3
import calendar
from datetime import datetime, timedelta
from utils.config_loader import get_path_from_yaml
from logic.factory_manage.sql import load_recent_dates_from_sql
from logic.factory_manage.style import calendar_header_html, calendar_body_html


def render_calendar_section():
    """
    過去3ヶ月分の搬入データのある日をカレンダー形式で表示する。
    データがある日は緑色でハイライトされる。

    Returns:
        None
    """
    sql_url = get_path_from_yaml("weight_data", section="sql_database")
    dates_with_data = load_recent_dates_from_sql(sql_url)
    months = [datetime.today() - pd.DateOffset(months=i) for i in range(2, -1, -1)]
    cols = st.columns(3)
    today = datetime.today().date()

    for i, month in enumerate(months):
        with cols[i]:
            html = generate_calendar_html(
                month.year, month.month, dates_with_data, today
            )
            st.markdown(html, unsafe_allow_html=True)


def generate_calendar_html(
    year: int,
    month: int,
    highlight_dates: list[datetime.date],
    today: datetime.date = None,
) -> str:
    """
    指定された年月のHTMLカレンダーを生成する。

    Args:
        year (int): 対象の年（例: 2025）
        month (int): 対象の月（例: 5）
        highlight_dates (list[date]): ハイライトする日付リスト
        today (date, optional): 今日の日付

    Returns:
        str: HTMLカレンダーの文字列
    """
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)

    html = calendar_header_html(year, month)
    html += calendar_table_header_row()
    html += calendar_body_html(weeks, year, month, highlight_dates, today)
    html += "</table></div><br>"

    return html


def calendar_table_header_row() -> str:
    """
    カレンダーの曜日ヘッダー行をHTMLとして生成。
    白・黒背景両対応の視認性の高い配色を使用。

    Returns:
        str: 曜日ヘッダーのHTML
    """
    weekdays = ["日", "月", "火", "水", "木", "金", "土"]
    # 白黒背景両対応：少し明るめ・濃すぎない色
    weekday_colors = [
        "#e53935",  # 日: 明るめ赤
        "#444",  # 月: 中間グレー
        "#444",  # 火
        "#444",  # 水
        "#444",  # 木
        "#444",  # 金
        "#1e88e5",  # 土: 明るめ青
    ]

    # 適度に自然なtext-shadow（光沢感は抑えめ）
    text_shadow = "0 1px 1px rgba(0,0,0,0.3), 0 1px 2px rgba(255,255,255,0.1)"

    row = "<tr>"
    for i, day in enumerate(weekdays):
        row += (
            f"<th style='padding: 6px; "
            f"color: {weekday_colors[i]}; "
            f"font-weight: 600; "
            f"text-shadow: {text_shadow}; "
            f"font-size: 14px;'>"
            f"{day}</th>"
        )
    row += "</tr>"
    return row
