from datetime import datetime
import calendar


# --- 表のスタイル指定用（ラベル別色） ---
def style_label(val):
    """
    ラベル値に応じてセルのスタイル（色・太字）を返す。

    Parameters
    ----------
    val : str
        判定ラベル（例: "警告", "注意" など）

    Returns
    -------
    str
        CSSスタイル文字列
    """
    if val == "警告":
        return "color: red; font-weight: bold"
    elif val == "注意":
        return "color: orange"
    return ""


def calendar_body_html(
    weeks,
    year: int,
    month: int,
    highlight_dates: list[datetime.date],
    today: datetime.date = None,
) -> str:
    """
    カレンダーの日付部分のHTMLを生成。

    Args:
        weeks (list): 月の週ごとの日リスト
        year (int): 年
        month (int): 月
        highlight_dates (list[date]): ハイライト対象日
        today (date, optional): 今日の日付

    Returns:
        str: カレンダー本体のHTML
    """
    html = ""
    for week in weeks:
        html += "<tr>"
        for i, day in enumerate(week):
            if day == 0:
                html += "<td style='padding: 6px;'></td>"
            else:
                date_obj = datetime(year, month, day).date()
                has_data = date_obj in highlight_dates
                bg_color = "#90ee90" if has_data else "#f5f5f5"
                text_color = "#222" if has_data else "#999"
                border = (
                    "2px solid red"
                    if today is not None and date_obj == today
                    else "1px solid #ddd"
                )

                html += f"""
                <td style='
                    padding: 6px;
                    text-align: center;
                    background-color: {bg_color};
                    color: {text_color};
                    border-radius: 8px;
                    border: {border};
                    width: 36px;
                    height: 36px;
                    font-weight: 500;
                '>{day}</td>"""
        html += "</tr>"
    return html


def calendar_header_html(year: int, month: int) -> str:
    """
    カレンダー全体の外枠とタイトルのHTMLを生成。

    Returns:
        str: HTMLの開始部分
    """
    return f"""
    <div style='
        background-color: var(--background);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 0 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0 auto;
        color: var(--text);
    '>
    <h4 style='margin-bottom: 0.5rem; font-weight: 600;'>{year}年 {month}月</h4>
    <table style='border-collapse: collapse; margin: 0 auto; font-size: 13px;'>
    """
