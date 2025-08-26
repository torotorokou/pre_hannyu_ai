import pandas as pd
from datetime import date, timedelta

import altair as alt
import streamlit as st

from app_pages.factory_manage.pages.inbound_volume_forecast.controller import (
    csv_controller,
)
from logic.factory_manage.calender import render_calendar_section
from logic.factory_manage.download_bottun import render_download_button
from logic.factory_manage.predict_model_ver2 import predict_controller
from logic.factory_manage.style import style_label


def render_prediction_section(start_date, end_date):
    """
    搬入量予測を実行し、結果をセッションに保存する。

    Parameters:
        start_date (date): 予測対象の開始日
        end_date (date): 予測対象の終了日
    """
    with st.spinner("予測中..."):
        df_pred = predict_controller(start_date=str(start_date), end_date=str(end_date))

    df_pred = df_pred.copy()
    df_pred["曜日"] = pd.to_datetime(df_pred.index).weekday.map(
        lambda x: "月火水木金土日"[x]
    )
    df_pred["日付"] = df_pred.index
    st.session_state["df_import_prediction"] = df_pred
    st.success("予測が完了しました。")


def render_prediction_table_and_chart():
    """
    セッションに保存された予測結果を表と棒グラフで表示する。
    ラベルごとのフィルタリングも可能。
    """
    df_pred = st.session_state["df_import_prediction"]

    label_filter = st.multiselect(
        "表示するラベル",
        options=df_pred["判定ラベル"].unique(),
        default=list(df_pred["判定ラベル"].unique()),
    )
    df_filtered = df_pred[df_pred["判定ラベル"].isin(label_filter)]

    df_display = df_filtered.copy()
    for col in ["予測値", "補正後予測", "下限95CI", "上限95CI"]:
        df_display[col] = df_display[col].round(0).astype(int)
    df_display["未満確率"] = df_display["未満確率"].apply(
        lambda x: f"{float(x) * 100:.1f}%" if pd.notnull(x) else ""
    )

    df_show = df_display[
        [
            "日付",
            "曜日",
            "予測値",
            "補正後予測",
            "下限95CI",
            "上限95CI",
            "判定ラベル",
            "未満確率",
        ]
    ].reset_index(drop=True)
    st.dataframe(df_show.style.applymap(style_label, subset=["判定ラベル"]))

    chart_data = df_display.copy()
    chart_data["日付"] = pd.to_datetime(chart_data["日付"])
    chart_data["日付_str"] = chart_data["日付"].dt.strftime("%m/%d")

    y_max = chart_data["補正後予測"].max()
    y_buffer = int(y_max * 0.1)

    bar = (
        alt.Chart(chart_data)
        .mark_bar(size=30, stroke="blue", strokeWidth=1)
        .encode(
            x=alt.X("日付_str:N", title="日付"),
            y=alt.Y(
                "補正後予測:Q",
                title="補正後予測",
                scale=alt.Scale(domain=[0, y_max + y_buffer]),
            ),
            color=alt.Color(
                "判定ラベル:N",
                scale=alt.Scale(
                    domain=["警告", "注意", "通常"], range=["red", "orange", "#4c78a8"]
                ),
                legend=None,
            ),
            tooltip=["日付_str:N", "補正後予測:Q", "判定ラベル:N"],
        )
    )
    st.altair_chart(bar.properties(height=400), use_container_width=True)


def render_import_volume():
    """
    Streamlitアプリの搬入量予測ページを構成。
    - カレンダーの表示
    - CSVのアップロード
    - 予測期間の指定と実行
    - 結果表示とチャート描画
    - CSVダウンロードボタンの表示
    """
    st.title("📊 搬入量予測AI")

    st.subheader("📅 読込済CSVカレンダー")
    st.markdown(
        """現在読込済みのCSV一覧表です。  
    追加する場合は、以下からCSVをアップロードして下さい。"""
    )
    render_calendar_section()

    with st.expander("📂 CSVのアップロードはこちらをクリック", expanded=False):
        st.markdown("""追加したいCSVをアップロードして下さい。""")
        csv_controller()

    st.subheader("📅 予測期間の選択")
    st.markdown(
        """予測したい期間を選択して下さい。  
    デフォルトは今日から土曜日までです。"""
    )

    today = date.today()
    # 今日から今週の土曜日まで
    days_until_saturday = (5 - today.weekday()) % 7
    default_start = today
    default_end = today + timedelta(days=days_until_saturday)
    selected_dates = st.date_input("期間を選択", value=(default_start, default_end))

    if not (isinstance(selected_dates, tuple) and len(selected_dates) == 2):
        st.info("開始日と終了日を選択してください。")
        return

    start_date, end_date = selected_dates

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("予測を実行する"):
            render_prediction_section(start_date, end_date)

    if "df_import_prediction" in st.session_state:
        render_prediction_table_and_chart()
        render_download_button(start_date, end_date)
