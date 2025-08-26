import streamlit as st
import tempfile
from components.status_box import render_status_box
from components.custom_button import centered_download_button
from components.ui_message import show_warning_bubble
from logic.detect_csv import detect_csv_type
from io import BytesIO
from typing import Optional
from utils.config_loader import get_csv_label_map


def render_manage_page(template_dict, template_descriptions):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 帳票作成メニュー")
    template_label = st.sidebar.radio(
        "出力したい項目を選択してください", list(template_dict.keys())
    )

    st.subheader(f"📝 {template_label} の作成")
    description = template_descriptions.get(template_label, "")
    if description:
        st.markdown(
            f"""<div style="margin-left: 2em; color:#ccc;">{description}</div>""",
            unsafe_allow_html=True,
        )

    return template_label  # ✅ 選択結果を返す


def show_upload_status(file):
    if not file:
        render_status_box(
            message="  未アップロード",
            bg_rgba="rgba(244, 67, 54, 0.07)",  # やや赤みのある背景
            text_color="#e57373",  # 明るめの赤
        )


def render_upload_header(title: str):
    st.markdown(
        f"""
    <div style="
        background-color: rgba(255, 223, 89, 0.15);  /* 上品な薄黄色 */
        color: #222;
        padding: 10px 16px;
        margin-top: 24px;
        margin-bottom: 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 15px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span style="font-size: 17px;">📁</span>
        <span>{title}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_semi_required_upload_header(title: str, description: str = ""):
    st.markdown(
        f"""
    <div style="
        background-color: rgba(255, 102, 0, 0.08);  /* 薄いオレンジレッド */
        color: #222;
        padding: 10px 16px;
        margin-top: 24px;
        margin-bottom: 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 15px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        gap: 4px;
    ">
        <div style="display: flex; align-items: center; gap: 8px; color: #cc5200;">
            <span style="font-size: 17px;">🟠</span>
            <span>{title}</span>
        </div>
        <div style="font-size: 13px; color: #cc5200;">
            {description}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# app_pages/manage/view.py
def render_status_message_ui(
    file_ready: bool,
    file_name: Optional[str] = None,
    output_excel: Optional[BytesIO] = None,
    uploaded_count: int = 0,
    total_count: int = 0,
):
    if file_ready and output_excel:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.info("✅ ファイルが生成されました。下のボタンからダウンロードできます👇")
        centered_download_button(
            label="📥 Excelファイルをダウンロード",
            data=output_excel.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")
