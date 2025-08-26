# app_pages/manage/page.py
import streamlit as st
from app_pages.base_page import BasePage
from utils.page_config import PageConfig
from logic.sanbo_navi.scr.controller import contoroller_education_gpt_page


class SanboNaviPage(BasePage):
    """
    SanboNaviPage class for managing Sanbo Navi pages.
    """

    def __init__(self):
        config = PageConfig(
            page_id="sanbo_navi",
            title="参謀くんNAVI",
            parent_title="参謀くんNAVI",
        )
        super().__init__(config)

    def render(self):
        self.render_title()
        # ここにページの内容を描画するためのコードを追加
        st.write("参謀くんNAVIのコンテンツをここに表示します。")
        contoroller_education_gpt_page()
