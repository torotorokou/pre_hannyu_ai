# app_pages/manage/page.py
import streamlit as st
from app_pages.base_page import BasePage
from app_pages.manage.controller import manage_work_controller
from utils.page_config import PageConfig


class ManageWorkPage(BasePage):
    def __init__(self):
        config = PageConfig(page_id="manage", title="帳票作成", parent_title="管理業務")
        super().__init__(config)

    def render(self):
        self.render_title()
        # st.divider()
        manage_work_controller()  # コントローラ呼び出し
