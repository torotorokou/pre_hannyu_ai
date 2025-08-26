# app_pages/manage/page.py
import streamlit as st
from app_pages.base_page import BasePage
from app_pages.factory_manage.controller import factory_manage_work_controller
from utils.page_config import PageConfig


class FactoryManageWorkPage(BasePage):
    """
    FactoryManageWorkPage class for managing factory work pages.
    """

    def __init__(self):
        config = PageConfig(
            page_id="factory_manage",
            title="工場管理",
            parent_title="工場管理",
        )
        super().__init__(config)

    def render(self):
        self.render_title()
        factory_manage_work_controller()  # コントローラ呼び出し
