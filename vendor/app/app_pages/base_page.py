# app_pages/base_page.py

import streamlit as st
from abc import ABC, abstractmethod
from utils.page_config import PageConfig
from utils.logger import app_logger
from utils.config_loader import get_template_config


class BasePage(ABC):
    def __init__(self, config: PageConfig):
        self.page_id = config.page_id
        self.config = get_template_config().get(self.page_id, {})
        self.title = config.title or self.config.get("title", self.page_id)
        self.parent_title = config.parent_title
        self.show_parent_title = config.show_parent_title  # ← これ追加
        self.logger = app_logger()

    def render_title(self):
        st.markdown(f"## {self.title}")

    def log(self, message: str):
        self.logger.info(f"[{self.page_id}] {message}")

    def render_centered(self, render_fn):
        col1, col2, col3 = st.columns([2, 6, 2])
        with col2:
            render_fn()

    @abstractmethod
    def render(self):
        """サブクラスでページ描画を実装する必要があります"""
        pass
