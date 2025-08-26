import streamlit as st
from utils.config_loader import load_factory_menu_options
from utils.config_loader import load_yaml
from app_pages.factory_manage.view.menu import render_sidebar
from app_pages.factory_manage.view.css import inject_sidebar_css

# レンダーに必要な関数
from app_pages.factory_manage.pages.inbound_outbound_records.render import (
    render_inbound_outbound_records,
)
from app_pages.factory_manage.pages.balance_management_table.controller import (
    factory_manage_controller,
)
from app_pages.factory_manage.pages.inbound_volume_forecast.render import (
    render_import_volume,
)


def factory_manage_work_controller():
    controller = FactoryManageWorkController()
    controller.run()


class FactoryManageWorkController:
    def __init__(self):
        self.menu_options = load_factory_menu_options()
        self.selected_menu_key = None

    def run(self):
        inject_sidebar_css()
        labels = [item["label"] for item in self.menu_options]
        selected_label = render_sidebar(labels)
        self.selected_menu_key = next(
            item["key"] for item in self.menu_options if item["label"] == selected_label
        )
        self.route()

    def route(self):
        option = next(
            (
                menu_key
                for menu_key in self.menu_options
                if menu_key["key"] == self.selected_menu_key
            ),
            None,
        )

        if option is None:
            st.error("メニューが正しく選択されていません。")
            return

        handler_name = option.get("handler")
        handler_func = globals().get(handler_name)

        if callable(handler_func):
            handler_func()
        else:
            st.error(f"ハンドラ '{handler_name}' が定義されていません。")
