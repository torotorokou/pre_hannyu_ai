import streamlit as st
import time
import random
from app_pages.base_page import BasePage
from utils.page_config import PageConfig


class HomePage(BasePage):
    def __init__(self):
        config = PageConfig(
            page_id="home",
            title="参謀くん Chat Guide",
            parent_title="ホームガイド",
        )
        super().__init__(config)

        self.chat = [
            "こんにちは！私は **WEB版 参謀くん** です。\n\n"
            "このツールは、**現場業務の効率化**と**帳票作成の自動化**をサポートします。\n\n"
            "ご利用の際は、👈 左側の **サイドバー** または下記ボタンを選んでください。"
        ]

        self.menu_options = [
            {"label": "📝 工場日報の作成", "page_key": "factory_report"},
            {"label": "📦 搬出入収支表の集計", "page_key": "shipping_balance"},
            {"label": "📋 管理票の自動生成", "page_key": "management_sheet"},
        ]

    def typewriter_chat(self, message: str, delay=0.02):
        placeholder = st.empty()
        displayed = ""
        for char in message:
            displayed += char
            placeholder.markdown(displayed)
            time.sleep(delay)

    def render_menu_buttons(self):
        st.markdown("---")
        st.markdown("### 🚀 メニューを選択してください(未完成)")

        menu_groups = {
            "📂 管理業務": [
                {"label": "📝 工場日報の作成", "page_key": "factory_report"},
                {"label": "📋 管理票の自動生成", "page_key": "management_sheet"},
            ],
            "🏭 工場管理": [
                {"label": "📦 搬出入収支表の集計", "page_key": "shipping_balance"},
                {"label": "📈 月間推移グラフ", "page_key": "monthly_graph"},
            ],
            "🧰 便利ツール": [
                {"label": "🧮 重量計算ツール", "page_key": "weight_calc"},
                {"label": "📆 日付変換ツール", "page_key": "date_converter"},
            ],
            "🧾 その他": [
                {"label": "📚 操作マニュアル", "page_key": "manual"},
                {"label": "📨 サポート連絡", "page_key": "support"},
            ],
        }

        # 2列表示のためのレイアウト
        col1, col2 = st.columns(2)
        columns = [col1, col2]
        col_index = 0

        for category, options in menu_groups.items():
            with columns[col_index]:
                st.markdown(f"#### {category}")
                for option in options:
                    if st.button(
                        option["label"],
                        key=option["page_key"],
                        use_container_width=True,
                    ):
                        st.session_state.page = option["page_key"]
            col_index = (col_index + 1) % 2  # 切り替え（0 → 1 → 0 → ...）

    def render(self):
        self.render_title()
        # self.render_sidebar_info()
        self.log("トップページを表示しました")

        if "top_page_viewed" not in st.session_state:
            st.session_state.top_page_viewed = False

        if not st.session_state.top_page_viewed:
            for msg in self.chat:
                with st.chat_message("assistant"):
                    self.typewriter_chat(msg)
                time.sleep(random.uniform(0.2, 0.3))
            st.session_state.top_page_viewed = True
        else:
            for msg in self.chat:
                with st.chat_message("assistant"):
                    st.markdown(msg)

        self.render_menu_buttons()
