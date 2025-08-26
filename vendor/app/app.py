import streamlit as st
from app_pages.page_router import route_page
from components.ui_style import GlobalStyle
from utils.config_loader import get_app_config
from config.settings.loader import load_settings
from utils.logger import app_logger


# 総ページ設定
title = get_app_config()
st.set_page_config(page_title=title["title"], layout="centered")

# グローバルCSS
GlobalStyle().apply_all()

# ルーティング制御（URLとsession_state）
# 以下に個々の作業を記す
route_page()


# 開発環境設定
settings = load_settings()

if settings.get("ENV_NAME") in ["dev", "development"]:
    st.write(f"🛠 現在の環境: {settings['ENV_NAME']}")
    st.write(f"🔌 ポート番号: {settings['STREAMLIT_SERVER_PORT']}")
    st.write(f"🐞 デバッグモード: {settings['DEBUG']}")
