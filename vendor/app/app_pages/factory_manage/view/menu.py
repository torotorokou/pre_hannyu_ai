import streamlit as st


def render_sidebar(menu_options: list[dict]) -> str:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏗 工場管理メニュー")
    selected = st.sidebar.radio(
        "表示する項目を選択してください",
        menu_options,
        key="factory_menu",
    )
    return selected
