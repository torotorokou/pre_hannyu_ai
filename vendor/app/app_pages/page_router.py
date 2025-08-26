import streamlit as st

# from config.page_config import page_dict, page_dict_reverse, page_labels
from utils.config_loader import get_page_dicts, get_app_config
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.update_log import show_update_log
from components.version_info import show_version_info
from utils.config_loader import get_page_config


# controller: route_page.py


def route_page():
    """
    Streamlitアプリのルーティング処理を行うメイン関数。

    - URLクエリパラメータとセッション状態を同期し、
    - サイドバーにページメニューを表示、
    - 選択されたページの中身を描画する。

    ページ構成情報（ID・ラベル）は YAML から読み込み、
    MVC構成のController的役割を担う。
    """
    # ページ構成情報を取得（ラベルとURL ID）
    page_dict, page_dict_reverse, page_labels = get_page_dicts()

    # URLパラメータとセッションの同期
    _handle_query_params(page_dict, page_dict_reverse)

    # サイドバーに選択メニュー表示
    _render_sidebar(page_labels)

    # 選択されたページの中身を描画
    _render_selected_page()


def _handle_query_params(page_dict, page_dict_reverse):
    params = st.query_params
    page_id = params.get("page", "home")
    default_label = page_dict_reverse.get(page_id, "トップページ")

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = default_label

    st.query_params["page"] = page_dict[st.session_state.selected_page]


def _render_sidebar(page_labels):
    st.sidebar.selectbox("📂 機能を選択", page_labels, key="selected_page")


from app_pages.page_registry import TOPPAGE_INSTANCES


def _render_selected_page():
    title = get_app_config()["title"]
    selected_label = st.session_state.selected_page
    pages = get_page_config()

    for page in pages:
        if page["label"] == selected_label:
            if page["id"] == "home":
                st.title(title)
            if "message" in page:
                st.info(page["message"])

            elif "function" in page:
                func_name = page["function"]
                page_instance = TOPPAGE_INSTANCES.get(func_name)
                if page_instance:
                    page_instance.render()  # ← クラスの render を呼ぶ
                else:
                    st.warning(f"⚠️ `{func_name}` は存在しません。")

            # トップページだけ追加表示
            if page.get("addons") is True:
                _render_sidebar_addons()
            break


def _render_sidebar_addons():
    with st.sidebar:
        st.markdown("---")
        show_notice()
        show_manual_links()
        show_update_log()
        show_version_info()
