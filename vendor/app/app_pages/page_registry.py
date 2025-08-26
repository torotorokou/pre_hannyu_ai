# =====================================================
# TOPPAGE_INSTANCES
# - Streamlitアプリのトップページ切り替え用
# - YAML内 function: xxxx に対応させてルーティングする
# - ここでインスタンス化したものが main から呼ばれる
# =====================================================

from app_pages.home.home import HomePage
from app_pages.util_pages.utilpage import UtilPage
from app_pages.manage.page import ManageWorkPage
from app_pages.factory_manage.page import FactoryManageWorkPage
from app_pages.sanbo_navi.page import SanboNaviPage

TOPPAGE_INSTANCES = {
    # --- ホームページ ---
    "render_home": HomePage(),
    # --- ユーティリティページ ---
    "render_util_page": UtilPage(),
    # --- 管理業務ページ ---
    "manage_work_controller": ManageWorkPage(),
    # --- 工場管理ページ ---
    "factory_manage_work_controller": FactoryManageWorkPage(),
    # --- Sanboナビページ ---
    "sanbo_navi": SanboNaviPage(),
}
