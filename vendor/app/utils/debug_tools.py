"""
デバッグ用ユーティリティ関数
"""

import os
import pandas as pd
from typing import Dict
from config.loader.main_path import MainPath


def save_debug_csvs(dfs: Dict[str, pd.DataFrame], folder: str = "debug_data") -> None:
    os.makedirs(folder, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(
            os.path.join(folder, f"debug_{name}.csv"), index=False, encoding="utf-8-sig"
        )


def check_dfs(dfs: dict, rows: int = 5, show_columns: bool = True):
    for key, df in dfs.items():
        print(f"\n📘 {key} - {len(df)}件")
        if show_columns:
            print("🧾 カラム:", df.columns.tolist())
        print(df.head(rows))


def save_debug_parquets(
    dfs: Dict[str, pd.DataFrame],
) -> None:
    mainpath = MainPath()
    folder = mainpath.get_path("input", section="directories")
    print("📁 フォルダパス:", folder)
    os.makedirs(folder, exist_ok=True)
    for name, df in dfs.items():
        file_path = os.path.join(folder, f"debug_{name}.parquet")
        df.to_parquet(file_path, index=False)


def visualize_df(df: pd.DataFrame, method: str = "hist", **kwargs):
    """
    DataFrame を開発時に可視化するための汎用ツール。
    """
    print(f"📊 visualize_df - method={method}")

    # ← 関数の中で必要なときだけimportするよう変更
    if method in ["hist", "plot"]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib がインストールされていません")
            return

    if method == "head":
        print(df.head(kwargs.get("n", 5)))
    elif method == "info":
        df.info()
    elif method == "describe":
        print(df.describe())
    elif method == "hist":
        df.hist(figsize=(10, 6))
        plt.tight_layout()
        plt.show()
    elif method == "plot":
        df.plot(**kwargs)
        plt.tight_layout()
        plt.show()
    else:
        print(f"❓ 不明なmethod: {method}")


def show_with_gui(df, name="DataFrame"):
    """
    pandasgui を使って DataFrame をGUI表示（開発環境専用）
    """
    from config.settings.config import IS_DEV

    if IS_DEV:
        try:
            from pandasgui import show

            print(f"🖥 pandasgui で [{name}] を表示します")
            show(df)
        except ImportError:
            print("⚠️ pandasgui がインストールされていません（GUI表示スキップ）")


from config.settings.config import IS_DEV


class DevTools:
    def __init__(self, enabled: bool = IS_DEV):
        self.enabled = enabled

    def display(self, df, name="DataFrame", rows=20, cols=10):
        if not self.enabled:
            return

        print(f"\n🛠 [{name}] の表示（先頭 {rows} 行, 最大 {cols} 列）")
        try:
            print(df.to_string(max_rows=rows, max_cols=cols))
        except Exception as e:
            print(f"⚠️ 表示中にエラー発生: {e}")

    def visualize_df(self, df):
        if self.enabled:
            try:
                from utils.debug_tools import visualize_df

                visualize_df(df)
            except ImportError:
                print("⚠️ visualize_df 未使用（未インポート）")

    def show_gui(self, df, name="DataFrame"):
        if self.enabled:
            try:
                from pandasgui import show

                print(f"🖥 [{name}] をGUI表示します")
                show(df)
            except ImportError:
                print("⚠️ pandasgui が未インストール")


import time
from config.settings.loader import load_settings
from utils.logger import app_logger


def debug_pause(message: str = "🛑 DEBUG PAUSE", seconds: int = 3):
    """開発環境でのみ一時停止するユーティリティ"""
    settings = load_settings()
    if settings.get("ENV_NAME") in ["dev", "development"]:
        logger = app_logger()
        logger.info(f"{message} ({seconds}秒停止)")
        time.sleep(seconds)
