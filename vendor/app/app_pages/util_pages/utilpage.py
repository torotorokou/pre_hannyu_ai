from app_pages.base_page import BasePage
from utils.page_config import PageConfig
import streamlit as st


class UtilPage(BasePage):
    def __init__(self):
        config = PageConfig(
            page_id="util", title="ユーティリティ機能", parent_title="補助ツール"
        )
        super().__init__(config)

    def render(self):
        self.render_title()
        st.write("各機能を実装予定です")
        st.write("やよい会計インポートなど")
        self.drag_drop()  # ← 修正① クラスメソッドとして呼び出し

    def drag_drop(self):
        import streamlit as st
        import pandas as pd
        import os

        # --- 設定 ---
        locations = ["第一工場", "第二工場", "ライン"]
        time_slots = [
            "7:00-16:00",
            "8:00-17:00",
            "9:00-18:00",
            "10:00-19:00",
            "14:00-23:00",
        ]
        candidates = [f"作業者{i+1}" for i in range(25)]  # 25人の例
        history_file = "shift_assignment_history.csv"

        # --- 履歴読み込み ---
        loaded_assignments = {}
        if os.path.exists(history_file):
            df_hist = pd.read_csv(history_file)
            for _, row in df_hist.iterrows():
                loc = row["場所"]
                time = row["時間帯"]
                person = row["担当者"]
                key = f"{loc}_{time}_vertical".replace(":", "").replace("-", "_")
                loaded_assignments.setdefault(key, []).append(person)

        st.title("📋 シフト割当（時間帯縦・場所固定）")

        # --- UI表示：場所ごとに縦方向で時間帯を展開 ---
        for loc in locations:
            st.markdown(f"## 📍 {loc}")
            for time in time_slots:
                key = f"{loc}_{time}_vertical".replace(":", "").replace("-", "_")

                # ✅ 初期化（履歴があれば使用）
                if key not in st.session_state:
                    st.session_state[key] = loaded_assignments.get(key, [])

                st.multiselect(
                    label=f"⏰ {time}",
                    options=candidates,
                    default=st.session_state[key],
                    key=key,
                )

        # --- 集計と保存 ---
        if st.button("📋 割当を集計"):
            data = []
            for loc in locations:
                for time in time_slots:
                    key = f"{loc}_{time}_vertical".replace(":", "").replace("-", "_")
                    for person in st.session_state.get(key, []):
                        data.append({"場所": loc, "時間帯": time, "担当者": person})

            df_result = pd.DataFrame(data)
            st.success("✅ 割当結果")
            st.dataframe(df_result, use_container_width=True)

            # ✅ CSVに保存
            df_result.to_csv(history_file, index=False)
