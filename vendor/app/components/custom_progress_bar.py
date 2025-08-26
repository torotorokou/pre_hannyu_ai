import streamlit as st


class CustomProgressBar:
    def __init__(
        self, total_steps: int = 3, labels: list[str] = None, bar_color: str = "#4f8df7"
    ):
        self.total_steps = total_steps
        self.bar_color = bar_color
        self.labels = labels or ["📥 データ読み込み", "🧮 処理中", "📄 完了"]

    def render(self, current_step: int):
        percentage = int((current_step + 1) / self.total_steps * 100)
        label = self.labels[min(current_step, len(self.labels) - 1)]

        st.markdown(
            f"""
            <style>
            .cpb-wrapper {{
                margin: 1em 0;
                font-family: "Helvetica Neue", sans-serif;
            }}
            .cpb-label {{
                font-size: 0.9rem;
                color: #4b5563;
                margin-bottom: 4px;
            }}
            .cpb-bar {{
                background-color: #e5e7eb;
                border-radius: 8px;
                height: 12px;
                width: 100%;
                overflow: hidden;
            }}
            .cpb-fill {{
                background: linear-gradient(90deg, {self.bar_color}, #93c5fd);
                height: 100%;
                width: {percentage}%;
                border-radius: 8px;
                transition: width 1.0s ease-in-out;
            }}
            </style>

            <div class="cpb-wrapper">
                <div class="cpb-label">{label}</div>
                <div class="cpb-bar">
                    <div class="cpb-fill"></div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
