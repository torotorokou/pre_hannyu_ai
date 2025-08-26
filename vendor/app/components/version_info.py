import streamlit as st


def show_version_info():
    st.subheader("🔖 バージョン・サポート情報")
    st.markdown(
        """
    - **バージョン**：`v1.0.1`（最終更新：`2025-04-15`）
    - **開発・管理**：黒木・土田
    - **お問い合わせ**：[support@example.com](mailto:support@example.com)
    """
    )
