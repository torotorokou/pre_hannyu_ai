import streamlit as st


def show_notice():
    st.subheader("🛎️ お知らせ")
    st.warning(
        "📢 2025年4月より、CSVテンプレートが新仕様になりました。旧ファイルとの互換性にご注意ください。"
    )
