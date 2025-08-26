import streamlit as st


def show_manual_links():
    st.subheader("📄 マニュアル・操作説明")
    st.markdown(
        """
    - 👉 [操作マニュアルを開く（PDF）](https://example.com/manual.pdf)
    - 👉 [ファイル形式のガイド](https://example.com/template-guide)
    - 👉 [サポートに問い合わせ](mailto:support@example.com)
    """
    )

    st.info(
        "⚠️ このアプリは社内専用です。個人情報や機密情報のアップロードは避けてください。"
    )
