import streamlit as st


def show_update_log():
    st.subheader("🕘 最近の更新")
    st.markdown(
        """
    - `2025-04-04`：トップページの表示機能を追加
    - `2025-04-03`：CSVアップロード機能にテンプレートごとの出力対応を実装
    - `2025-04-01`：ログイン画面とパスワード認証を導入
    """
    )
