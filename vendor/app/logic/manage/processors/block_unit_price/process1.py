import streamlit as st
import pandas as pd
import re

from logic.manage.processors.block_unit_price.style import (
    _get_transport_selection_styles,
    _get_vendor_card_styles,
)


def _apply_transport_selection_styles() -> None:
    """運搬業者選択画面のスタイルを適用する"""
    st.markdown(
        f"<style>{_get_transport_selection_styles()}</style>",
        unsafe_allow_html=True,
    )


def _render_vendor_card(gyousha_name: str, hinmei: str, meisai: str) -> None:
    """業者情報カードを描画する

    Args:
        gyousha_name (str): 業者名
        hinmei (str): 品名
        meisai (str): 明細備考
    """
    styles = _get_vendor_card_styles()

    st.markdown(
        f"""
        <div style='{styles["card_container"]}'>
            <div style='{styles["info_container"]}'>
                <div style='{styles["vendor_name"]}'>
                    🗑️ {gyousha_name}
                </div>
                <div style='{styles["item_name"]}'>
                    品名：{hinmei}
                </div>
                <div style='{styles["detail"]}'>
                    明細備考：{meisai}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_transport_selection_form(
    df_after: pd.DataFrame, df_transport: pd.DataFrame
) -> pd.DataFrame:
    """運搬業者選択フォームを作成し、選択結果を処理する

    Args:
        df_after (pd.DataFrame): 処理対象の出荷データ
        df_transport (pd.DataFrame): 運搬業者マスターデータ

    Returns:
        pd.DataFrame: 運搬業者が選択された出荷データ
    """
    # セッション状態の初期化
    if "block_unit_price_confirmed" not in st.session_state:
        st.session_state.block_unit_price_confirmed = False
    if "block_unit_price_transport_map" not in st.session_state:
        st.session_state.block_unit_price_transport_map = {}

    # 運搬社数が1以外の行を抽出
    target_rows = df_after[df_after["運搬社数"] != 1].copy()

    # UI表示
    st.title("運搬業者の選択")
    _apply_transport_selection_styles()

    if not st.session_state.block_unit_price_confirmed:
        with st.form("transport_selection_form"):
            selected_map = {}

            for idx, row in target_rows.iterrows():
                # 業者情報の取得と整形
                gyousha_cd = row["業者CD"]
                gyousha_name = str(row.get("業者名", gyousha_cd))
                hinmei = str(row.get("品名", "")).strip() or "-"
                meisai = str(row.get("明細備考", "")).strip() or "-"
                gyousha_name_clean = re.sub(r"（\s*\d+\s*）", "", gyousha_name)

                # 運搬業者の選択肢を取得
                options = df_transport[df_transport["業者CD"] == gyousha_cd][
                    "運搬業者"
                ].tolist()
                if not options:
                    st.warning(
                        f"{gyousha_name_clean} に対応する運搬業者が見つかりません。"
                    )
                    continue

                # セレクトボックスの初期値を設定
                select_key = f"select_block_unit_price_row_{idx}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = options[0]

                # 2カラムレイアウト
                col1, col2 = st.columns([2, 3])

                # 左カラム：業者情報
                with col1:
                    _render_vendor_card(gyousha_name_clean, hinmei, meisai)

                # 右カラム：運搬業者選択
                with col2:
                    selected = st.selectbox(
                        label="🚚 運搬業者を選択してください",
                        options=options,
                        key=select_key,
                    )

                selected_map[idx] = selected

            # 確定ボタン
            submitted = st.form_submit_button("✅ 選択を確定して次へ進む")
            if submitted:
                if len(selected_map) < len(target_rows):
                    st.warning("未選択の行があります。すべての行を選択してください。")
                else:
                    st.session_state.block_unit_price_transport_map = selected_map
                    st.session_state.block_unit_price_confirmed = True

                    # 選択結果をデータフレームに反映
                    selected_df = pd.DataFrame.from_dict(
                        st.session_state.block_unit_price_transport_map,
                        orient="index",
                        columns=["運搬業者"],
                    )
                    selected_df.index.name = df_after.index.name
                    df_after = df_after.merge(
                        selected_df, how="left", left_index=True, right_index=True
                    )
                    st.success("✅ 選択が確定されました。")
                    return df_after

        st.stop()

    return df_after
