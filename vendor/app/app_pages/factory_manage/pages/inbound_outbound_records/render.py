import streamlit as st
from components.custom_button import centered_button
from utils.config_loader import load_factory_required_files
from utils.check_uploaded_csv import (
    render_csv_upload_section,
    check_single_file_uploaded,
)
from logic.manage.utils.upload_handler import handle_uploaded_files


def render_inbound_outbound_records():
    # CSVの読込
    csv_controller()


def csv_controller():
    """
    Streamlit上でCSVファイルのアップロードと整合性チェック、
    加工・保存処理を行うコントローラー。
    """
    selected_template = "inbound_outbound_records"

    # --- 必要なファイルキーを設定ファイルから取得 ---
    required_keys = load_factory_required_files()[selected_template]

    # --- アップロードUI表示（テンプレートに応じたファイル形式） ---
    csv_file_type = "receive"
    render_csv_upload_section(csv_file_type)

    csv_file_type = "shipping"
    render_csv_upload_section(csv_file_type)

    # --- ファイルアップロードの整合性チェック ---
    uploaded_files = handle_uploaded_files(required_keys)
    uploaded_file = uploaded_files.get(csv_file_type)
    all_uploaded, missing_key = check_single_file_uploaded(uploaded_file, csv_file_type)
    print(all_uploaded, missing_key)

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        if centered_button("⏩ CSVをアップロード"):
            # # --- CSVファイル読み込み・整形 ---
            # df = pd.read_csv(uploaded_file)
            # df = make_csv(df)

            # # --- SQLite DBに保存 ---
            # make_sql_db(df)

            # # --- 完了通知とUIリセット ---
            # st.success("📥 CSVファイルの読み込みと保存が完了しました。")
            # st.toast("CSV処理が正常に完了しました", icon="📁")

            # # --- セッションからアップロード情報を削除し再描画 ---
            # key_to_clear = f"uploaded_{csv_file_type}"
            # if key_to_clear in st.session_state:
            #     del st.session_state[key_to_clear]

            st.rerun()

    pass
