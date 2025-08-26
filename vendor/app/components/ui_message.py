import streamlit as st


def show_warning_bubble(expected_name, detected_name):
    st.markdown(
        f"""
    <div style="
        background-color: #fffbea;
        border-left: 6px solid #e6b800;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        font-size: 14px;
        line-height: 1.6;
        color: #333;
        position: relative;
    ">
        <strong>⚠️ ファイル形式の確認</strong><br>
        アップロードされたCSVは「<strong>{expected_name}</strong>」ではなく「<strong>{detected_name}</strong>」と判別されました。<br>
        正しいファイルをご確認のうえ再アップロードをお願いします。
        <div style="
            position: absolute;
            top: -12px;
            left: 20px;
            width: 0;
            height: 0;
            border-left: 12px solid transparent;
            border-right: 12px solid transparent;
            border-bottom: 12px solid #fffbea;
        "></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_success(message: str):
    st.success(message)


def show_warning(message: str):
    st.warning(message)


def show_error(message: str):
    st.error(message)


def show_date_mismatch(details: dict):
    for key, values in details.items():
        st.write(f"- `{key}`: {values}")
