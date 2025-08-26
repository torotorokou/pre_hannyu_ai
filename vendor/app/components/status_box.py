import streamlit as st


def render_status_box(message, bg_rgba="rgba(255, 255, 255, 0.03)", text_color="#aaa"):
    st.markdown(
        f"""
        <div style="
            margin-top: -0.3em;
            margin-bottom: 0.8em;
            padding: 0.3em 0.8em;
            background-color: {bg_rgba};
            border-radius: 3px;
            font-weight: 400;
            font-size: 13px;
            color: {text_color};
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
