import streamlit as st


def inject_sidebar_css():
    css = """
    <style>
    div[data-baseweb="radio"] label {
        display: block;
        margin-bottom: 1.2em;
        font-weight: normal;
        font-size: 1rem;
    }
    </style>
    """
    st.sidebar.markdown(css, unsafe_allow_html=True)
