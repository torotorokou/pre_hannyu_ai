import streamlit as st
import pandas as pd

st.set_page_config(page_title="CSVアップロードテスト", layout="centered")
st.title("📂 CSVアップロード確認")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSVの読み込みに成功しました！")
        st.write("📊 データプレビュー:")
        st.dataframe(df.head())
    except Exception as e:
        st.error("❌ 読み込みエラーが発生しました")
        st.exception(e)
else:
    st.info("⏳ CSVファイルを選択してください")
