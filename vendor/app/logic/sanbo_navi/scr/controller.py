# --- ライブラリ読み込み ---
import streamlit as st
from langchain_community.vectorstores import FAISS

from logic.sanbo_navi.scr.view import load_pdf_first_page

from logic.sanbo_navi.scr.loader import (
    load_config,
    load_json_data,
    extract_categories_and_titles,
    load_question_templates,
)

# AI関連
from logic.sanbo_navi.scr.ai_loader import OpenAIConfig
from logic.sanbo_navi.scr.ai_loader import load_ai
from logic.sanbo_navi.scr.llm_utils import OpenAIClient
from logic.sanbo_navi.scr.llm_utils import suggest_category, generate_answer
from logic.sanbo_navi.scr.utils import load_vectorstore

# css
from components.custom_button import centered_button
from logic.sanbo_navi.scr.view import render_pdf_pages


# --- メイン処理 ---
def contoroller_education_gpt_page():
    # =============================
    # 各種設定・データの読み込み
    # =============================
    FAISS_PATH, PDF_PATH, JSON_PATH = load_config()
    json_data = load_json_data(JSON_PATH)
    categories, _ = extract_categories_and_titles(json_data)
    templates = load_question_templates()

    # =============================
    # AI クライアント設定
    # =============================
    client = load_ai(OpenAIConfig)
    llm_client = OpenAIClient(client)  # LLMClientBase 継承インスタンスを作成
    vectorstore = load_vectorstore(api_key=client.api_key, FAISS_PATH=FAISS_PATH)

    # =============================
    # UI 構築
    # =============================
    st.title("📘 教育GPTアシスタント")
    st.markdown(
        "SOLVESTについて質問できます。まず最初に知りたいことを一言で入力してください。"
    )

    # --- PDF プレビュー ---
    with st.expander("📄 PDFプレビュー（1ページ目のみ先に表示）"):
        pdf_first_page = load_pdf_first_page(PDF_PATH)
        st.image(pdf_first_page[0], caption="Page 1", use_container_width=True)

    # --- ユーザーからの質問入力 ---
    query_input = st.text_input(
        "知りたいこと（例：六面梱包機の処理能力、許認可の取得方法 など）"
    )

    # =============================
    # カテゴリ自動提案ボタン
    # =============================
    suggested_category = None
    if centered_button("➡️ カテゴリを自動提案"):
        if query_input.strip():
            with st.spinner("🤖 カテゴリを提案中..."):
                suggested_category = suggest_category(query_input, llm_client)
                st.session_state.suggested_category = suggested_category
        else:
            st.warning("まず知りたいことを入力してください。")

    # =============================
    # カテゴリ選択 UI
    # =============================
    default_category = st.session_state.get("suggested_category", None)
    main_category = st.selectbox(
        "カテゴリを選択",
        options=categories,
        index=(
            categories.index(default_category) if default_category in categories else 0
        ),
    )

    # =============================
    # サブカテゴリ選択 or 自由入力
    # =============================
    category_template = templates.get(main_category, templates["default"])
    subcategory_options = ["自由入力"] + category_template
    sub_category = st.selectbox("対象を選択", options=subcategory_options)

    # --- 質問文の組み立て ---
    if sub_category == "自由入力":
        user_input = st.text_area("さらに詳しく質問を入力", height=100)
        query = user_input if user_input.strip() else query_input
    else:
        query = (
            f"{query_input} に関して、{sub_category} を詳しく教えてください。"
            if query_input.strip()
            else f"{sub_category} に関して、詳しく教えてください。"
        )

    # =============================
    # 回答生成ボタン
    # =============================
    if centered_button("➡️ 送信") and query.strip():
        with st.spinner("🤖 回答生成中..."):
            answer, sources = generate_answer(
                query, main_category, vectorstore, llm_client
            )
            st.session_state.last_response = answer
            st.session_state.sources = sources

    # =============================
    # 回答表示 + 出典表示
    # =============================
    if "last_response" in st.session_state:
        st.success("✅ 回答")
        st.markdown(st.session_state.last_response)

        if "sources" in st.session_state:
            if "cache_pdf_pages" not in st.session_state:
                st.session_state.cache_pdf_pages = {}

            # --- ページの抽出（整数 or "3-5" などの文字列を許容） ---
            pages = set()
            for _, p in st.session_state.sources:
                if p is None:
                    continue
                p_str = str(p).strip()
                if p_str.isdigit() or "-" in p_str:
                    pages.add(p_str)

            # 表示用の文字列
            st.markdown(
                "📄 **出典ページ:** " + ", ".join([f"Page {p}" for p in sorted(pages)])
            )

            # 複数ページ対応関数に渡す
            render_pdf_pages(PDF_PATH, pages)


if __name__ == "__main__":
    contoroller_education_gpt_page()
