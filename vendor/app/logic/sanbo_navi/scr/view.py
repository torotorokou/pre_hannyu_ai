import streamlit as st
from pdf2image import convert_from_path


# --- PDF画像の読み込み ---
@st.cache_resource
def load_pdf_first_page(path, dpi=100):
    # PDFの1ページ目を画像として読み込む
    return convert_from_path(path, dpi=dpi, first_page=1, last_page=1)


@st.cache_resource
def load_pdf_page(path, page_number, dpi=100):
    # 指定ページのPDFを画像として読み込む
    return convert_from_path(
        path, dpi=dpi, first_page=page_number, last_page=page_number
    )[0]


# --- PDFページ表示 ---
def render_pdf_pages(PDF_PATH, pages):
    """
    指定されたページ番号または範囲をもとに PDF ページを表示。
    pages: list[str or int] → 例: [3, "5-7", 9]
    """
    if "cache_pdf_pages" not in st.session_state:
        st.session_state.cache_pdf_pages = {}

    # 整形されたページ番号リスト
    page_numbers = []

    for p in pages:
        if isinstance(p, str) and "-" in p:
            try:
                start_page, end_page = map(int, p.split("-"))
                page_numbers.extend(range(start_page, end_page + 1))
            except Exception as e:
                st.warning(f"ページ範囲の解析エラー: {p}")
                continue
        else:
            try:
                page_numbers.append(int(p))
            except Exception as e:
                st.warning(f"ページ番号の解析エラー: {p}")
                continue

    with st.expander("📘 出典ページのプレビュー"):
        for p in sorted(set(page_numbers)):
            if p >= 1:
                if p in st.session_state.cache_pdf_pages:
                    st.image(
                        st.session_state.cache_pdf_pages[p],
                        caption=f"Page {p} (cached)",
                        use_container_width=True,
                    )
                else:
                    with st.spinner(f"📄 Page {p} 読み込み中..."):
                        page_image = load_pdf_page(PDF_PATH, p)
                        st.session_state.cache_pdf_pages[p] = page_image
                        st.image(
                            page_image,
                            caption=f"Page {p}",
                            use_container_width=True,
                        )
