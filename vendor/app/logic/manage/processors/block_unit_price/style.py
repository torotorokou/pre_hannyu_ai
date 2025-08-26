def _get_transport_selection_styles() -> str:
    """運搬業者選択画面のスタイルを定義する

    Returns:
        str: CSSスタイル定義
    """
    return """
    /* ヘッダースタイル */
    h3 {
        border: none !important;
        margin-bottom: 0.5rem !important;
    }

    /* セレクトボックスのスタイル（ダークモード対応） */
    div[data-baseweb="select"] > div {
        border-width: 1.5px !important;
        border-color: #999999 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    div[data-baseweb="select"]:focus-within {
        box-shadow: 0 0 0 2px #cbd5e1 !important;
    }

    div[data-baseweb="select"] span {
        color: #f1f5f9 !important;
        font-weight: 600;
    }

    /* ラベルのスタイル（明暗両対応） */
    label[data-testid="stWidgetLabel"] {
        color: #e5e7eb !important;
        font-size: 14px;
    }
    """


def _get_vendor_card_styles() -> dict:
    """業者情報カードのスタイルを定義する

    Returns:
        dict: スタイル定義の辞書
    """
    return {
        "card_container": """
            background-color:#1e293b;
            padding:1px 4px;
            margin-bottom:6px;
            border-radius:2px;
            border:0.3px solid #3b4252;
        """,
        "info_container": "padding-right:10px;",
        "vendor_name": """
            font-size:18px;
            font-weight:600;
            color:#38bdf8;
        """,
        "item_name": """
            font-size:15px;
            color:inherit;
            margin-top: 2px;
        """,
        "detail": """
            font-size:14.5px;
            color:inherit;
            margin-top: 2px;
        """,
    }
