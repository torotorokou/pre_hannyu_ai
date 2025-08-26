from streamlit import markdown


class GlobalStyle:
    def __init__(self):
        self.base = BaseStyle()
        self.button = ButtonStyle()
        self.input = InputStyle()
        self.sidebar = SidebarStyle()
        self.table = TableStyle()

    def apply_all(self):
        self.base.apply()
        self.button.apply()
        self.input.apply()
        self.sidebar.apply()
        self.table.apply()


# --- base_style.py ---
class BaseStyle:
    def apply(self):
        markdown(
            """
            <style>
            section.main > div:first-child {
                padding-top: 0rem;
                margin-top: 0rem;
            }
            h1, h2, h3 {
                color: #1e3a8a;
                font-family: 'Segoe UI', sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# --- button_style.py ---
class ButtonStyle:
    def apply(self):
        markdown(
            """
            <style>
            .stButton>button {
                background: linear-gradient(to right, #3b82f6, #60a5fa);
                color: white;
                border: none;
                padding: 0.6em 1.5em;
                border-radius: 6px;
                font-weight: 600;
                font-size: 16px;
                transition: all 0.3s ease-in-out;
            }
            .stButton>button:hover {
                background: linear-gradient(to right, #1e40af, #2563eb);
                color: white;
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
                transform: translateY(-1px);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# --- input_style.py ---
class InputStyle:
    def apply(self):
        markdown(
            """
            <style>
            input, textarea {
                background-color: #ffffff;
                border: 1px solid #93c5fd;
                border-radius: 6px;
                padding: 0.5em;
                color: #1e293b;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# --- sidebar_style.py ---
class SidebarStyle:
    def apply(self):
        markdown(
            """
            <style>
            .css-1d391kg {
                background-color: #e0f2ff;
                border-right: 1px solid #b6e0fe;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# --- table_style.py ---
class TableStyle:
    def apply(self):
        markdown(
            """
            <style>
            .stDataFrame {
                background-color: #f0f9ff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
