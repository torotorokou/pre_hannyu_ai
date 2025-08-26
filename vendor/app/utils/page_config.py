from dataclasses import dataclass


@dataclass
class PageConfig:
    page_id: str
    title: str = ""
    parent_title: str = ""
    show_parent_title: bool = False  # ← これを追加
