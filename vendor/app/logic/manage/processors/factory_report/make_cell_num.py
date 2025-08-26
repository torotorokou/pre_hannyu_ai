import re
import pandas as pd


def make_cell_num(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    各カテゴリに対して開始セルを指定し、そこからセルを割り当てて返す。
    横方向に展開し、列数を超えたら次の行に折り返す。
    """
    # カテゴリごとの開始セルを定義
    start_cells = {"有価": "C15", "処分": "C23", "ヤード": "C33"}

    # 横展開に使用する列の順番を定義（必要に応じて拡張可能）
    col_list = ["C", "F", "I", "L", "O"]

    # 実際のセル割り当て処理を実行
    return make_cell_num_cal(master_csv, start_cells=start_cells, col_list=col_list)


def make_cell_num_cal(
    master_csv: pd.DataFrame,
    start_cells: dict[str, str],
    col_list: list[str] = None,
    category_col: str = "カテゴリ",
) -> pd.DataFrame:
    """
    カテゴリごとに指定した開始セルから、横方向にセルを割り当てる。
    列が足りなくなった場合は、指定した行ステップで下に折り返す。

    Parameters:
    - master_csv: セル割り当て対象のデータフレーム
    - start_cells: 各カテゴリの開始セル（例: {"有価": "C14"}）
    - col_list: セルの列指定リスト（例: ["C", "F", "I", ...]）
    - category_col: カテゴリを示す列名（デフォルトは "カテゴリ"）

    Returns:
    - セル列（"セル"）を追加したデータフレーム
    """

    if col_list is None:
        col_list = ["C", "F", "I", "L", "O"]

    # セル割り当て対象の行を抽出（ロックあり または 値 ≠ 0）
    df_target = (
        master_csv[
            (master_csv["セルロック"] == 1)
            | ((master_csv["セルロック"] != 1) & (master_csv["値"] != 0))
        ]
        .copy()
        .sort_values("順番")
    )

    # 新しい "セル" 列を追加（初期化）
    df_target["セル"] = ""

    # カテゴリごとに処理
    for category, group in df_target.groupby(category_col):
        # 指定された開始セルを取得（なければエラー）
        start_cell = start_cells.get(category)
        if not start_cell:
            raise ValueError(
                f"[カテゴリ: {category}] に対する開始セルが指定されていません。"
            )

        # セル文字列（例: "C14"）から列と行を抽出
        match = re.match(r"([A-Z]+)(\d+)", start_cell)
        if not match:
            raise ValueError(
                f"[カテゴリ: {category}] の開始セル '{start_cell}' は 'C14' の形式である必要があります。"
            )

        base_row = int(match[2])  # 開始行番号（例: 14）
        row_step = 2  # 折り返し時の行の増分

        # グループ内の各行に対してセルを割り当て
        for i, df_idx in enumerate(group.index):
            col = col_list[i % len(col_list)]  # 列はcol_list内で循環
            row = (
                base_row + (i // len(col_list)) * row_step
            )  # 行は列数を超えたら折り返す
            df_target.at[df_idx, "セル"] = f"{col}{row}"  # セル文字列を生成して代入

    return df_target
