import re
import pandas as pd


def shift_cell_row(cell: str, offset: int) -> str:
    match = re.match(r"([A-Z]+)(\d+)", cell)
    if match:
        col, row = match.groups()
        new_row = max(1, int(row) + offset)
        return f"{col}{new_row}"
    return cell


def create_label_rows_generic(
    df: pd.DataFrame,
    label_source_col: str,
    offset: int = -1,
    cell_column: str = "セル",
    value_column: str = "値",
) -> pd.DataFrame:
    """
    任意の列の値をラベルとして使用し、セル位置を指定行だけずらしてラベル行を生成する。

    Parameters:
        df : pd.DataFrame
            元のテンプレートDataFrame
        cell_column : str
            セル位置を記載した列名（例: "セル"）
        label_source_col : str
            ラベルとして "値" に転記する元の列名（例: "業者名", "有価名"）
        offset : int
            セルの行をずらす量（例: -1で上に）
        value_column : str
            ラベルを書き込む対象列（デフォルト: "値"）

    Returns:
        pd.DataFrame
            ラベル行だけを含むDataFrame（他の列はすべて空欄）
    """
    df_label = df.copy()

    # セルをオフセット行だけずらす
    df_label[cell_column] = df_label[cell_column].apply(
        lambda x: shift_cell_row(x, offset)
    )

    # ラベル値を値列にコピー
    df_label[value_column] = df[label_source_col]

    # 値・セル以外の列は空にする
    for col in df_label.columns:
        if col not in [cell_column, value_column]:
            df_label[col] = ""

    return df_label


def sort_by_cell_row(df: pd.DataFrame, cell_col: str = "セル") -> pd.DataFrame:
    """
    セル番地列の「行番号（数字部分）」をもとに DataFrame をソートする。
    非文字列や欠損値があってもエラーを出さずに処理する。

    Parameters
    ----------
    df : pd.DataFrame
        並び替え対象のデータフレーム
    cell_col : str
        セル番地が格納されている列名（デフォルトは "セル"）

    Returns
    -------
    pd.DataFrame
        行番号で昇順にソートされたDataFrame（インデックスはリセットされる）
    """
    df = df.copy()

    def extract_row_number(x):
        if isinstance(x, str):
            match = re.findall(r"\d+", x)
            return int(match[0]) if match else None
        return None

    df["_セル行"] = df[cell_col].apply(extract_row_number)
    df = (
        df.sort_values("_セル行", na_position="last")
        .drop(columns="_セル行")
        .reset_index(drop=True)
    )
    return df


def add_label_rows(
    master_csv: pd.DataFrame,
    label_source_col: str = "業者名",
    offset: int = -1,
) -> pd.DataFrame:
    """
    マスターCSVにラベル行（業者名など）を追加する。

    Parameters:
        master_csv : pd.DataFrame
        label_source_col : ラベル元の列（例："業者名"）
        offset : ラベルを入れる位置（-1 で1行上など）

    Returns:
        pd.DataFrame : ラベル行を追加し、セル順にソートしたDataFrame
    """
    df_label = create_label_rows_generic(master_csv, label_source_col, offset=offset)
    df_extended = pd.concat([master_csv, df_label], ignore_index=True)
    df_extended = sort_by_cell_row(df_extended)
    return df_extended


def add_label_rows_and_restore_sum(
    df: pd.DataFrame, label_col: str, offset: int = -1, sum_keyword: str = "合計"
) -> pd.DataFrame:
    """
    指定列でラベル行を追加し、"合計"行を除外→ラベル付加→合計行を復元して整形。

    Parameters:
        df : pd.DataFrame
        label_col : ラベル元となる列名（例: "業者名", "有価名" など）
        offset : セルの挿入位置（例: -1 → 1行上にラベル）
        sum_keyword : "合計"として検出するキーワード

    Returns:
        pd.DataFrame
    """
    sum_rows = df[df[label_col].str.contains(sum_keyword, na=False)]
    filtered = df[~df[label_col].str.contains(sum_keyword, na=False)]
    labeled = add_label_rows(filtered, label_source_col=label_col, offset=offset)
    combined = pd.concat([labeled, sum_rows], ignore_index=True)
    return sort_by_cell_row(combined)
