import pandas as pd
from logic.models.csv_processor import process_csv_by_date, check_date_alignment
from components.ui_message import (
    show_warning,
    show_error,
    show_date_mismatch,
)
from utils.file_loader import load_uploaded_csv_files
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import get_expected_dtypes_by_template
from utils.logger import app_logger


def apply_expected_dtypes(
    dfs: dict[str, pd.DataFrame],
    template_key: str,
) -> dict[str, pd.DataFrame]:
    """
    アップロードされた各CSVに対して、テンプレート定義に基づきデータ型を強制適用する。

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        読み込んだCSVファイル群
    template_key : str
        テンプレート名（例: average_sheet）

    Returns
    -------
    dict[str, pd.DataFrame]
        型強制後のDataFrame群
    """
    logger = app_logger()
    expected_dtypes_per_file = get_expected_dtypes_by_template(template_key)

    for key in dfs:
        dfs[key] = strip_whitespace(dfs[key])  # 🔽 空白除去

        dtypes = expected_dtypes_per_file.get(key)
        if dtypes:
            dfs[key] = enforce_dtypes(dfs[key], dtypes)
            logger.info(f"✅ 型を適用しました: {key}")

    return dfs


def prepare_csv_data(
    uploaded_files: dict, date_columns: dict, template_key: str
) -> dict:
    """
    アップロードされたCSVファイルを読み込み、型を適用し、
    日付の整合性をチェックしたうえで、処理対象のDataFrame群を返す。

    Parameters:
        uploaded_files (dict): {key: csvファイルパス} の辞書
        date_columns (dict): {key: 日付列名} の辞書
        template_key (str): テンプレート識別子（型適用などに使用）

    Returns:
        tuple[dict[str, DataFrame], list[str]]: 各キーに対応する整形済みDataFrame群と、共通日付リスト
        ※ 整合性エラー時は空のdictを返す
    """
    logger = app_logger()
    logger.info("📄 これからCSVの書類を作成します...")

    # --- ① アップロードされたCSVファイルを読み込む（DataFrame化） ---
    dfs = load_uploaded_csv_files(uploaded_files)

    # --- ② テンプレートに応じた型変換を実施（文字列→数値・日付など） ---
    dfs = apply_expected_dtypes(dfs, template_key)

    logger.info("📄 CSVの日付を確認中です...")

    # --- ③ 各CSVにおいて、日付列の存在と内容を確認 ---
    for key, df in dfs.items():
        # テンプレートで定義された日付列名を取得
        date_col = date_columns.get(key)

        # 日付列定義が存在しない場合は警告を表示して中断
        if not date_col:
            show_warning(f"⚠️ {key} の日付カラム定義が存在しません。")
            return {}

        # 実際のCSVに日付列が存在しない場合も中断
        if date_col not in df.columns:
            show_warning(f"⚠️ {key} のCSVに「{date_col}」列が見つかりませんでした。")
            return {}

        # 日付列を基準とした処理（並び替え・フィルタなど）を実施
        dfs[key] = process_csv_by_date(df, date_col)

    # --- ④ 各CSV間で日付の整合性（共通の日付があるか）をチェック ---
    result = check_date_alignment(dfs, date_columns)
    if not result["status"]:
        # 整合性NG時：エラー表示と詳細内容の提示
        show_error(result["error"])
        if "details" in result:
            show_date_mismatch(result["details"])
        return {}

    logger.info(f"✅ すべてのCSVで日付が一致しています：{result['dates']}")

    # --- ⑤ 正常終了：整形済みDataFrameと日付一覧を返す ---
    return dfs, result["dates"]
