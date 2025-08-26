import pandas as pd
import numpy as np
from utils.config_loader import get_expected_dtypes_by_template


def load_column_types(schema_name: str = "balance_management_table") -> dict:
    """
    YAMLファイルからカラムの型定義を読み込む

    Args:
        schema_name (str): 使用するスキーマ名（デフォルトは'balance_management_table'）

    Returns:
        dict: カラム名とデータ型の対応辞書
    """
    return get_expected_dtypes_by_template(schema_name).get("shipping", {})


def clean_numeric_string(s):
    """
    数値文字列からカンマを除去し、数値に変換する

    Args:
        s: 入力値（文字列、数値、その他）

    Returns:
        クリーニング済みの値
    """
    if pd.isna(s) or s == "":
        return 0
    if isinstance(s, (int, float)):
        return s
    if isinstance(s, str):
        # カンマを除去して数値に変換を試みる
        try:
            cleaned = s.replace(",", "")
            if cleaned == "":
                return 0
            return float(cleaned)
        except ValueError:
            return 0
    return 0


def convert_series_to_int(series: pd.Series) -> pd.Series:
    """
    シリーズを整数型に変換する

    Args:
        series: 変換対象のシリーズ

    Returns:
        整数型に変換されたシリーズ
    """
    try:
        # まず数値として扱えるように変換
        numeric_series = series.apply(clean_numeric_string)
        # 整数に変換
        return numeric_series.astype("int64")
    except Exception as e:
        print(f"数値変換中にエラーが発生: {str(e)}")
        # エラーが発生した場合は0で埋める
        return pd.Series(0, index=series.index, dtype="int64")


def apply_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームの各カラムに対して、期待される型を適用する

    Args:
        df (pd.DataFrame): 型変換対象のデータフレーム

    Returns:
        pd.DataFrame: 型変換後のデータフレーム
    """
    # コピーを作成して警告を防ぐ
    df = df.copy()

    # 型定義を取得
    dtype_defs = get_expected_dtypes_by_template("balance_management_table")
    shipping_dtypes = dtype_defs.get("shipping", {})

    # 各カラムに型を適用
    for column, dtype in shipping_dtypes.items():
        if column in df.columns:
            try:
                if dtype == "int64":
                    df[column] = convert_series_to_int(df[column])
                elif dtype == "float64":
                    # 浮動小数点数の場合
                    df[column] = (
                        df[column].apply(clean_numeric_string).astype("float64")
                    )
                else:
                    # 文字列型などその他の型
                    df[column] = df[column].fillna("").astype(dtype)
            except Exception as e:
                print(
                    f"警告: カラム '{column}' の型変換中にエラーが発生しました: {str(e)}"
                )
                # エラーが発生した場合、そのカラムは元の値のまま保持

    return df
