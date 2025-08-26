import yaml
import pandas as pd
from pathlib import Path
from utils.type_converter import resolve_dtype
import os
from typing import Optional
from config.loader.main_path import MainPath

from typing import Union


def get_path_from_yaml(
    key: Union[str, list[str]], section: Optional[str] = None
) -> str:
    mainpath = MainPath()
    path = mainpath.get_path(key, section)
    return str(path)


def load_yaml(key_or_path: str, section: Optional[str] = None) -> dict:
    """
    YAMLファイルを辞書形式で読み込む。

    Parameters:
        key_or_path (str): 相対パスまたはmain_paths.yamlのキー名
        section (str, optional): キーが格納されているmain_paths.yamlのセクション名（例: 'config_files'）

    Returns:
        dict: YAMLから読み込まれた辞書データ
    """
    mainpath = MainPath()
    path = mainpath.get_path(key_or_path, section)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_app_config() -> dict:
    """main_paths.yaml 経由で app_config.yaml を読み込む"""
    return load_yaml("app_config", section="config_files")


def get_expected_dtypes() -> dict:
    """main_paths.yaml 経由で expected_dtypes.yaml を読み込み、型を解決する"""
    raw_yaml = load_yaml("expected_dtypes", section="config_files")

    resolved = {}

    for template_key, file_map in raw_yaml.items():
        resolved[template_key] = {}

        for file_key, dtype_map in file_map.items():
            flattened = {}

            # --- ✅ dict（通常の形式）の場合
            if isinstance(dtype_map, dict):
                flattened = {
                    col: resolve_dtype(dtype_str)
                    for col, dtype_str in dtype_map.items()
                }

            # --- ✅ list（アンカー＋追加型）の場合
            elif isinstance(dtype_map, list):
                for item in dtype_map:
                    if isinstance(item, dict):
                        for col, dtype_str in item.items():
                            flattened[col] = resolve_dtype(dtype_str)
                    else:
                        raise TypeError(f"unexpected format in expected_dtypes: {item}")

            else:
                raise TypeError(f"unexpected format for {file_key}: {dtype_map}")

            resolved[template_key][file_key] = flattened

    return resolved


def get_template_config() -> dict:
    """main_paths.yaml 経由で templates_config.yaml を読み込む"""
    return load_yaml("templates_config", section="config_files")


def get_page_config() -> list:
    """main_paths.yaml 経由で page_config.yaml のページ定義を取得"""
    return load_yaml("page_config", section="config_files")["pages"]


def get_unit_price_table_csv() -> pd.DataFrame:
    """
    単価表CSVを読み込んでDataFrameとして返す。
    """
    mainpath = MainPath()
    csv_path = mainpath.get_path("unit_price_table", section="csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def receive_header_definition() -> pd.DataFrame:
    """
    受入ヘッダー定義CSVを読み込んでDataFrameとして返す。
    """
    mainpath = MainPath()
    csv_path = mainpath.get_path("receive_header_definition", section="csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def get_page_dicts():
    """
    ページ設定（page_config.yaml）から
    - 表示名→IDの辞書
    - ID→表示名の逆引き
    - 表示名リスト（UI用）
    を返す
    """
    pages = get_page_config()  # これはすでに修正済み
    page_dict = {p["label"]: p["id"] for p in pages}
    reverse_dict = {v: k for k, v in page_dict.items()}
    labels = list(page_dict.keys())
    return page_dict, reverse_dict, labels


def get_csv_sources_config() -> dict:
    """main_paths.yaml 経由で csv_sources_config.yaml を読み込む"""
    return load_yaml("csv_sources_config", section="config_files")


def get_csv_label_map() -> dict:
    config = get_csv_sources_config()
    return {key: value["label"] for key, value in config.items()}


def get_csv_date_columns() -> dict:
    """
    各CSVファイル種別に対応する日付カラム名を取得する。

    Returns:
        dict: { "receive": "伝票日付", "yard": "伝票日付", ... } の形式
    """
    config = get_csv_sources_config()
    return {key: value["date_column"] for key, value in config.items()}


def get_required_files_map() -> dict:
    """
    各テンプレートに必要なファイル（required_files）を辞書形式で取得。

    Returns:
        dict: テンプレートキー → 必須ファイルリスト
    """
    config = get_template_config()
    return {key: value.get("required_files", []) for key, value in config.items()}


def get_file_keys_map() -> dict:
    """
    各テンプレートに必要・任意ファイルを辞書形式で取得。

    Returns:
        dict: テンプレートキー → {'required': [...], 'optional': [...]}
    """
    config = get_template_config()
    return {
        key: {
            "required": value.get("required_files", []),
            "optional": value.get("optional_files", []),
        }
        for key, value in config.items()
    }


def get_template_descriptions() -> dict:
    config = get_template_config()
    return {key: value.get("description", []) for key, value in config.items()}


def get_template_dict() -> dict:
    """
    テンプレートの表示ラベル → テンプレートキー の辞書を返す。

    Returns:
        dict: 例 {"工場日報": "factory_report", ...}
    """
    config = get_template_config()
    return {value["label"]: key for key, value in config.items()}


def get_expected_dtypes_by_template(template_key: str) -> dict:
    """
    指定されたテンプレートキーに対応するCSVファイルごとの
    カラム型定義（expected_dtypes）を返す。

    この関数は expected_dtypes.yaml を読み込んだ結果から、
    指定テンプレートに対応する型定義だけを抽出して返します。

    Parameters:
        template_key (str): テンプレート名（例: "average_sheet", "factory_report"）

    Returns:
        dict: ファイルキーごとのカラム名と型の辞書。
              例:
              {
                  "receive": {
                      "金額": float,
                      "正味重量": int,
                      "伝票日付": "datetime64[ns]"
                  },
                  "yard": {
                      "品名": str,
                      ...
                  }
              }
              対応テンプレートが存在しない場合は空の辞書を返します。
    """
    config = get_expected_dtypes()
    return config.get(template_key, {})


def get_required_columns_definition(template_name: str) -> dict:
    all_defs = load_yaml("required_columns_definition", section="config_files")
    return all_defs.get(template_name, {})


# 工場用のメニューリストを辞書形式で返す
def load_factory_menu_options() -> list[dict]:
    config = load_yaml("factory_manage_menu_config", section="config_files")
    return config["menu_options"]  # ✅ dict list を返す


# 工場用_必要なファイルを辞書形式で返す
def load_factory_required_files() -> list[dict]:
    config = load_yaml("factory_manage_menu_config", section="config_files")
    return config["required_files"]  # ✅ dict list を返す
