from utils.logger import app_logger
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from config.loader.main_path import MainPath
from logic.manage.readers.read_transport_discount import ReadTransportDiscount
import streamlit as st
import time
import re


from logic.manage.processors.block_unit_price.process0 import (
    make_df_shipping_after_use,
    apply_unit_price_addition,
    apply_transport_fee_by1,
)
from logic.manage.processors.block_unit_price.process1 import (
    create_transport_selection_form,
)
from logic.manage.processors.block_unit_price.process2 import (
    confirm_transport_selection,
    apply_transport_fee_by_vendor,
    apply_weight_based_transport_fee,
    make_total_sum,
    df_cul_filtering,
    first_cell_in_template,
    make_sum_date,
)

# デバッグ用
from utils.debug_tools import debug_pause


def process(dfs):
    """
    ブロック単価計算の主処理を行う関数

    Args:
        dfs: 入力データフレーム群

    Returns:
        pd.DataFrame: 計算結果を含むマスターCSV
    """
    logger = app_logger()

    # --- 内部ステップ管理 ---
    mini_step = st.session_state.get("process_mini_step", 0)

    # --- 設定とマスターデータの読み込み ---
    # テンプレート設定
    template_key = "block_unit_price"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # マスターデータ
    config = get_template_config()["block_unit_price"]
    master_path = config["master_csv_path"]["vendor_code"]
    master_csv = load_master_and_template(master_path)

    # 運搬費データ
    mainpath = MainPath()
    reader = ReadTransportDiscount(mainpath)
    df_transport_cost = reader.load_discounted_df()

    # 出荷データ
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")

    # --- 段階的処理の実行 ---
    if mini_step == 0:
        _process_step0(df_shipping, master_csv, df_transport_cost)
        return None
    elif mini_step == 1:
        _process_step1(df_transport_cost)
        return None
    elif mini_step == 2:
        return _process_step2(df_shipping, master_csv, df_transport_cost)

    return None


def _process_step0(
    df_shipping: pd.DataFrame, master_csv: pd.DataFrame, df_transport_cost: pd.DataFrame
) -> None:
    """繰り返し不必要な処理
        基本データ処理と運搬費（固定のもの、社数１）を行うステップ0の実装
        繰り返す必要がないため、このステップに入れる。

    Args:
        df_shipping: 出荷データ
        master_csv: マスターデータ
        df_transport: 運搬費データ
    """
    logger = app_logger()
    logger.info("▶️ Step0: フィルタリング・単価追加・固定運搬費")

    df_shipping = make_df_shipping_after_use(master_csv, df_shipping)  # フィルタリング
    df_shipping = apply_unit_price_addition(master_csv, df_shipping)  # 単価追加
    df_shipping = apply_transport_fee_by1(df_shipping, df_transport_cost)  # 固定運搬費

    st.session_state.df_shipping_first = df_shipping
    st.session_state.process_mini_step = 1
    st.rerun()


def _process_step1(df_transport: pd.DataFrame) -> None:
    """繰り返し必要な処理
        運搬業者選択を行うステップ1の実装

    処理の流れ:
        1. 前のステップで保存した出荷データを取得
        2. 運搬業者の選択状態を確認
            - 未選択：選択フォームを表示して選択を促す
            - 選択済：次のステップに進む
        3. 選択結果をセッションに保存してページを再読み込み

    Args:
        df_transport: 運搬費データ
    """
    # ロガーの初期化
    logger = app_logger()
    logger.info("▶️ Step1: 選択式運搬費")

    # ステップ1: 前のステップの出荷データを取得
    df_after = st.session_state.df_shipping_first

    # ステップ2: 運搬業者の選択状態を確認
    if not st.session_state.get("block_unit_price_confirmed", False):
        # 未選択の場合：選択フォームを表示
        df_after = create_transport_selection_form(df_after, df_transport)

        # ステップ3: 選択結果を保存して再読み込み
        st.session_state.df_shipping = df_after
        st.rerun()
    else:
        # 選択済みの場合：次のステップへ進む
        logger.info("▶️ 選択済みなのでスキップ")
        st.session_state.process_mini_step = 2
        st.rerun()


def _process_step2(
    df_shipping: pd.DataFrame, master_csv: pd.DataFrame, df_transport: pd.DataFrame
) -> pd.DataFrame:
    """最終計算処理を行うステップ2の実装

    Args:
        df_shipping: 出荷データ
        df_transport: 運搬費データ

    Returns:
        pd.DataFrame: 計算結果を含むマスターCSV
    """
    logger = app_logger()
    logger.info("▶️ Step2: 加算処理実行中")
    df_after = st.session_state.df_shipping

    # 運搬業者選択の確認
    confirm_transport_selection(df_after)

    # 運搬費の計算と適用
    df_after = apply_transport_fee_by_vendor(df_after, df_transport)
    df_after = apply_weight_based_transport_fee(df_after, df_transport)

    # ブロック単価の計算と表示用データの作成
    df_after = make_total_sum(df_after, master_csv)
    df_after = df_cul_filtering(df_after)
    master_csv = first_cell_in_template(df_after)
    master_csv = make_sum_date(master_csv, df_shipping)

    # ステートの初期化
    st.session_state.process_mini_step = 0

    return master_csv
