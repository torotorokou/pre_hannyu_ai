import pandas as pd


def make_df_shipping_after_use(
    master_csv: pd.DataFrame, df_shipping: pd.DataFrame
) -> pd.DataFrame:
    """出荷データをマスターデータに基づいてフィルタリングし、必要な情報を追加する

    Args:
        master_csv (pd.DataFrame): マスターデータ（業者CD、品名、運搬社数などを含む）
        df_shipping (pd.DataFrame): 出荷データ

    Returns:
        pd.DataFrame: フィルタリングと情報追加が完了した出荷データ
    """

    def _filter_by_vendor_code(df: pd.DataFrame) -> pd.DataFrame:
        """マスターCSVの業者CDでフィルタリング"""
        return df[df["業者CD"].isin(master_csv["業者CD"])].copy()

    def _filter_by_item_name(df: pd.DataFrame) -> pd.DataFrame:
        """丸源のフィルタリング"""
        # 品名指定があるものを抽出
        item_filter_df = master_csv[master_csv["品名"].notna()][
            ["業者CD", "品名"]
        ].drop_duplicates()

        if item_filter_df.empty:
            return df

        # 業者CDと品名のペアが一致する行のみ残す
        df = df.merge(item_filter_df, on=["業者CD", "品名"], how="left", indicator=True)
        df = df[
            (df["_merge"] == "both") | (~df["業者CD"].isin(item_filter_df["業者CD"]))
        ]
        return df.drop(columns=["_merge"])

    def _add_transport_info(df: pd.DataFrame) -> pd.DataFrame:
        """運搬関連情報の追加"""
        # 運搬社数を追加（業者CDごとに1件に絞る）
        unique_master = master_csv.drop_duplicates(subset=["業者CD"])[
            ["業者CD", "運搬社数"]
        ]
        df = df.merge(unique_master, on="業者CD", how="left")

        # 運搬費カラムを初期化
        df["運搬費"] = 0
        return df

    # メイン処理の実行
    df_after = _filter_by_vendor_code(df_shipping)
    df_after = _filter_by_item_name(df_after)

    # 正味重量が0のデータを除外
    df_after = df_after[df_after["正味重量"].fillna(0) != 0]

    # 運搬情報の追加
    df_after = _add_transport_info(df_after)

    # 業者CDでソート
    return df_after.sort_values(by="業者CD").reset_index(drop=True)


def apply_unit_price_addition(
    master_csv: pd.DataFrame, df_shipping: pd.DataFrame
) -> pd.DataFrame:
    """出荷データの単価に手数料を加算する

    マスターデータに登録されている業者ごとの手数料を、
    出荷データの単価に加算します。加算は業者CDをキーにして行われます。

    Args:
        master_csv (pd.DataFrame): 業者CDと手数料情報を含むマスターデータ
        df_shipping (pd.DataFrame): 単価を含む出荷データ

    Returns:
        pd.DataFrame: 手数料が加算された出荷データ

    Note:
        - 業者CDをキーにしてマスターデータと結合します
        - 手数料が設定されていない業者は、元の単価がそのまま維持されます
    """
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    return apply_column_addition_by_keys(
        base_df=df_shipping,
        addition_df=master_csv,
        join_keys=["業者CD"],
        value_col_to_add="手数料",
        update_target_col="単価",
    )


def apply_transport_fee_by1(
    df_shipping: pd.DataFrame, df_transport: pd.DataFrame
) -> pd.DataFrame:
    """運搬社数が1の業者に対して固定運搬費を適用する

    処理の流れ:
        ① 運搬社数 = 1 の行だけを抽出
        ② 抽出した行に運搬費を加算
        ③ 運搬社数 != 1 の行は変更せずに保持
        ④ ②と③の結果を結合し、業者CDでソート

    Args:
        df_shipping (pd.DataFrame): 出荷データ（運搬社数カラムを含む）
        df_transport (pd.DataFrame): 運搬費マスターデータ

    Returns:
        pd.DataFrame: 固定運搬費が適用された出荷データ（業者CD順にソート済み）
    """
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    def _extract_single_transport_rows(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """① 運搬社数 = 1 の行とそれ以外の行を分離"""
        single_transport = df[df["運搬社数"] == 1].copy()
        other_transport = df[df["運搬社数"] != 1].copy()
        return single_transport, other_transport

    def _apply_transport_fee(target_df: pd.DataFrame) -> pd.DataFrame:
        """② 運搬費の加算処理を適用"""
        return apply_column_addition_by_keys(
            base_df=target_df,
            addition_df=df_transport,
            join_keys=["業者CD"],
            value_col_to_add="運搬費",
            update_target_col="運搬費",
        )

    # ①と③: 運搬社数による行の分離
    target_rows, other_rows = _extract_single_transport_rows(df_shipping)

    # ②: 運搬社数=1の行に運搬費を適用
    updated_target_rows = _apply_transport_fee(target_rows)

    # ④: 全ての行を結合して業者CDでソート
    df_after = pd.concat([updated_target_rows, other_rows], ignore_index=True)
    return df_after.sort_values(by="業者CD").reset_index(drop=True)
