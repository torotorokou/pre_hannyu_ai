import pandas as pd
from config.loader.main_path import MainPath


class ReadTransportDiscount:
    """
    運搬費CSVを読み込み、合積割引を自動適用したDataFrameを返すクラス。
    """

    def __init__(self, mainpath: MainPath):
        self.mainpath = mainpath
        self.discount_service = TransportDiscountService()

    def load_discounted_df(self) -> pd.DataFrame:
        """
        CSVファイルを読み込み、合積に基づいた割引を適用したDataFrameを返す。
        """
        # YAML経由でCSVパスを取得
        csv_path = self.mainpath.get_path("transport_costs", section="csv")

        # CSV読み込み
        df = pd.read_csv(csv_path)

        # 割引適用
        df = self.discount_service.apply_discount(df)

        return df


class TransportDiscountService:
    def __init__(self, discount_rate: float = 0.5):
        self.discount_rate = discount_rate

    def apply_discount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        合積 == 1 の行をコピーし、運搬業者名に「・合積」を追加し、
        運搬費を割引して新たな行として追加する。
        元の行はそのまま残す。
        """
        df = df.copy()

        if "合積" not in df.columns:
            raise KeyError("入力データに '合積' 列が存在しません")

        # 合積対象の行を抽出
        mask = df["合積"] == 1
        discount_rows = df[mask].copy()

        # 運搬業者名に「・合積」を付与
        discount_rows["運搬業者"] = discount_rows["運搬業者"].astype(str) + "・合積"

        # 運搬費を割引
        discount_rows["運搬費"] = (
            discount_rows["運搬費"].astype(float) * self.discount_rate
        )

        # 合積フラグを 0 に（新しい行なので既に適用済）
        discount_rows["合積"] = 0

        # 元のdfに追加
        df = pd.concat([df, discount_rows], ignore_index=True)

        return df
