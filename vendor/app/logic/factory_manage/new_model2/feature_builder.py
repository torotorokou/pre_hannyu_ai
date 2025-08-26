import pandas as pd
import numpy as np
import requests


class WeatherFeatureBuilder:
    """
    指定された期間の天気データ（気温・降水量）を取得し、
    台風・大雨・雨・晴れの4分類を one-hot エンコードで返すクラス。
    """

    def __init__(self, start_date, end_date, lat=35.6895, lon=139.6917):
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "precipitation_sum"],
            "timezone": "Asia/Tokyo",
        }

    def build(self):
        try:
            res = requests.get(self.url, params=self.params)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            raise RuntimeError(f"天気データの取得に失敗しました: {e}")

        # DataFrame 化
        df_weather = pd.DataFrame(
            {
                "日付": data["daily"]["time"],
                "平均気温": data["daily"]["temperature_2m_mean"],
                "降水量": data["daily"]["precipitation_sum"],
            }
        )
        df_weather["日付"] = pd.to_datetime(df_weather["日付"])
        df_weather = df_weather.set_index("日付")

        # 天気分類
        def classify_weather(row):
            if row["降水量"] > 100 and row["平均気温"] < 27:
                return "台風"
            elif row["降水量"] > 50:
                return "大雨"
            elif row["降水量"] > 1:
                return "雨"
            else:
                return "晴れ"

        df_weather["天気分類"] = df_weather.apply(classify_weather, axis=1)

        # One-hot エンコーディング
        df_weather = pd.get_dummies(df_weather, columns=["天気分類"], prefix="天気")

        # 欠損補完（存在しないクラス列を追加）
        for col in ["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]:
            if col not in df_weather.columns:
                df_weather[col] = 0

        # データ型保証
        df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]] = (
            df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]]
            .fillna(0)
            .astype(int)
        )

        return df_weather


class WeightFeatureBuilder:
    def __init__(self, past_raw, target_items, holidays, weather_features=None):
        self.past_raw = past_raw
        self.target_items = target_items
        self.holidays = holidays
        self.weather_features = (
            weather_features  # optional DataFrame with weather features
        )

    def build(self):
        df_pivot = create_weight_pivot(self.past_raw, self.target_items)
        df_feat = add_stat_features(df_pivot, self.target_items, self.past_raw)
        df_feat = add_calendar_features(df_feat, self.holidays)

        if self.weather_features is not None:
            df_feat = df_feat.merge(
                self.weather_features, left_index=True, right_index=True, how="left"
            ).fillna(0)

        df_feat = df_feat.dropna()
        df_pivot = df_pivot.loc[df_feat.index]
        return df_feat, df_pivot


class ReserveFeatureBuilder:
    def __init__(self, df_reserve, top_k_clients=10):
        self.df_reserve = df_reserve.copy()
        self.top_k_clients = top_k_clients

    def build(self):
        self.df_reserve["予約日"] = pd.to_datetime(self.df_reserve["予約日"])
        self.df_reserve["予約台数"] = pd.to_numeric(
            self.df_reserve["予約台数"], errors="coerce"
        ).fillna(0)

        top_clients = (
            self.df_reserve["予約得意先名"]
            .value_counts()
            .head(self.top_k_clients)
            .index
        )
        self.df_reserve["上位得意先フラグ"] = (
            self.df_reserve["予約得意先名"].isin(top_clients).astype(int)
        )

        df_feat = self.df_reserve.groupby("予約日").agg(
            予約件数=("予約得意先名", "count"),
            固定客予約数=("固定客", lambda x: x.sum()),
            非固定客予約数=("固定客", lambda x: (~x).sum()),
            上位得意先予約数=("上位得意先フラグ", "sum"),
            予約合計台数=("予約台数", "sum"),
            平均台数=("予約台数", "mean"),
        )
        df_feat["固定客比率"] = df_feat["固定客予約数"] / df_feat["予約件数"]
        return df_feat.fillna(0)


def create_weight_pivot(past_raw, target_items):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    for item in target_items:
        if item not in df_pivot.columns:
            df_pivot[item] = 0
    df_pivot = df_pivot.sort_index()
    df_pivot["合計"] = df_pivot[target_items].sum(axis=1)
    return df_pivot


def add_stat_features(df_pivot, target_items, past_raw):
    df_feat = pd.DataFrame(index=df_pivot.index)

    for item in target_items:
        df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
        df_feat[f"{item}_前週平均"] = df_pivot[item].shift(1).rolling(7).mean()

    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["合計_前週平均"] = df_pivot["合計"].shift(1).rolling(7).mean()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = (
        daily_avg.shift(1).rolling(60, min_periods=10).median()
    )

    return df_feat


def add_calendar_features(df_feat, holidays):
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    holiday_dates = pd.to_datetime(holidays).sort_values()
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat["祝日前フラグ"] = df_feat.index.map(
        lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df_feat["祝日後フラグ"] = df_feat.index.map(
        lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)

    # --- 連休前・連休後フラグの追加 ---
    holiday_diff = holiday_dates.to_series().diff().dt.days
    start_of_sequence = holiday_dates[(holiday_diff != 1) | (holiday_diff.isna())]
    end_of_sequence = holiday_dates[
        (holiday_diff.shift(-1) != 1) | (holiday_diff.shift(-1).isna())
    ]

    long_holiday_ranges = [
        (start, end)
        for start, end in zip(start_of_sequence, end_of_sequence)
        if (end - start).days + 1 >= 2
    ]

    long_holiday_before = [
        start - pd.Timedelta(days=1) for start, _ in long_holiday_ranges
    ]
    long_holiday_after = [end + pd.Timedelta(days=1) for _, end in long_holiday_ranges]

    df_feat["連休前フラグ"] = df_feat.index.isin(long_holiday_before).astype(int)
    df_feat["連休後フラグ"] = df_feat.index.isin(long_holiday_after).astype(int)

    return df_feat
