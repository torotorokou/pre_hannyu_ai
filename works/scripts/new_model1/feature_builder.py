import pandas as pd
import numpy as np
import requests


class WeatherFeatureBuilder:
    """
    天気関連の特徴量を生成するクラス。

    Attributes:
        start_date (str): データ取得開始日。
        end_date (str): データ取得終了日。
        lat (float): 緯度。
        lon (float): 経度。
    """

    def __init__(self, start_date, end_date, lat=35.6895, lon=139.6917):
        """
        WeatherFeatureBuilderの初期化。

        Args:
            start_date (str): データ取得開始日。
            end_date (str): データ取得終了日。
            lat (float): 緯度。
            lon (float): 経度。
        """
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "precipitation_sum"],
            "timezone": "Asia/Tokyo",
        }

    def fetch(self):
        """
        天気データをAPIから取得し、データフレームに変換します。

        Returns:
            pd.DataFrame: 天気データを含むデータフレーム。
        """
        res = requests.get(self.url, params=self.params)
        data = res.json()

        df_weather = pd.DataFrame(
            {
                "日付": data["daily"]["time"],
                "平均気温": data["daily"]["temperature_2m_mean"],
                "降水量": data["daily"]["precipitation_sum"],
            }
        )
        df_weather["日付"] = pd.to_datetime(df_weather["日付"])
        df_weather = df_weather.set_index("日付")

        def classify_weather(row):
            if row["降水量"] > 100 and row["平均気温"] < 27:
                return "台風"
            elif row["降水量"] > 50:
                return "大雨"
            elif row["降水量"] > 1:
                return "雨"
            else:
                return "晴れ"

        df_weather["天気4分類"] = df_weather.apply(classify_weather, axis=1)
        df_weather = pd.get_dummies(df_weather, columns=["天気4分類"], prefix="天気")

        for col in ["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]:
            if col not in df_weather.columns:
                df_weather[col] = 0
        df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]] = (
            df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]]
            .fillna(0)
            .astype(int)
        )

        return df_weather


class WeightFeatureBuilder:
    """
    重量関連の特徴量を生成するクラス。

    Attributes:
        past_raw (pd.DataFrame): 過去のデータ。
        target_items (list): 対象品目のリスト。
        holidays (list): 祝日のリスト。
        weather_features (pd.DataFrame, optional): 天気関連の特徴量。
    """

    def __init__(self, past_raw, target_items, holidays, weather_features=None):
        """
        WeightFeatureBuilderの初期化。

        Args:
            past_raw (pd.DataFrame): 過去のデータ。
            target_items (list): 対象品目のリスト。
            holidays (list): 祝日のリスト。
            weather_features (pd.DataFrame, optional): 天気関連の特徴量。
        """
        self.past_raw = past_raw
        self.target_items = target_items
        self.holidays = holidays
        self.weather_features = (
            weather_features  # optional DataFrame with weather features
        )

    def build(self):
        """
        重量関連の特徴量を生成します。

        Returns:
            tuple: 特徴量データフレームとピボットデータフレーム。
        """
        # Debug info
        print(f"[DEBUG] WeightFeatureBuilder.build: past_raw.shape={getattr(self.past_raw, 'shape', None)}, target_items={self.target_items}")
        print(f"[DEBUG] WeightFeatureBuilder.build: holidays type={type(self.holidays)}, len_or_none={len(self.holidays) if hasattr(self.holidays, '__len__') else 'n/a'}")

        # Build pivot and statistical features
        df_pivot = create_weight_pivot(self.past_raw, self.target_items)
        df_feat = add_stat_features(df_pivot, self.target_items, self.past_raw)
        df_feat = add_calendar_features(df_feat, self.holidays)
        print(f"[DEBUG] WeightFeatureBuilder.build: df_pivot.shape={getattr(df_pivot, 'shape', None)}, df_feat.shape={getattr(df_feat, 'shape', None)}")

        # Merge weather features if provided
        if self.weather_features is not None:
            df_feat = (
                df_feat.merge(self.weather_features, left_index=True, right_index=True, how="left").fillna(0)
            )

        # Align pivot to features and drop NA rows
        df_feat = df_feat.dropna()
        df_pivot = df_pivot.loc[df_feat.index]
        return df_feat, df_pivot


class ReserveFeatureBuilder:
    """
    予約関連の特徴量を生成するクラス。

    Attributes:
        df_reserve (pd.DataFrame): 予約データ。
        top_k_clients (int): 上位得意先の数。
    """

    def __init__(self, df_reserve, top_k_clients=10):
        """
        ReserveFeatureBuilderの初期化。

        Args:
            df_reserve (pd.DataFrame): 予約データ。
            top_k_clients (int): 上位得意先の数。
        """
        self.df_reserve = df_reserve.copy()
        self.top_k_clients = top_k_clients

    def build(self):
        """
        予約関連の特徴量を生成します。

        Returns:
            pd.DataFrame: 予約関連の特徴量を含むデータフレーム。
        """
        print(f"[DEBUG] ReserveFeatureBuilder.build: incoming columns={list(self.df_reserve.columns)}")
        print(f"[DEBUG] ReserveFeatureBuilder.build: sample rows=\n{self.df_reserve.head(3).to_dict(orient='list')}")
        # Normalize common alternative column names coming from different CSV exports.
        # Some files use '台数' instead of '予約台数'. Map it to the expected name.
        if "台数" in self.df_reserve.columns and "予約台数" not in self.df_reserve.columns:
            self.df_reserve = self.df_reserve.rename(columns={"台数": "予約台数"})

        # Ensure '予約日' exists and convert to datetime
        if "予約日" in self.df_reserve.columns:
            self.df_reserve["予約日"] = pd.to_datetime(self.df_reserve["予約日"])
        else:
            # If the expected date column is missing, try some common alternatives or raise a clear error
            alt = None
            for c in self.df_reserve.columns:
                if "日" in c and ("予約" in c or "日" in c):
                    alt = c
                    break
            if alt:
                self.df_reserve = self.df_reserve.rename(columns={alt: "予約日"})
                self.df_reserve["予約日"] = pd.to_datetime(self.df_reserve["予約日"])
            else:
                raise KeyError("予約日 column not found in reservation data")

        # Coerce numeric for reservation counts; if missing, fill with zeros
        if "予約台数" in self.df_reserve.columns:
            self.df_reserve["予約台数"] = pd.to_numeric(
                self.df_reserve["予約台数"], errors="coerce"
            ).fillna(0)
        else:
            self.df_reserve["予約台数"] = 0

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
    """
    重量データをピボット形式に変換します。

    Args:
        past_raw (pd.DataFrame): 過去のデータ。
        target_items (list): 対象品目のリスト。

    Returns:
        pd.DataFrame: ピボット形式の重量データ。
    """
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
    """
    統計関連の特徴量を追加します。

    Args:
        df_pivot (pd.DataFrame): ピボット形式の重量データ。
        target_items (list): 対象品目のリスト。
        past_raw (pd.DataFrame): 過去のデータ。

    Returns:
        pd.DataFrame: 統計関連の特徴量を含むデータフレーム。
    """
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


def is_second_sunday(date: pd.Timestamp) -> bool:
    """
    指定された日付が第2日曜日かどうかを判定します。

    Args:
        date (pd.Timestamp): 判定する日付。

    Returns:
        bool: 第2日曜日の場合はTrue、それ以外はFalse。
    """
    if date.weekday() != 6:  # Sunday
        return False
    first_day = date.replace(day=1)
    sundays = [
        d
        for d in pd.date_range(start=first_day, end=first_day + pd.DateOffset(days=14))
        if d.weekday() == 6
    ]
    return date == sundays[1] if len(sundays) >= 2 else False


def add_calendar_features(df_feat, holidays):
    """
    カレンダー関連の特徴量を追加します。

    Args:
        df_feat (pd.DataFrame): 特徴量データフレーム。
        holidays (list): 祝日のリスト。

    Returns:
        pd.DataFrame: カレンダー関連の特徴量を含むデータフレーム。
    """
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    # holidays may be a set or other iterable; coerce to list first to ensure
    # pandas treats it as a list-like collection
    holiday_dates = pd.to_datetime(list(holidays)).sort_values()
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
        for start, end in zip(start_of_sequence, end_of_sequence, strict=False)
        if (end - start).days + 1 >= 2
    ]

    long_holiday_before = [
        start - pd.Timedelta(days=1) for start, _ in long_holiday_ranges
    ]
    long_holiday_after = [end + pd.Timedelta(days=1) for _, end in long_holiday_ranges]

    df_feat["連休前フラグ"] = df_feat.index.isin(long_holiday_before).astype(int)
    df_feat["連休後フラグ"] = df_feat.index.isin(long_holiday_after).astype(int)

    # --- 第2日曜日フラグの追加 ---
    df_feat["第2日曜フラグ"] = df_feat.index.map(is_second_sunday).astype(int)

    return df_feat
