import pandas as pd
import numpy as np
import requests

# 予約台数の列マッピングログを多重に出さないためのフラグ
_RESERVE_MAPPING_LOGGED = False


class WeatherFeatureBuilder:
    """Open-Meteo アーカイブ API から日次天気指標を取得し分類特徴量へ変換。

    変更点:
      - start_date/end_date を必ず 'YYYY-MM-DD' 文字列に正規化（Timestamp に含まれる時刻部で 400 になる問題対策）
      - 取得失敗時はフォールバック (指定期間を index に晴れ/0mm として埋める) を返し、呼び出し側の学習継続を可能に
      - デバッグログ出力 (開始/終了日, リクエストURL, ステータス, レコード数)
    """

    def __init__(self, start_date, end_date, lat=35.6895, lon=139.6917, enable_fallback=True):
        def _norm(d):
            # pandas Timestamp / datetime -> date isoformat
            try:
                import pandas as _pd
                if isinstance(d, _pd.Timestamp):
                    return d.date().isoformat()
            except Exception:
                pass
            if hasattr(d, 'date'):
                try:
                    return d.date().isoformat()
                except Exception:
                    pass
            if isinstance(d, str):
                # 文字列の場合は 最初の 'T' や 空白でスプリットし YYYY-MM-DD だけ抽出
                for sep in ['T', ' ']:
                    if sep in d:
                        d = d.split(sep)[0]
                return d[:10]
            return str(d)[:10]

        self.start_date = _norm(start_date)
        self.end_date = _norm(end_date)
        self.enable_fallback = enable_fallback
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "daily": ["temperature_2m_mean", "precipitation_sum"],
            "timezone": "Asia/Tokyo",
        }

    def _fallback(self):
        idx = pd.date_range(self.start_date, self.end_date, freq="D")
        df = pd.DataFrame(index=idx)
        df["平均気温"] = np.nan  # 後工程で必要なら別途補完
        df["降水量"] = 0.0
        # 晴れのみ one-hot
        df["天気_晴れ"], df["天気_雨"], df["天気_大雨"], df["天気_台風"] = 1, 0, 0, 0
        return df

    def build(self):
        print(f"[Weather] fetch {self.start_date} -> {self.end_date}")
        try:
            # 最終サニタイズ（万一属性が汚れていてもここで修正）
            for k in ("start_date", "end_date"):
                v = getattr(self, k)
                if isinstance(v, str):
                    for sep in ['T', ' ']:
                        if sep in v:
                            v = v.split(sep)[0]
                    v = v[:10]
                self.params[k] = v
            print(f"[Weather] final params start_date={self.params['start_date']} end_date={self.params['end_date']}")
            res = requests.get(self.url, params=self.params, timeout=30)
            status = res.status_code
            print(f"[Weather] status={status} url={res.url}")
            res.raise_for_status()
            data = res.json()
            # DataFrame 化
            daily = data.get("daily", {})
            if not daily:
                raise ValueError("'daily' key missing in response")
            df_weather = pd.DataFrame(
                {
                    "日付": daily.get("time", []),
                    "平均気温": daily.get("temperature_2m_mean", []),
                    "降水量": daily.get("precipitation_sum", []),
                }
            )
            df_weather["日付"] = pd.to_datetime(df_weather["日付"])  # already date strings
            df_weather = df_weather.set_index("日付")
        except Exception as e:
            print(f"[Weather][WARN] fetch failed: {e}")
            if self.enable_fallback:
                print("[Weather] using fallback zero-precipitation data")
                return self._fallback()
            raise RuntimeError(f"天気データの取得に失敗しました: {e}")

        # 天気分類
        def classify_weather(row):
            if row["降水量"] > 100 and row.get("平均気温", np.nan) < 27:
                return "台風"
            elif row["降水量"] > 50:
                return "大雨"
            elif row["降水量"] > 1:
                return "雨"
            else:
                return "晴れ"

        df_weather["天気分類"] = df_weather.apply(classify_weather, axis=1)
        df_weather = pd.get_dummies(df_weather, columns=["天気分類"], prefix="天気")

        # 欠損クラス補完
        for col in ["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]:
            if col not in df_weather.columns:
                df_weather[col] = 0
        df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]] = (
            df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]].fillna(0).astype(int)
        )
        print(f"[Weather] rows={len(df_weather)} cols={list(df_weather.columns)}")
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
        global _RESERVE_MAPPING_LOGGED
        # --- 日付列 正規化 ---
        if "予約日" not in self.df_reserve.columns:
            raise KeyError(f"予約日 列が存在しません: cols={list(self.df_reserve.columns)[:20]}")
        self.df_reserve["予約日"] = pd.to_datetime(self.df_reserve["予約日"])

        # --- 予約台数 列 フォールバック対応 ---
        if "予約台数" not in self.df_reserve.columns:
            candidate_map = [
                ("台数", "台数"),
                ("予約_台数", "予約_台数"),
                ("合計台数", "合計台数"),
            ]
            for src, label in candidate_map:
                if src in self.df_reserve.columns:
                    self.df_reserve["予約台数"] = self.df_reserve[src]
                    if not _RESERVE_MAPPING_LOGGED:
                        print(f"[ReserveFeatureBuilder] マッピング: {src} -> 予約台数")
                        _RESERVE_MAPPING_LOGGED = True
                    break
        # パターンマッチによる自動検出 (上記で未決定の場合)
        if "予約台数" not in self.df_reserve.columns:
            pattern_candidates = [c for c in self.df_reserve.columns if "台" in c and len(c) <= 8]
            if pattern_candidates:
                chosen = sorted(pattern_candidates, key=len)[0]
                self.df_reserve["予約台数"] = self.df_reserve[chosen]
                if not _RESERVE_MAPPING_LOGGED:
                    print(f"[ReserveFeatureBuilder] 自動検出: {chosen} -> 予約台数")
                    _RESERVE_MAPPING_LOGGED = True
        if "予約台数" not in self.df_reserve.columns:
            # ここで止めることで上位セルで原因を即座に把握できる
            raise KeyError(
                "予約台数 列が見つかりません (期待: 予約台数 / 台数). 現在の列: "
                + ",".join(list(self.df_reserve.columns)[:40])
            )

        self.df_reserve["予約台数"] = pd.to_numeric(
            self.df_reserve["予約台数"], errors="coerce"
        ).fillna(0)

        # --- 固定客 列 正規化 (bool/0-1 想定) ---
        if "固定客" in self.df_reserve.columns:
            if self.df_reserve["固定客"].dtype == object:
                self.df_reserve["固定客"] = (
                    self.df_reserve["固定客"].astype(str).str.contains("1|True|固定")
                )
            self.df_reserve["固定客"] = self.df_reserve["固定客"].astype(int)

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
