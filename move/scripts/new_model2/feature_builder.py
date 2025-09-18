# --- WeightFeatureBuilder: 重量データから特徴量を生成 ---
class WeightFeatureBuilder:
    def __init__(self, past_raw, target_items, holidays, weather_features=None):
        self.past_raw = past_raw
        self.target_items = target_items
        self.holidays = holidays
        self.weather_features = weather_features  # optional DataFrame with weather features

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

# --- WeightFeatureBuilderで使用される補助関数 ---
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
    # 追加: 月 / 前営業日フラグ / 翌営業日フラグ
    try:
        df_feat["月"] = df_feat.index.month
    except Exception:
        # 古いpandasでも安全に
        df_feat["月"] = df_feat.index.to_period('M').month.astype(int)

    # 営業日判定（祝日と土日を非営業日とし、それ以外を営業日）
    import pandas as _pd
    holiday_dates = _pd.to_datetime(holidays).sort_values()
    holiday_set = set(_pd.to_datetime(holiday_dates).date)
    def _is_business_day(dts):
        d = _pd.Timestamp(dts).date()
        # 月-金 かつ 祝日でない
        return (0 <= _pd.Timestamp(dts).weekday() <= 4) and (d not in holiday_set)
    # 前/翌の営業日を計算
    def _prev_business_day(dts):
        cur = _pd.Timestamp(dts) - _pd.Timedelta(days=1)
        for _ in range(10):
            if _is_business_day(cur):
                return cur
            cur -= _pd.Timedelta(days=1)
        return None
    def _next_business_day(dts):
        cur = _pd.Timestamp(dts) + _pd.Timedelta(days=1)
        for _ in range(10):
            if _is_business_day(cur):
                return cur
            cur += _pd.Timedelta(days=1)
        return None
    df_feat["前営業日フラグ"] = df_feat.index.map(lambda d: 1 if _prev_business_day(d) is not None else 0).astype(int)
    df_feat["翌営業日フラグ"] = df_feat.index.map(lambda d: 1 if _next_business_day(d) is not None else 0).astype(int)

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
# --- ReserveFeatureBuilder: 予約データから特徴量を生成 ---
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

import pandas as pd
import numpy as np
import requests

# 予約台数の列マッピングログを多重に出さないためのフラグ
_RESERVE_MAPPING_LOGGED = False

# --- WeatherFeatureBuilder: Open-Meteo APIから天気特徴量を取得 ---
class WeatherFeatureBuilder:
    """Open-Meteo アーカイブ API から日次天気指標を取得し分類特徴量へ変換。"""
    def __init__(self, start_date, end_date, lat=35.6895, lon=139.6917, enable_fallback=True):
        def _norm(d):
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
        df["平均気温"] = np.nan
        df["降水量"] = 0.0
        df["天気_晴れ"], df["天気_雨"], df["天気_大雨"], df["天気_台風"] = 1, 0, 0, 0
        return df
    def build(self):
        print(f"[Weather] fetch {self.start_date} -> {self.end_date}")
        try:
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
            daily = data.get("daily", {})
            if not daily:
                raise ValueError("'daily' key missing in response")
            df_weather = pd.DataFrame({
                "日付": daily.get("time", []),
                "平均気温": daily.get("temperature_2m_mean", []),
                "降水量": daily.get("precipitation_sum", []),
            })
            df_weather["日付"] = pd.to_datetime(df_weather["日付"])
            df_weather = df_weather.set_index("日付")
        except Exception as e:
            print(f"[Weather][WARN] fetch failed: {e}")
            if self.enable_fallback:
                print("[Weather] using fallback zero-precipitation data")
                return self._fallback()
            raise RuntimeError(f"天気データの取得に失敗しました: {e}")
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
        for col in ["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]:
            if col not in df_weather.columns:
                df_weather[col] = 0
        df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]] = (
            df_weather[["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]].fillna(0).astype(int)
        )
        print(f"[Weather] rows={len(df_weather)} cols={list(df_weather.columns)}")
        return df_weather


# 完成版: API納品用モデル
# 依存を局所化するため、StackingEnsemble をローカル定義（学習成果物の辞書から再構築）
class StackingEnsemble:
    """再構築可能なスタッキング推論器。

    Components
    ----------
    stage1_models: dict
        { 'items': {ITEM: {'scaler','selector','base_models','meta_model','feature_list',...}}, 'raw_feature_names': [...]} 形式
    stage2: dict
        { 'scaler','selector','estimator','feature_list' } 形式
    target_items: list[str]
        品目名のリスト（stage1の順序に一致）
    """

    def __init__(self, stage1_models, stage2, target_items):
        self.stage1 = stage1_models or {}
        self.stage2 = stage2 or {}
        self.target_items = list(target_items or [])

    def __getstate__(self):
        """Pickle互換: モジュール参照等を状態から除外する。"""
        return {
            'stage1': self.stage1,
            'stage2': self.stage2,
            'target_items': self.target_items,
        }

    def __setstate__(self, state):
        self.stage1 = state.get('stage1', {})
        self.stage2 = state.get('stage2', {})
        self.target_items = list(state.get('target_items', []) or [])

    def _predict_stage1(self, X_df):
        import numpy as np
        out = {}
        items = self.stage1.get("items", {}) if isinstance(self.stage1, dict) else {}
        for item, art in items.items():
            try:
                scaler = art["scaler"]
                selector = art["selector"]
                base_models = art.get("base_models", [])
                meta_model = art["meta_model"]
                feat_list = art.get("feature_list") or self.stage1.get("raw_feature_names", [])
                X = X_df.reindex(columns=feat_list, fill_value=0.0)
                Xf = selector.transform(scaler.transform(X))
                meta_in = np.column_stack([m.predict(Xf) for m in base_models]) if base_models else Xf
                pred = float(meta_model.predict(meta_in)[0])
                out[item] = pred
            except Exception:
                out[item] = 0.0
        return out

    def predict_with_breakdown(self, X_df):
        import pandas as pd
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)
        item_preds = self._predict_stage1(X_df)
        feat_list = self.stage2.get("feature_list", [])
        item_cols = [f"{item}_予測" for item in self.target_items]
        X2 = {col: [0.0] for col in feat_list}
        for item in self.target_items:
            key = f"{item}_予測"
            if key in X2:
                X2[key][0] = item_preds.get(item, 0.0)
        for col in feat_list:
            if col not in item_cols and col in X_df.columns:
                X2[col][0] = X_df.iloc[0][col]
        import pandas as pd
        X2_df = pd.DataFrame(X2)
        try:
            scaler2 = self.stage2["scaler"]
            selector2 = self.stage2["selector"]
            est2 = self.stage2["estimator"]
            total = float(est2.predict(selector2.transform(scaler2.transform(X2_df)))[0])
        except Exception:
            total = float(sum(item_preds.values()))
        per_item = {k: float(item_preds.get(k, 0.0)) for k in self.target_items}
        if not any(per_item.values()) and total:
            n = len(self.target_items) or 1
            per_item = {k: float(total / n) for k in self.target_items}
        return {"per_item": per_item, "total": total}

    def predict(self, X):
        import numpy as np
        import pandas as pd
        X = pd.DataFrame(X)
        out = self.predict_with_breakdown(X)
        return np.array([out.get("total", 0.0)])
# import経路を堅牢化（どちらか存在する方を使う）
try:
    from scripts.new_model2.cache_utils import (
        full_walkforward_cached,
        get_feature_list,
        get_target_items,
    )
except Exception:  # pragma: no cover
    try:
        from works.scripts.new_model2.cache_utils import (
            full_walkforward_cached,
            get_feature_list,
            get_target_items,
        )
    except Exception:
        # pickle読込のみでwalkforwardを使わないケースのために遅延エラー化
        full_walkforward_cached = None  # type: ignore
        get_feature_list = None  # type: ignore
        get_target_items = None  # type: ignore

class NewModel2Predictor:
    """
    APIバックエンド納品用: 日付指定で本番推論を実行
    - 必要なデータ（df_raw, df_reserve, df_weather, holidays）を保持
    - full_walkforward_cached を用いて、特徴量生成・モデル推論・キャッシュを統合
    """
    def __init__(self, df_raw, df_reserve, holidays, df_weather, top_n=5, min_stage1_days=30, min_stage2_days=30, allowed_features=None, model_profile: str = "fast", stage1_meta: str = "gbr"):
        self.df_raw = df_raw
        self.df_reserve = df_reserve
        self.holidays = holidays
        self.df_weather = df_weather
        self.top_n = top_n
        self.min_stage1_days = min_stage1_days
        self.min_stage2_days = min_stage2_days
        self.model_profile = model_profile
        self.stage1_meta = stage1_meta
        # Noneの場合でも後段で扱えるよう標準化
        self.allowed_features = list(allowed_features) if allowed_features is not None else None
        # 推論キャッシュ
        self._cache = None
        self._meta = None
        self._dates = None
        self._actual = None
        self._pred = None
        self._model = None            # 推論器（sklearn estimator想定）を格納することを目標
        self._model_raw = None        # 互換用: 元のモデル構造（dict等）
        self._target_items = None     # 品目内訳名
        self._item_shares = None      # 品目の平均シェア（合計=1）
        self._init_predict()

    def _init_predict(self):
        """学習・キャッシュ初期化。
        pickleのロード時は__init__は呼ばれないため（既存インスタンスの復元）、
        この処理は学習時のみ実行される想定。cache_utilsが無い環境ではスキップ。
        """
        if full_walkforward_cached is None:
            return
        actual, pred, model, dates, meta = full_walkforward_cached(
            df_raw=self.df_raw,
            holidays=self.holidays,
            df_reserve=self.df_reserve,
            df_weather=self.df_weather,
            min_stage1_days=self.min_stage1_days,
            min_stage2_days=self.min_stage2_days,
            top_n=self.top_n,
            allowed_features=self.allowed_features,
            force_recompute=False,
            describe=False,
            model_profile=self.model_profile,
            stage1_meta=self.stage1_meta,
        )
        self._actual = actual
        self._pred = pred
        # モデル格納: まずrawを保持し、可能なら推論器に正規化
        self._model_raw = model
        try:
            self._model = self._resolve_estimator_from(model)
        except Exception:
            self._model = model  # 見つからなければそのまま保持（フォールバック可）
        self._dates = dates
        self._meta = meta
        if isinstance(meta, dict):
            self._cache = meta.get('_cache', {})
        # allowed_featuresが未設定の場合、メタから復元できるなら使う
        if self.allowed_features is None and isinstance(meta, dict):
            af = meta.get('allowed_features') or meta.get('features')
            if isinstance(af, (list, tuple)):
                self.allowed_features = list(af)
        # ターゲット品目の推定
        self._target_items = self._infer_target_items(meta)
        # 品目シェア（平均）を推定（totalしか予測できない場合の内訳に使用）
        self._item_shares = self._estimate_item_shares(self.df_raw, self._target_items)

    # --- 特徴量行の構築（指定日付） ----------------------------------
    def _build_df_feat_today(self, date_str: str) -> pd.DataFrame:
        """内部データから指定日付の特徴量1行を構築。

        - df_raw/holidays から基本特徴
        - df_reserve, df_weather を当日まででマージ
        - 学習時の target_items とできるだけ同じ feature_list を用意
        """
        if not self._target_items:
            # target_items を df_raw から推定
            try:
                if get_target_items is not None:
                    self._target_items = get_target_items(self.df_raw)
            except Exception:
                pass
        # 日付型安全化
        d = pd.to_datetime(str(date_str)[:10])
        df_raw = self.df_raw.copy()
        df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"]).dt.floor('D')
        df_raw = df_raw.sort_values("伝票日付")
        # 基本特徴
        wfb = WeightFeatureBuilder(df_raw, self._target_items, self.holidays)
        df_feat, df_pivot = wfb.build()
        # 予約・天気当日まで
        df_r = self.df_reserve.copy()
        if len(df_r):
            df_r["予約日"] = pd.to_datetime(df_r["予約日"]).dt.floor('D')
        rfb = ReserveFeatureBuilder(df_r)
        df_res_feat = rfb.build().loc[:d]
        df_w = self.df_weather.copy()
        if len(df_w):
            df_w.index = pd.to_datetime(df_w.index).floor('D')
        # マージ
        df_feat_all = df_feat.merge(df_res_feat, left_index=True, right_index=True, how="left")
        df_feat_all = df_feat_all.merge(df_w.loc[:d], left_index=True, right_index=True, how="left")
        df_feat_all = df_feat_all.fillna(0)
        # 行抽出（存在しない場合は最も近い過去日を使用）
        if d in df_feat_all.index:
            row = df_feat_all.loc[[d]].copy()
        else:
            past_idx = df_feat_all.index[df_feat_all.index <= d]
            if len(past_idx) == 0:
                # 最初の行を使う
                row = df_feat_all.tail(1).copy()
            else:
                row = df_feat_all.loc[[past_idx.max()]].copy()
        return row

    # --- 内部ユーティリティ -------------------------------------------
    def _resolve_estimator(self):
        """self._model から実際に predict を持つ推論器を取り出す。
        - そのまま estimator
        - dict の中の 'stage1_model' など
        - ('meta', estimator) の形 など
        見つからない場合は ValueError
        """
        m = self._model
        # すでに推論器
        if hasattr(m, 'predict'):
            return m
        # 新形式: {'stage1_models': {...}, 'stage2_model': {...}}
        if isinstance(m, dict) and 'stage1_models' in m and 'stage2_model' in m and StackingEnsemble is not None:
            try:
                target_items = m.get('target_items') or self._target_items or []
                ens = StackingEnsemble(m.get('stage1_models'), m.get('stage2_model'), list(target_items))
                # キャッシュして以後はこれを使う
                self._model = ens
                return ens
            except Exception:
                pass
        # タプル/リストの中にある場合
        if isinstance(m, (tuple, list)):
            for x in m:
                if hasattr(x, 'predict'):
                    return x
        # dictのよくあるキーを探索
        if isinstance(m, dict):
            for key in (
                'stage1_model', 'model', 'estimator', 'regressor', 'pipe', 'pipeline',
                'final_model', 'sk_model'
            ):
                x = m.get(key)
                if hasattr(x, 'predict'):
                    return x
        raise ValueError('内部モデル（predict可能）が見つかりません')

    def _resolve_estimator_from(self, model_like):
        """与えられたオブジェクトから predict を持つ推論器を見つける。"""
        m = model_like
        if hasattr(m, 'predict'):
            return m
        # 新形式: {'stage1_models': {...}, 'stage2_model': {...}}
        if isinstance(m, dict) and 'stage1_models' in m and 'stage2_model' in m and StackingEnsemble is not None:
            try:
                target_items = m.get('target_items') or self._target_items or []
                return StackingEnsemble(m.get('stage1_models'), m.get('stage2_model'), list(target_items))
            except Exception:
                pass
        if isinstance(m, (tuple, list)):
            for x in m:
                if hasattr(x, 'predict'):
                    return x
        if isinstance(m, dict):
            for key in (
                'stage1_model', 'model', 'estimator', 'regressor', 'pipe', 'pipeline',
                'final_model', 'sk_model'
            ):
                x = m.get(key)
                if hasattr(x, 'predict'):
                    return x
        raise ValueError('内部モデル（predict可能）が見つかりません')

    def _infer_target_items(self, meta):
        """メタやユーティリティから品目名リストを推定。失敗時はNone。"""
        # 1) meta
        if isinstance(meta, dict):
            for k in ('target_items', 'items', 'labels'):
                v = meta.get(k)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return list(v)
        # 2) 関数があれば生データから
        try:
            if get_target_items is not None:
                v = get_target_items(self.df_raw)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return list(v)
        except Exception:
            pass
        return None

    def _estimate_item_shares(self, past_raw, target_items):
        """過去データから各品目の平均シェアを推定（合計1）。失敗時はNone。"""
        try:
            if past_raw is None or target_items is None:
                return None
            # 日毎の合計と各品目合計を計算し、平均比率をとる
            df = past_raw.copy()
            df = df[['伝票日付', '品名', '正味重量']].copy()
            df['伝票日付'] = pd.to_datetime(df['伝票日付'])
            pivot = df.groupby(['伝票日付', '品名'])['正味重量'].sum().unstack(fill_value=0)
            for item in target_items:
                if item not in pivot.columns:
                    pivot[item] = 0.0
            pivot = pivot[target_items]
            total = pivot.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                shares = (pivot.T / np.where(total.values == 0, 1, total.values)).T
            mean_shares = shares.replace([np.inf, -np.inf], 0).fillna(0).mean(axis=0)
            s = mean_shares.values
            s_sum = s.sum()
            if s_sum <= 0:
                # 全ゼロなら等分
                return np.ones(len(target_items)) / len(target_items)
            return (s / s_sum)
        except Exception:
            return None

    def predict(self, date_str):
        """指定日付で推論結果を返す（API契約に準拠）。"""
        date_str = str(date_str)[:10]
        if self._dates is None or not self._dates:
            return {"error": "モデル未推論 or 日付データなし"}
        idx = None
        for i, d in enumerate(self._dates):
            if str(d)[:10] == date_str:
                idx = i
                break
        if idx is None:
            return {"error": f"指定日付 {date_str} の予測データがありません"}
        # _pred がベクトル/行列の場合は品目別を優先
        per_item = {}
        total = None
        try:
            p = self._pred[idx]
            # pandas Series/ndarray の場合
            if hasattr(p, 'shape'):
                arr = np.asarray(p)
                if arr.ndim == 1 and self._target_items is not None and len(arr) == len(self._target_items):
                    per_item = {k: float(v) for k, v in zip(self._target_items, arr.tolist())}
                    total = float(arr.sum())
                elif arr.ndim == 0:
                    total = float(arr)
            # dictの場合
            if isinstance(p, dict):
                per_item = {str(k): float(v) for k, v in p.items()}
                total = sum(per_item.values())
        except Exception:
            pass
        # totalが未定義ならactual/predからフォールバック
        if total is None:
            try:
                total = float(self._pred[idx])
            except Exception:
                total = None
        # per_item が無ければシェアで分配
        if not per_item and total is not None and self._target_items:
            shares = self._item_shares
            if shares is None:
                shares = np.ones(len(self._target_items)) / len(self._target_items)
            per_item = {k: float(total * s) for k, s in zip(self._target_items, shares)}
        return {
            "date": date_str,
            "per_item": per_item,
            "total": float(total) if total is not None else None,
            "used_features": list(self.allowed_features) if self.allowed_features else [],
        }

    def predict_last(self):
        """最新日付の推論結果を返す。API互換用。"""
        if self._dates:
            last_date = str(self._dates[-1])[:10]
            return self.predict(last_date)
        return {"error": "モデル未推論 or 日付データなし"}

    def predict_features(self, features):
        """
        特徴量ベクトル（DataFrame, 1行）を直接入力して予測値を返す。
        - features: 1行のDataFrame もしくは dict
        - allowed_features に無いカラムは無視、足りないカラムは0埋め
        - 未来日付でも実行可能（dateは入力しない）
        """
        # allowed_features が未設定のときはエラー
        if not self.allowed_features:
            return {"error": "allowed_features が設定されていません"}
        # 特徴量をDataFrame化
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            features = features.to_frame().T
        elif not isinstance(features, pd.DataFrame):
            return {"error": f"features は dict/Series/DataFrame のいずれかが必要です: got {type(features)}"}
        # 欠損カラムを0で補完、余分なカラムは捨てる
        X_df = features.copy()
        for col in self.allowed_features:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[self.allowed_features]
        # 数値化（安全側）：非数値は0に
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        # 推論器の解決
        try:
            estimator = self._resolve_estimator()
            per_item = {}
            total = None
            # StackingEnsemble なら breakdown API を優先
            if hasattr(estimator, 'predict_with_breakdown'):
                out = estimator.predict_with_breakdown(X_df)
                per_item = {str(k): float(v) for k, v in (out.get('per_item') or {}).items()}
                total = float(out.get('total')) if out.get('total') is not None else None
            else:
                y_pred = estimator.predict(X_df.values)
                arr = np.asarray(y_pred)
                if arr.ndim == 2 and arr.shape[0] == 1 and self._target_items is not None and arr.shape[1] == len(self._target_items):
                    per_item = {k: float(v) for k, v in zip(self._target_items, arr[0].tolist())}
                    total = sum(per_item.values())
                elif arr.ndim == 1 and arr.shape[0] == 1:
                    total = float(arr[0])
                elif arr.ndim == 0:
                    total = float(arr)
            # per_itemが無ければシェアで分配
            if not per_item and total is not None and self._target_items:
                shares = self._item_shares
                if shares is None:
                    shares = np.ones(len(self._target_items)) / len(self._target_items)
                per_item = {k: float(total * s) for k, s in zip(self._target_items, shares)}
            return {
                "per_item": per_item,
                "total": float(total) if total is not None else None,
                "used_features": list(self.allowed_features) if self.allowed_features else [],
            }
        except Exception:
            # フォールバック: 単純な線形結合（総和）
            try:
                fallback_total = float(X_df.sum(axis=1).values[0])
            except Exception:
                fallback_total = 0.0
            per_item = {}
            if self._target_items:
                shares = self._item_shares
                if shares is None:
                    shares = np.ones(len(self._target_items)) / len(self._target_items)
                per_item = {k: float(fallback_total * s) for k, s in zip(self._target_items, shares)}
            return {
                "per_item": per_item,
                "total": float(fallback_total),
                "used_features": list(self.allowed_features) if self.allowed_features else [],
                "fallback_used": True,
            }

    def predict_with_features(self, date_str, features):
        """未来日付/任意日付 + ユーザー提供の一部特徴量で推論。

        仕組み:
        - 内部データから指定日付の特徴量1行を構築（ラグ等を含む）
        - ユーザー提供の features で上書き
        - StackingEnsemble があればそれで一段目→二段目を推論
        - なければ既存の predict_features にフォールバック
        """
        # 1) 特徴量行を内部生成
        try:
            base_row = self._build_df_feat_today(date_str)
        except Exception:
            base_row = None
        # 2) 上書き（数値化）
        feats = {str(k): float(v) for k, v in (features or {}).items()}
        if base_row is not None and isinstance(base_row, pd.DataFrame):
            X_df = base_row.copy()
            for k, v in feats.items():
                if k not in X_df.columns:
                    X_df[k] = 0.0
                X_df.iloc[0, X_df.columns.get_loc(k)] = float(v)
        else:
            # 内部生成できない場合は、与えられた特徴だけで推論にフォールバック
            X_df = pd.DataFrame([feats])

        # 3) 推論
        try:
            estimator = self._resolve_estimator()
            if hasattr(estimator, 'predict_with_breakdown'):
                out = estimator.predict_with_breakdown(X_df)
                used = list(X_df.columns)
                return {
                    "date": str(date_str)[:10],
                    "per_item": {str(k): float(v) for k, v in (out.get('per_item') or {}).items()},
                    "total": float(out.get('total')) if out.get('total') is not None else None,
                    "used_features": used,
                }
        except Exception:
            pass
        # フォールバック: 既存の predict_features を使用
        out = self.predict_features(X_df)
        if 'error' in out:
            return out
        return {
            "date": str(date_str)[:10],
            "per_item": out.get('per_item', {}),
            "total": out.get('total'),
            "used_features": out.get('used_features', list(X_df.columns)),
        }

# --- pickle再生成用サンプルコード ---
if __name__ == "__main__":
    import pickle
    # 必要なデータをロード（例: CSV, JSON, pkl など）
    df_recent = pd.DataFrame()
    df_reserve_recent = pd.DataFrame()
    holidays = []
    df_weather = pd.DataFrame()
    allowed_features = ["feature1", "feature2", "feature3"]
    top_n = 10
    min_stage1_days = 30
    predictor = NewModel2Predictor(
        df_raw=df_recent,
        df_reserve=df_reserve_recent,
        holidays=holidays,
        df_weather=df_weather,
        allowed_features=allowed_features,
        top_n=top_n,
        min_stage1_days=min_stage1_days
    )
    with open("/works/data/final_stage1_model_api.pkl", "wb") as f:
        pickle.dump(predictor, f)
    print("[SAVE][API] /works/data/final_stage1_model_api.pkl saved")
