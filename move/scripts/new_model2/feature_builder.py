import pandas as pd
import numpy as np


# 完成版: API納品用モデル
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
    def __init__(self, df_raw, df_reserve, holidays, df_weather, top_n=5, min_stage1_days=30, min_stage2_days=30, allowed_features=None):
        self.df_raw = df_raw
        self.df_reserve = df_reserve
        self.holidays = holidays
        self.df_weather = df_weather
        self.top_n = top_n
        self.min_stage1_days = min_stage1_days
        self.min_stage2_days = min_stage2_days
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
            y_pred = estimator.predict(X_df.values)
            arr = np.asarray(y_pred)
            # per-item出力 or total出力を判定
            per_item = {}
            total = None
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
        """
        未来日付を含む任意日付と特徴量で推論するヘルパー。
        APIの契約に近い形の辞書を返す。
        """
        out = self.predict_features(features)
        if 'error' in out:
            return out
        return {
            "date": str(date_str)[:10],
            "per_item": out.get('per_item', {}),
            "total": out.get('total'),
            "used_features": out.get('used_features', list(self.allowed_features) if self.allowed_features else []),
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
