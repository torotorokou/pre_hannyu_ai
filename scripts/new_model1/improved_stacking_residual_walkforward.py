import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
import warnings
import re
import time
import inspect
from typing import Any, Dict, List, Tuple, Optional, Iterable, Sequence
from dataclasses import dataclass, asdict
from datetime import timedelta

# --- External Library Imports ---
import sklearn
import numpy
import pandas

from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

# --- Optional Dependencies Check ---
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import catboost
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    import jpholiday
    _HAS_JPH = True
except ImportError:
    _HAS_JPH = False

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

warnings.filterwarnings("once", category=ConvergenceWarning)

# =====================================================
# 評価指標 (Metrics)
# =====================================================

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(valid_idx) < 2:
            return float("nan")
        return float(r2_score(y_true[valid_idx], y_pred[valid_idx]))
    except Exception:
        return float("nan")

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(valid_idx) == 0:
            return float("nan")
        return float(mean_absolute_error(y_true[valid_idx], y_pred[valid_idx]))
    except Exception:
        return float("nan")

def bootstrap_mae_diff_ci(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    diffs = []
    valid_idx = np.isfinite(y_true) & np.isfinite(y_a) & np.isfinite(y_b)
    y_true, y_a, y_b = y_true[valid_idx], y_a[valid_idx], y_b[valid_idx]
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        mae_a = mean_absolute_error(y_true[idx], y_a[idx])
        mae_b = mean_absolute_error(y_true[idx], y_b[idx])
        diffs.append(mae_a - mae_b)
    diffs = np.array(diffs)
    mean_diff = float(diffs.mean())
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, float(ci_low), float(ci_high)

# =====================================================
# 設定クラス (Configuration)
# =====================================================

@dataclass
class ColumnMapping:
    raw_date: str = "伝票日付"
    raw_item: str = "品名"
    raw_weight: str = "正味重量"
    reserve_date: str = "予約日"
    reserve_count: str = "予約台数"
    reserve_fixed: str = "固定客"
    weather_date: str = "日付"

@dataclass
class WFConfig:
    colmap: ColumnMapping = ColumnMapping()
    n_splits: int = 5
    gap: int = 1
    max_train_window_days: Optional[int] = 365
    min_train_days: int = 120
    target_mode: str = "log1p"
    use_same_day_info: bool = False
    time_decay: Optional[str] = "exponential"
    residual_model: str = "hgbr"
    residual_gbr_params: Dict[str, Any] = None
    residual_hgbr_params: Dict[str, Any] = None
    residual_lgbm_params: Dict[str, Any] = None
    residual_cat_params: Dict[str, Any] = None
    residual_clip_quantile: float = 0.99
    residual_min_days: int = 60
    residual_cv_guard: bool = True
    residual_alpha: float = 0.55
    meta_model: str = "ridge"
    calibration_window_days: int = 28
    seed: int = 42
    n_jobs: int = -1
    print_progress_every: int = 30
    top_n_items: Optional[int] = None
    top_rank_by: str = "weight_sum"
    top_window_days: Optional[int] = 180
    explicit_target_items: Optional[List[str]] = None
    window_bagging_days: Optional[List[int]] = None
    use_reserve: bool = True
    use_weather: bool = True
    # --- Checkpoint / Resume ---
    resume: bool = False
    checkpoint_dir: Optional[str] = None  # if None -> <out-dir>/.checkpoints
    checkpoint_every: int = 5  # save every N prediction days

    def __post_init__(self):
        self.residual_gbr_params = self.residual_gbr_params or dict(n_estimators=150, learning_rate=0.05, max_depth=3)
        self.residual_hgbr_params = self.residual_hgbr_params or dict(max_iter=300, learning_rate=0.05, max_depth=3)
        self.residual_lgbm_params = self.residual_lgbm_params or dict(n_estimators=400, learning_rate=0.03, num_leaves=32, feature_fraction=0.9)
        self.residual_cat_params = self.residual_cat_params or dict(iterations=400, learning_rate=0.05, depth=4, verbose=0)
        
        try:
            q = float(self.residual_clip_quantile)
            if not (0.5 < q <= 1.0): self.residual_clip_quantile = 0.99
        except Exception:
            self.residual_clip_quantile = 0.99
        
        if self.window_bagging_days is None:
            if self.max_train_window_days:
                self.window_bagging_days = [self.max_train_window_days]
            else:
                self.window_bagging_days = [120, 240, 365]


# =====================================================
# 共通ユーティリティ (General Utilities)
# =====================================================

def set_global_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def fillna_with_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaNs with 0 and adds a binary indicator column for each original column with NaNs."""
    df = df.copy()
    na_cols = df.columns[df.isna().any()].tolist()
    if not na_cols:
        return df
    na_flags = {f"{c}_na": df[c].isna().astype(int) for c in na_cols}
    df = df.fillna(0.0)
    df = df.join(pd.DataFrame(na_flags, index=df.index))
    return df

def _ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path

def _resolve_checkpoint_dir(cfg: WFConfig) -> str:
    base = os.getcwd()
    ck = cfg.checkpoint_dir
    if ck is None:
        ck = os.path.join(base, ".checkpoints")
    if not os.path.isabs(ck):
        ck = os.path.join(base, ck)
    return _ensure_dir(ck)

def _load_checkpoint(ck_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(ck_dir, "results.csv")
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            rows.append(r.to_dict())
        print(f"[CHECKPOINT] Loaded {len(rows)} rows from {path}")
        return rows
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {e}")
        return []

def _save_checkpoint(ck_dir: str, results: List[Dict[str, Any]], tag: str = "progress") -> None:
    if not results:
        return
    try:
        df = pd.DataFrame(results)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(os.path.join(ck_dir, "results.csv"), index=False)
        meta = {
            "tag": tag,
            "n_rows": int(len(df)),
            "last_date": (pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d") if "date" in df.columns else None),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(ck_dir, "progress.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[CHECKPOINT] Saved {len(df)} rows -> {ck_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")

def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts DataFrame to float32 to save memory."""
    return df.astype(np.float32, copy=False)

def _clean_date_string(s: Any) -> str:
    """Cleans a date string by removing extraneous characters like weekday names."""
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"[\(（][^\)）]*[\)）]", "", s)
    s = s.replace("年", "/").replace("月", "/").replace("日", "")
    s = s.replace("-", "/")
    s = s.strip()
    return s

def preprocess_raw_df(df: pd.DataFrame, date_col: str, item_col: str, weight_col: str) -> pd.DataFrame:
    """Preprocesses raw data for consistent column types and formats."""
    df = df.copy()
    if item_col not in df.columns and '商品' in df.columns:
        df.rename(columns={'商品': item_col}, inplace=True)
    cols = [date_col, item_col, weight_col]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in raw data: {missing_cols}")
    df = df[cols].copy()
    df[date_col] = df[date_col].apply(_clean_date_string)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[weight_col, date_col]).copy()
    if len(df) == 0:
        raise ValueError("Preprocessed raw data is empty.")
    return df

def preprocess_reserve_df(df: pd.DataFrame, date_col: str, count_col: str, fixed_col: str) -> pd.DataFrame:
    """Preprocesses reservation data for consistent column types and formats."""
    df = df.copy()
    if count_col not in df.columns and '台数' in df.columns:
        df.rename(columns={'台数': count_col}, inplace=True)
    if date_col in df.columns:
        df[date_col] = df[date_col].apply(_clean_date_string)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    else:
        df[date_col] = pd.to_datetime(df.index, errors="coerce").dt.normalize()
    if df[date_col].isna().all():
        raise ValueError("Failed to convert reservation date column.")
    if count_col in df.columns:
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    if len(df) == 0:
        warnings.warn("Preprocessed reservation data is empty.")
        return pd.DataFrame(columns=[date_col, count_col, fixed_col])
    return df

# =====================================================
# 特徴量エンジニアリング (Feature Engineering)
# =====================================================

def get_target_items(df_raw: pd.DataFrame, cfg: WFConfig) -> List[str]:
    """Filters target items based on a predefined list or recent activity."""
    cm = cfg.colmap
    if cfg.explicit_target_items and isinstance(cfg.explicit_target_items, list):
        return [item for item in cfg.explicit_target_items if item in df_raw[cm.raw_item].unique()]

    min_days = 10
    lookback_days = cfg.top_window_days or 365
    latest_date = pd.to_datetime(df_raw[cm.raw_date]).max()
    cutoff_date = latest_date - pd.Timedelta(days=lookback_days)
    df_recent = df_raw[pd.to_datetime(df_raw[cm.raw_date]) >= cutoff_date]
    item_counts = df_recent.groupby(cm.raw_item)[cm.raw_date].nunique()
    target_items = item_counts[item_counts >= min_days].index.tolist()

    if cfg.top_n_items:
        wsum = (df_raw[pd.to_datetime(df_raw[cm.raw_date]) >= cutoff_date]
                .groupby(cm.raw_item)[cm.raw_weight].sum().sort_values(ascending=False))
        target_items = [i for i in wsum.index if i in target_items][:cfg.top_n_items]

    return target_items

def generate_holidays_index(start: pd.Timestamp, end: pd.Timestamp, holidays: Optional[Iterable]) -> pd.DatetimeIndex:
    """Generates a DatetimeIndex for holidays."""
    all_days = pd.date_range(start, end, freq="D")
    if holidays is not None and len(holidays) > 0:
        hol = pd.to_datetime(list(holidays))
        return pd.DatetimeIndex(hol).intersection(all_days)
    if _HAS_JPH:
        hol = [d for d in all_days if jpholiday.is_holiday(d)]
        return pd.DatetimeIndex(hol)
    hol = [d for d in all_days if d.weekday() >= 5]
    return pd.DatetimeIndex(hol)

def build_calendar_features(index: pd.DatetimeIndex, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds calendar-based features with cyclical encoding."""
    df = pd.DataFrame(index=index)
    df["dow"] = index.weekday
    df["weekofyear"] = index.isocalendar().week.astype(int)
    df["month"] = index.month
    df["day"] = index.day
    df["is_weekend"] = (index.weekday >= 5).astype(int)
    df["is_holiday"] = index.isin(holidays).astype(int)
    df["dayofyear"] = index.dayofyear
    
    hol_plus_minus_2d = holidays.union(holidays + timedelta(days=1)).union(holidays + timedelta(days=2)).union(holidays - timedelta(days=1)).union(holidays - timedelta(days=2))
    df["is_holiday_nearby"] = index.isin(hol_plus_minus_2d).astype(int)

    def _cyc_encode(vals, period, prefix):
        ang = 2 * np.pi * vals / period
        return pd.DataFrame({f"{prefix}_sin": np.sin(ang), f"{prefix}_cos": np.cos(ang)}, index=index)

    df = df.join(_cyc_encode(df["dow"], 7, "dow"))
    df = df.join(_cyc_encode(df["dayofyear"], 365.25, "dayofyear"))
    df = df.join(_cyc_encode(df["month"], 12, "month"))
    return df

def aggregate_reserve(df_reserve: Optional[pd.DataFrame], cfg: WFConfig, index: pd.DatetimeIndex, shift_one_day: bool) -> pd.DataFrame:
    """Aggregates reservation data into daily features."""
    feats = pd.DataFrame(index=index)
    if df_reserve is None or len(df_reserve) == 0:
        feats[["reserve_count", "reserve_sum", "fixed_ratio"]] = 0.0
        return feats
    cm = cfg.colmap
    df = df_reserve.copy()
    df[cm.reserve_date] = pd.to_datetime(df[cm.reserve_date]).dt.normalize()
    df = df.sort_values(cm.reserve_date)
    grp = df.groupby(df[cm.reserve_date])
    reserve_count = grp.size().rename("reserve_count").astype(float)
    if cm.reserve_count in df.columns:
        reserve_sum = grp[cm.reserve_count].sum().rename("reserve_sum").astype(float)
    else:
        reserve_sum = reserve_count.rename("reserve_sum")
    if cm.reserve_fixed in df.columns:
        df[cm.reserve_fixed] = df[cm.reserve_fixed].apply(lambda x: 1.0 if str(x).lower() in ["1", "true", "yes", "fixed"] else 0.0)
        fixed_ratio = grp[cm.reserve_fixed].mean().rename("fixed_ratio").astype(float)
    else:
        fixed_ratio = pd.Series(0.0, index=reserve_count.index, name="fixed_ratio")
    agg = pd.concat([reserve_count, reserve_sum, fixed_ratio], axis=1)
    agg = agg.reindex(index, fill_value=0.0)
    if shift_one_day:
        agg = agg.shift(1).fillna(0.0)
    return agg

def encode_weather(df_weather: Optional[pd.DataFrame], cfg: WFConfig, index: pd.DatetimeIndex, shift_one_day: bool) -> pd.DataFrame:
    """Encodes weather data into numerical and categorical features."""
    if df_weather is None or len(df_weather) == 0:
        return pd.DataFrame(index=index)
    cm = cfg.colmap
    dfw = df_weather.copy()
    if cm.weather_date not in dfw.columns:
        if isinstance(dfw.index, pd.DatetimeIndex):
            dfw[cm.weather_date] = dfw.index
        else:
            return pd.DataFrame(index=index)
    dfw[cm.weather_date] = pd.to_datetime(dfw[cm.weather_date]).dt.normalize()
    dfw = dfw.sort_values(cm.weather_date)
    dfw = dfw.set_index(cm.weather_date)
    dfw = dfw.reindex(index)
    if shift_one_day:
        dfw = dfw.shift(1)
    num_cols = [c for c in dfw.columns if pd.api.types.is_numeric_dtype(dfw[c])]
    cat_cols = [c for c in dfw.columns if c not in num_cols and c != cm.weather_date]
    out = pd.DataFrame(index=index)
    if num_cols:
        out = out.join(dfw[num_cols])
    for c in cat_cols:
        vc = dfw[c].nunique(dropna=False)
        if vc <= 15:
            dummies = pd.get_dummies(dfw[c].astype("category"), prefix=f"w_{c}", dummy_na=True)
            out = out.join(dummies)
    out = out.fillna(0.0)
    try:
        cols_lower = {c.lower(): c for c in out.columns}
        low_keys = ("rain", "precip", "precipitation")
        jp_keys  = ("雨", "降水", "降雨")
        rain_like_cols = [orig for low, orig in cols_lower.items() if any(k in low for k in low_keys)]
        rain_like_cols += [c for c in out.columns if any(k in c for k in jp_keys)]
        rain_like_cols = list(dict.fromkeys(rain_like_cols))
        
        if rain_like_cols:
            sub = out[rain_like_cols].copy()
            vals = (sub.values > 0).astype(int)
            w_is_rain = pd.Series(vals.max(axis=1), index=sub.index, name="w_is_rain").astype(int)
        else:
            w_is_rain = pd.Series(0, index=out.index, name="w_is_rain")
        out = out.join(w_is_rain)
    except Exception:
        pass
    return out

def build_item_series(df_raw: pd.DataFrame, cfg: WFConfig) -> Tuple[pd.DatetimeIndex, List[str], pd.DataFrame]:
    """Pivots raw data into a date x item DataFrame."""
    cm = cfg.colmap
    dfr = df_raw[[cm.raw_date, cm.raw_item, cm.raw_weight]].copy()
    dfr[cm.raw_date] = pd.to_datetime(dfr[cm.raw_date]).dt.normalize()
    dfr[cm.raw_weight] = pd.to_numeric(dfr[cm.raw_weight], errors="coerce")
    dfr = dfr.dropna(subset=[cm.raw_weight])
    pvt = dfr.groupby([cm.raw_date, cm.raw_item])[cm.raw_weight].sum().unstack(fill_value=0.0)
    pvt = pvt.sort_index()
    if len(pvt.index) > 0:
        full_idx = pd.date_range(pvt.index.min(), pvt.index.max(), freq="D")
        pvt = pvt.reindex(full_idx, fill_value=0.0)
    all_dates = pvt.index
    items = list(pvt.columns)
    return all_dates, items, pvt

def build_item_features(Y_pivot: pd.DataFrame, exog: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generates lag, rolling mean, and rolling std features for each item."""
    lags = (1, 2, 3, 7, 14, 28, 56, 90)
    mas = (3, 7, 14, 28, 90)
    stds = (7, 14, 28)
    
    results: Dict[str, pd.DataFrame] = {}
    for item in Y_pivot.columns:
        s = Y_pivot[item].astype(float)
        df = exog.copy()
        
        for L in lags:
            df[f"{item}_lag{L}"] = s.shift(L)
        for W in mas:
            df[f"{item}_ma{W}"] = s.rolling(W, min_periods=1).mean().shift(1)
        for W in stds:
            df[f"{item}_std{W}"] = s.rolling(W, min_periods=2).std().shift(1)
            
        if 1 in lags:
            if 7 in mas:
                df[f"{item}_dev_ma7"] = df.get(f"{item}_lag1", s.shift(1)) - df.get(f"{item}_ma7", s.rolling(7, min_periods=1).mean().shift(1))
            if 28 in mas:
                df[f"{item}_dev_ma28"] = df.get(f"{item}_lag1", s.shift(1)) - df.get(f"{item}_ma28", s.rolling(28, min_periods=1).mean().shift(1))
        
        if 7 in mas and 14 in mas:
            df[f"{item}_ma7-ma14"] = df.get(f"{item}_ma7", 0) - df.get(f"{item}_ma14", 0)
        df["y"] = s
        results[item] = df
    return results

def _clip_by_history(series: pd.Series, pred: float, win: int = 180, ql: float = 0.02, qh: float = 0.98) -> float:
    """Clips a prediction based on historical quantiles to prevent outliers."""
    h = series.tail(win)
    if len(h) < 20: return float(max(0.0, pred))
    lo, hi = np.quantile(h.values, [ql, qh])
    return float(np.clip(pred, max(0.0, lo), max(lo, hi)))

def _apply_ratio_guard(per_item_pred: Dict[str, float], Y_pivot: pd.DataFrame, pred_day, share_win: int = 56, lam: float = 0.2) -> Dict[str, float]:
    """Adjusts item predictions based on recent item share to maintain composition stability."""
    prev_days = Y_pivot.index[(Y_pivot.index < pred_day) & (Y_pivot.index >= pred_day - pd.Timedelta(days=share_win))]
    if len(prev_days) < 10: return per_item_pred
    recent = Y_pivot.loc[prev_days]
    total = recent.sum(axis=1).replace(0, np.nan)
    shares = (recent.T / total).T.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean_share = shares.mean(axis=0)
    s = {k: float(mean_share.get(k, 0.0)) for k in per_item_pred.keys()}
    S = sum(per_item_pred.values()) + 1e-9
    adjusted = {k: lam * per_item_pred[k] + (1 - lam) * s[k] * S for k in per_item_pred.keys()}
    return adjusted


# =====================================================
# モデル定義 (Model Specifications)
# =====================================================

def _base_model_specs(cfg: WFConfig) -> List[Tuple[str, Any, bool]]:
    """Defines the base models for stacking."""
    seed = cfg.seed
    models: List[Tuple[str, Any, bool]] = []
    
    models.append(("elastic", ElasticNet(alpha=0.02, l1_ratio=0.3, max_iter=20000, positive=True, random_state=seed), True))
    models.append(("extratrees", ExtraTreesRegressor(n_estimators=500, n_jobs=cfg.n_jobs, random_state=seed, min_samples_leaf=5, bootstrap=False), False))
    models.append(("gbr", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed), False))
    models.append(("hgbr", HistGradientBoostingRegressor(random_state=seed, max_iter=400, learning_rate=0.05, max_depth=3), False))
    
    if _HAS_LGBM:
        models.append(("lgbm", lgb.LGBMRegressor(random_state=seed, n_estimators=600, learning_rate=0.03, num_leaves=32, n_jobs=cfg.n_jobs, feature_fraction=0.9, verbosity=-1), False))
    if _HAS_CAT:
        cat_params = dict(random_seed=seed, iterations=500, learning_rate=0.05, depth=4, verbose=0, allow_writing_files=False)
        if isinstance(cfg.n_jobs, int) and cfg.n_jobs > 0:
            cat_params["thread_count"] = int(cfg.n_jobs)
        models.append(("catboost", CatBoostRegressor(**cat_params), False))
    
    return models


# =====================================================
# パイプラインヘルパー (Pipeline Helpers)
# =====================================================

def _build_preprocessor(needs_scaling: bool) -> List[Tuple[str, Any]]:
    steps: List[Tuple[str, Any]] = []
    if needs_scaling: steps.append(("scaler", StandardScaler()))
    steps.append(("varth", VarianceThreshold(1e-10)))
    return steps

def _fit_transform(steps: List[Tuple[str, Any]], X_tr: pd.DataFrame) -> Tuple[List[Tuple[str, Any]], np.ndarray]:
    Xt = X_tr.values
    fitted: List[Tuple[str, Any]] = []
    for name, t in steps:
        tt = clone(t)
        Xt = tt.fit_transform(Xt)
        fitted.append((name, tt))
    return fitted, Xt

def _transform(fitted_steps: List[Tuple[str, Any]], X: pd.DataFrame) -> np.ndarray:
    Xt = X.values
    for _, tt in fitted_steps:
        Xt = tt.transform(Xt)
    return Xt

def _supports_sample_weight(est: Any) -> bool:
    try:
        if isinstance(est, Pipeline):
            last = est.steps[-1][1]
            sig = inspect.signature(last.fit)
        else:
            sig = inspect.signature(est.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def apply_time_decay_weights(n: int, decay: Optional[str]) -> Optional[np.ndarray]:
    if decay is None or decay.lower() == "none":
        return None
    t = np.arange(n) / max(1, n - 1)
    if decay == "linear":
        w = t
    elif decay == "exponential":
        w = np.exp(t * 3)
    else:
        return None
    return w / np.sum(w) if np.sum(w) > 0 else w

def make_time_series_folds(n_samples: int, n_splits: int, gap: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2 or n_samples < n_splits * 2 + gap:
        warnings.warn("Insufficient samples for TimeSeriesSplit. Falling back to a single 80/20 train/val split.")
        tr_idx = np.arange(0, int(0.8 * n_samples))
        va_idx = np.arange(int(0.8 * n_samples) + gap, n_samples)
        if len(va_idx) == 0:
            va_idx = np.arange(n_samples-1, n_samples)
            tr_idx = np.arange(0, n_samples-1)
        if len(tr_idx) == 0: return []
        return [(tr_idx, va_idx)]
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    except TypeError:
        warnings.warn("`gap` not supported by this scikit-learn version. Falling back to no gap.")
        tscv = TimeSeriesSplit(n_splits=n_splits)

    for tr, va in tscv.split(range(n_samples)):
        folds.append((tr, va))
    return folds

def time_series_oof_predictions(X: pd.DataFrame, y: pd.Series, base_models: List[Tuple[str, Any, bool]], cfg: WFConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates Out-of-Fold (OOF) predictions for stacking and full models."""
    n = len(X)
    folds = make_time_series_folds(n, cfg.n_splits, cfg.gap)
    oof = pd.DataFrame(index=X.index, columns=[name for name, _, _ in base_models], dtype=float)
    fitted_full: Dict[str, Any] = {}

    for name, est, needs_scaling in base_models:
        val_maes, naive_maes = [], []
        y_train_unique_count = y.nunique()
        if y_train_unique_count <= 1:
            warnings.warn(f"All targets are equal for {name}. Skipping.")
            oof[name] = np.nan
            fitted_full[name] = {"steps": [], "model": DummyRegressor(strategy="mean")}
            continue
        
        for fold_idx, (tr, va) in enumerate(folds):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_va, y_va = X.iloc[va], y.iloc[va]
            if len(X_tr) < 30: continue

            naive_pred_va = np.full_like(y_va.values, fill_value=float(y_tr.iloc[-1]))
            naive_maes.append(mae(y_va.values, naive_pred_va))
            
            if y_tr.nunique() <= 1:
                model = DummyRegressor(strategy="mean"); model.fit(X_tr.values, y_tr.values)
                pred = model.predict(X_va.values)
            else:
                try:
                    if "lgbm" in name or "catboost" in name:
                        model = clone(est)
                        sw = apply_time_decay_weights(len(X_tr), cfg.time_decay)
                        if "lgbm" in name:
                            try:
                                model.fit(
                                    X_tr, y_tr.values,
                                    sample_weight=sw if _supports_sample_weight(model) and sw is not None else None,
                                    eval_set=[(X_va, y_va.values)],
                                    eval_metric="l1",
                                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                                )
                            except TypeError:
                                model.fit(
                                    X_tr, y_tr.values,
                                    sample_weight=sw if _supports_sample_weight(model) and sw is not None else None,
                                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                                )
                        else:  # catboost
                            model.fit(
                                X_tr, y_tr.values,
                                sample_weight=sw if sw is not None else None,
                                eval_set=(X_va, y_va.values),
                                use_best_model=True, verbose=False
                            )
                        pred = model.predict(X_va)
                    else:
                        steps = _build_preprocessor(needs_scaling)
                        Xtr_filled = fillna_with_indicator(_to_float32(X_tr))
                        fitted_steps, Xt_tr = _fit_transform(steps, Xtr_filled)
                        Xva_filled = fillna_with_indicator(_to_float32(X_va)).reindex(
                            columns=Xtr_filled.columns, fill_value=0.0
                        )
                        Xt_va = _transform(fitted_steps, Xva_filled)
                        model = clone(est)
                        sw = apply_time_decay_weights(len(X_tr), cfg.time_decay)
                        if _supports_sample_weight(model) and sw is not None: model.fit(Xt_tr, y_tr.values, sample_weight=sw)
                        else: model.fit(Xt_tr, y_tr.values)
                        pred = model.predict(Xt_va)
                except Exception as e:
                    warnings.warn(f"Fold fit failed for {name} due to {e}. Falling back to DummyRegressor.")
                    model = DummyRegressor(strategy="mean"); model.fit(X_tr, y_tr)
                    pred = model.predict(X_va)
            
            oof.loc[X_va.index, name] = pred
            val_maes.append(mae(y_va.values, pred))
            print(f"[FOLD-VAL] {name} fold={fold_idx+1}/{len(folds)} valMAE={val_maes[-1]:.3f} (vs naive_lag1 {naive_maes[-1]:.3f})")
        
        if not val_maes:
            print(f"[DROP] {name} -> no valid folds")
            oof[name] = np.nan
            continue
        
        mean_val_mae, mean_naive_mae = np.nanmean(val_maes), np.nanmean(naive_maes)
        print(f"[OOF] {name} mean_valMAE={mean_val_mae:.3f} vs mean_naiveMAE={mean_naive_mae:.3f}")
        
        if mean_val_mae >= mean_naive_mae:
            print(f"[DROP] {name} -> exclude from meta (worse than naive)")
            oof[name] = np.nan
        else:
            sw_full = apply_time_decay_weights(len(X), cfg.time_decay)
            if "lgbm" in name or "catboost" in name:
                model = clone(est)
                if _supports_sample_weight(model) and sw_full is not None:
                    model.fit(X, y.values, sample_weight=sw_full)
                else:
                    model.fit(X, y.values)
                fitted_full[name] = {"steps": [], "model": model, "cols": list(X.columns)}
            else:
                steps = _build_preprocessor(needs_scaling)
                Xfull_filled = fillna_with_indicator(_to_float32(X))
                fitted_steps, Xt_full = _fit_transform(steps, Xfull_filled)
                model = clone(est)
                if _supports_sample_weight(model) and sw_full is not None:
                    model.fit(Xt_full, y.values, sample_weight=sw_full)
                else:
                    model.fit(Xt_full, y.values)
                fitted_full[name] = {"steps": fitted_steps, "model": model, "cols": list(Xfull_filled.columns)}
    
    valid_oof_cols = oof.columns[~oof.isna().all()].tolist()
    oof = oof[valid_oof_cols]
    return oof, fitted_full


def fit_meta_from_oof(oof: pd.DataFrame, y: pd.Series, cfg: WFConfig) -> Tuple[Any, str, float]:
    """Fits the meta-model on OOF predictions."""
    if oof is None or oof.shape[1] == 0:
        warnings.warn("No usable base models (OOF has 0 columns). Skipping meta.")
        return None, "none", float("nan")

    valid = oof.dropna(how='any')
    yv = y.loc[valid.index]
    if len(valid) < 30:
        warnings.warn("Not enough OOF data to train meta-model. Falling back to simple Ridge.")
        model = Ridge(alpha=0.5)
        model.fit(oof.fillna(0.0).values, y.values)
        return model, "fallback_ridge", r2(y.values, model.predict(oof.fillna(0.0).values))

    if cfg.meta_model == "ridge":
        model = Ridge(alpha=0.5)
        model.fit(valid.values, yv.values)
        return model, "ridge", r2(yv.values, model.predict(valid.values))
    elif cfg.meta_model == "elastic":
        model = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=cfg.seed)
        model.fit(valid.values, yv.values)
        return model, "elastic", r2(yv.values, model.predict(valid.values))

    candidates: List[Tuple[str, Any]] = [
        ("ridge", Ridge(alpha=0.5)),
        ("elastic", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=cfg.seed)),
    ]
    best: Tuple[Any, str, float] = (None, "", -1e9)
    for name, model in candidates:
        try:
            model.fit(valid.values, yv.values)
            score = r2(yv.values, model.predict(valid.values))
        except Exception:
            score = -1e9
        if score > best[2]:
            best = (model, name, score)
    if best[0] is None:
        warnings.warn("All meta-model candidates failed. Falling back to Ridge.")
        model = Ridge(alpha=0.5)
        model.fit(valid.values, yv.values)
        score = r2(yv.values, model.predict(valid.values))
        return model, "fallback_ridge", score
        
    if best[2] is not None and best[2] < 0:
        warn_msg = f"[META] OOF R2 < 0 ({best[2]:.3f}). Falling back to ridge."
        warnings.warn(warn_msg)
        fb = Ridge(alpha=0.5)
        fb.fit(valid.values, yv.values)
        return fb, "ridge_fallback", r2(yv.values, fb.predict(valid.values))

    return best

def meta_predict_from_full(X_today: pd.DataFrame, fitted_full: Dict[str, Any], meta_model: Any, used_base_names: List[str]) -> float:
    """Makes a prediction for today using the full-fitted models and meta-model."""
    if meta_model is None: return 0.0
    names = [n for n in used_base_names if n in fitted_full]
    if not names: return 0.0
    preds = []
    for name in names:
        obj = fitted_full.get(name)
        if obj is None: preds.append(np.nan); continue
        steps, model = obj["steps"], obj["model"]
        cols = obj.get("cols")
        try:
            if "lgbm" in name or "catboost" in name:
                Xt_df = X_today.reindex(columns=cols, fill_value=0.0) if cols is not None else X_today
                Xt = Xt_df
            else:
                Xtod_filled = fillna_with_indicator(_to_float32(X_today))
                if cols is not None:
                    Xtod_filled = Xtod_filled.reindex(columns=cols, fill_value=0.0)
                Xt = _transform(steps, Xtod_filled)
            preds.append(float(np.ravel(model.predict(Xt))[0]))
        except Exception:
            warnings.warn(f"Prediction failed for base model {name}.")
            preds.append(np.nan)
    M = np.nan_to_num(np.array(preds, dtype=float).reshape(1, -1))
    if M.shape[1] == 0: return 0.0
    yhat = float(np.ravel(meta_model.predict(M))[0])
    return max(0.0, yhat)

# =====================================================
# 残差モデル学習 (Residual Learning)
# =====================================================

def fit_residual_model(X_res: pd.DataFrame, y_res: pd.Series, cfg: WFConfig) -> Any:
    """Fits a residual model to correct for overall prediction bias."""
    if cfg.residual_model == "none" or len(y_res) < cfg.residual_min_days: return None
    
    X_res_f = _to_float32(fillna_with_indicator(X_res))
    y_res_f = y_res.astype(np.float32, copy=False)
    
    if cfg.residual_model == "lgbm" and _HAS_LGBM:
        model = lgb.LGBMRegressor(random_state=cfg.seed, **cfg.residual_lgbm_params, n_jobs=cfg.n_jobs)
        sw = apply_time_decay_weights(len(X_res_f), cfg.time_decay)
        if len(X_res_f) > 100:
            split = int(len(X_res_f) * 0.8)
            X_tr, X_va = X_res_f.iloc[:split], X_res_f.iloc[split:]
            y_tr, y_va = y_res_f.iloc[:split], y_res_f.iloc[split:]
            sw_tr = None if sw is None else sw[:split]
            try:
                model.fit(
                    X_tr, y_tr,
                    sample_weight=sw_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            except TypeError:
                model.fit(
                    X_tr, y_tr,
                    sample_weight=sw_tr,
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
        else:
            model.fit(X_res_f, y_res_f, sample_weight=sw)
        setattr(model, "_wf_cols", list(X_res_f.columns))
        return model
    
    if cfg.residual_model == "catboost" and _HAS_CAT:
        cat_params = dict(random_seed=cfg.seed, **cfg.residual_cat_params)
        if isinstance(cfg.n_jobs, int) and cfg.n_jobs > 0:
            cat_params["thread_count"] = int(cfg.n_jobs)
        model = CatBoostRegressor(**cat_params)
        sw = apply_time_decay_weights(len(X_res_f), cfg.time_decay)
        if len(X_res_f) > 100:
            split = int(len(X_res_f) * 0.8)
            X_tr, X_va = X_res_f.iloc[:split], X_res_f.iloc[split:]
            y_tr, y_va = y_res_f.iloc[:split], y_res_f.iloc[split:]
            sw_tr = None if sw is None else sw[:split]
            model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_va, y_va), verbose=False, use_best_model=True)
        else:
            model.fit(X_res_f, y_res_f, sample_weight=sw)
        setattr(model, "_wf_cols", list(X_res_f.columns))
        return model
    
    model = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("varth", VarianceThreshold(1e-10)), ("ridge", Ridge(alpha=1.0))]),
        "hgbr": HistGradientBoostingRegressor(random_state=cfg.seed, **cfg.residual_hgbr_params),
        "gbr": GradientBoostingRegressor(random_state=cfg.seed, **cfg.residual_gbr_params)
    }.get(cfg.residual_model, HistGradientBoostingRegressor(random_state=cfg.seed, max_iter=300, learning_rate=0.05, max_depth=3))
    
    sw = apply_time_decay_weights(len(X_res_f), cfg.time_decay)
    try:
        if isinstance(model, Pipeline) and sw is not None:
            last_name = model.steps[-1][0]
            model.fit(X_res_f, y_res_f, **{f"{last_name}__sample_weight": sw})
        elif _supports_sample_weight(model) and sw is not None:
            model.fit(X_res_f, y_res_f, sample_weight=sw)
        else:
            model.fit(X_res_f, y_res_f)
    except Exception:
        model.fit(X_res_f, y_res_f)
    setattr(model, "_wf_cols", list(X_res_f.columns))
    return model

def _predict_residual(model: Any, X: pd.DataFrame, cfg: WFConfig) -> np.ndarray:
    """Helper to predict residuals."""
    try:
        is_lgbm = (_HAS_LGBM and isinstance(model, lgb.LGBMRegressor))
        is_cat  = (_HAS_CAT and isinstance(model, CatBoostRegressor))
        cols = getattr(model, "_wf_cols", None)
        if is_lgbm or is_cat:
            X_use = X.reindex(columns=cols, fill_value=0.0) if cols is not None else X
            pred = model.predict(X_use)
        else:
            X_use = fillna_with_indicator(_to_float32(X))
            if cols is not None:
                X_use = X_use.reindex(columns=cols, fill_value=0.0)
            pred = model.predict(X_use)
        return np.ravel(pred)
    except Exception:
        warnings.warn("Residual model prediction failed.")
        return np.zeros(len(X), dtype=float)


# =====================================================
# メインパイプライン (Main Pipeline)
# =====================================================
def _prepare_exogenous(df_raw: pd.DataFrame, holidays: Optional[Iterable], df_reserve: Optional[pd.DataFrame], df_weather: Optional[pd.DataFrame], cfg: WFConfig) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series, pd.Series, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Prepares exogenous and item-specific features."""
    all_dates, items, Y_pivot = build_item_series(df_raw, cfg)
    hol_index = generate_holidays_index(all_dates.min(), all_dates.max(), holidays)
    cal = build_calendar_features(all_dates, hol_index)
    shift_flag = not cfg.use_same_day_info
    res = aggregate_reserve(df_reserve, cfg, all_dates, shift_one_day=shift_flag)
    wea = encode_weather(df_weather, cfg, all_dates, shift_one_day=shift_flag)
    
    exog = cal.copy()
    if df_reserve is not None and cfg.use_reserve:
        exog = exog.join(res)
    if df_weather is not None and cfg.use_weather:
        exog = exog.join(wea)
    exog = exog.fillna(0.0)
    
    y_total_ts = Y_pivot.sum(axis=1).astype(float)
    total_feats = pd.DataFrame(index=y_total_ts.index)
    for L in [1, 2, 3, 7, 14, 28]: total_feats[f"合計_lag{L}"] = y_total_ts.shift(L)
    for W in [3, 7, 14, 28]: total_feats[f"合計_ma{W}"] = y_total_ts.rolling(W, min_periods=1).mean().shift(1)
    for W in [7, 14, 28]: total_feats[f"合計_std{W}"] = y_total_ts.rolling(W, min_periods=2).std().shift(1)
    if "合計_lag1" in total_feats.columns and "合計_ma7" in total_feats.columns: total_feats[f"合計_dev_ma7"] = total_feats[f"合計_lag1"] - total_feats[f"合計_ma7"]
    if "合計_ma7" in total_feats.columns and "合計_ma14" in total_feats.columns: total_feats[f"合計_ma7-ma14"] = total_feats[f"合計_ma7"] - total_feats[f"合計_ma14"]
    exog = exog.join(total_feats).fillna(0.0)

    item_feats = build_item_features(Y_pivot, exog)
    y_total = Y_pivot.sum(axis=1).astype(float)
    y_total_t = np.log1p(y_total) if cfg.target_mode == "log1p" else y_total.copy()
    return all_dates, exog, y_total, y_total_t, item_feats, Y_pivot

def _make_synthetic_data(seed: int = 42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    items = ["品目A", "品目B", "品目C"]
    rows = []
    for d in dates:
        base = 10000 + 5000 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        for i, item in enumerate(items):
            noise = rng.normal(0, 2000)
            val = max(0.0, base * (0.5 + 0.2 * i) + noise)
            rows.append({"伝票日付": d, "品名": item, "正味重量": val})
    df_raw = pd.DataFrame(rows)
    reserves = []
    for d in dates:
        cnt = max(0, int(rng.poisson(3)))
        for _ in range(cnt):
            reserves.append({"予約日": d, "予約台数": int(abs(rng.normal(3, 2)))+1, "固定客": rng.rand() < 0.4})
    df_reserve = pd.DataFrame(reserves)
    weather = []
    for d in dates:
        temp = 15 + 10 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25) + rng.normal(0, 2)
        rain = max(0.0, rng.gamma(2.0, 2.0) - 2.0)
        code = "rain" if rain > 3 else ("sunny" if temp > 18 else "cloud")
        weather.append({"日付": d, "temp": temp, "rain_mm": rain, "wcode": code})
    df_weather = pd.DataFrame(weather)
    holidays = [d for d in dates if d.weekday() >= 5]
    return df_raw, df_reserve, df_weather, holidays


def _clip_non_negative(a: np.ndarray) -> np.ndarray: return np.maximum(0.0, a)

def _predict_one_window(train_days: pd.DatetimeIndex, pred_day: pd.Timestamp, all_dates: pd.DatetimeIndex, exog: pd.DataFrame, item_feats: Dict[str, pd.DataFrame], y_total: pd.Series, cfg: WFConfig, Y_pivot: pd.DataFrame) -> Tuple[float, float, float, Dict[str, float]]:
    per_item_today_pred: Dict[str, float] = {}
    per_item_oof_meta_pred: Dict[str, pd.Series] = {}
    base_specs = _base_model_specs(cfg)

    def _to_target(y: pd.Series) -> pd.Series: return np.log1p(np.maximum(y.astype(float), 0.0)) if cfg.target_mode == "log1p" else y.astype(float)
    def _inv_target(arr: np.ndarray) -> np.ndarray: return _clip_non_negative(np.expm1(arr)) if cfg.target_mode == "log1p" else _clip_non_negative(arr)

    for item, df_item in item_feats.items():
        X_all, y_all = df_item.drop(columns=["y"]), df_item["y"].astype(float)
        X_tr, y_tr = X_all.loc[train_days], y_all.loc[train_days]
        
        y_tr_t = _to_target(y_tr)
        oof_df, fitted_full = time_series_oof_predictions(X_tr, y_tr_t, base_specs, cfg)
        
        if oof_df is None or oof_df.shape[1] == 0 or not fitted_full:
            yhat_tr = y_tr.rolling(7, min_periods=1).mean().shift(1).fillna(method="bfill")
            per_item_oof_meta_pred[item] = yhat_tr
            yhat_today = float(y_tr.tail(7).mean())
            per_item_today_pred[item] = max(0.0, yhat_today if np.isfinite(yhat_today) else 0.0)
            continue
        
        meta_model, _, _ = fit_meta_from_oof(oof_df, y_tr_t, cfg)
        yhat_tr = pd.Series(_inv_target(meta_model.predict(oof_df.fillna(0.0).values)), index=oof_df.index, dtype=float)
        per_item_oof_meta_pred[item] = yhat_tr

        hist_series = item_feats[item]["y"].loc[train_days]
        X_today = X_all.loc[[pred_day]].reindex(columns=X_tr.columns)
        yhat_today_t = meta_predict_from_full(X_today, fitted_full, meta_model, list(oof_df.columns))
        yhat_today = float(_inv_target(np.array([yhat_today_t]))[0])
        per_item_today_pred[item] = _clip_by_history(hist_series, yhat_today, win=180, ql=0.02, qh=0.98)
    
    if not per_item_today_pred:
        return 0.0, 0.0, 0.0, {}

    per_item_today_pred = _apply_ratio_guard(per_item_today_pred, Y_pivot=Y_pivot, pred_day=pred_day, share_win=56, lam=0.2)
    sum_items_pred_today = float(np.nansum(list(per_item_today_pred.values())))
    residual_pred_today = 0.0
    
    train_idx_for_res = y_total.loc[train_days].index
    if cfg.residual_model != "none" and len(train_idx_for_res) >= cfg.residual_min_days:
        sum_hat_full_oof = pd.DataFrame(per_item_oof_meta_pred).fillna(0.0).sum(axis=1)
        y_tot_full = y_total.loc[sum_hat_full_oof.index]
        residual_full = y_tot_full - sum_hat_full_oof
        exog_for_res = exog.loc[sum_hat_full_oof.index].copy()
        
        res_model = fit_residual_model(exog_for_res, residual_full, cfg)
        
        if res_model is not None and cfg.residual_cv_guard:
            folds = make_time_series_folds(len(exog_for_res), n_splits=3, gap=1)
            mae_model_cv = []; mae_zero_cv = []
            for tr, va in folds:
                X_va_fold, y_va_fold = exog_for_res.iloc[va], residual_full.iloc[va]
                res_model_fold = fit_residual_model(exog_for_res.iloc[tr], residual_full.iloc[tr], cfg)
                if res_model_fold is None: continue
                pred_va = _predict_residual(res_model_fold, X_va_fold, cfg)
                mae_model_cv.append(mae(y_va_fold.values, pred_va))
                mae_zero_cv.append(mae(y_va_fold.values, np.zeros_like(y_va_fold.values)))
            if np.nanmean(mae_model_cv) >= np.nanmean(mae_zero_cv):
                print(f"[RESIDUAL] CV guard triggered: model MAE={np.nanmean(mae_model_cv):.0f} vs zero={np.nanmean(mae_zero_cv):.0f}. Disabling residual.")
                res_model = None

        if res_model is not None:
            exog_for_today = exog.loc[[pred_day]].copy()
            r_hat = _predict_residual(res_model, exog_for_today, cfg)[0]
            clip_w = float(np.percentile(np.abs(residual_full.values), 100 * cfg.residual_clip_quantile))
            r_hat = float(np.clip(r_hat, -clip_w, clip_w))
            alpha = float(max(0.0, min(1.0, cfg.residual_alpha)))
            residual_pred_today = alpha * r_hat

    total_pred_today = max(0.0, sum_items_pred_today + residual_pred_today)
    
    return total_pred_today, sum_items_pred_today, residual_pred_today, per_item_today_pred


def walkforward_residual_stacking(df_raw: pd.DataFrame, holidays: Optional[Iterable] = None, df_reserve: Optional[pd.DataFrame] = None, df_weather: Optional[pd.DataFrame] = None, cfg: WFConfig = WFConfig()) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes the walk-forward validation and stacking pipeline."""
    set_global_seed(cfg.seed)
    
    target_items = get_target_items(df_raw, cfg)
    print(f"[INFO] Learning on {len(target_items)} items. (List: {target_items[:5]}{'...' if len(target_items)>5 else ''})")
    
    all_dates, exog, y_total, _, item_feats, Y_pivot = _prepare_exogenous(df_raw, holidays, df_reserve, df_weather, cfg)
    item_feats = {k: v for k, v in item_feats.items() if k in target_items}

    start_eval_idx = max(cfg.min_train_days, max(cfg.window_bagging_days or [0]))
    if len(all_dates) <= start_eval_idx:
        warnings.warn("Not enough data to start evaluation. Check min_train_days/max_train_window_days.")
        return pd.DataFrame(), {"message": "no predictions"}
        
    # --- Checkpoint / Resume setup ---
    ck_dir = _resolve_checkpoint_dir(cfg)
    results: List[Dict[str, Any]] = []
    if cfg.resume:
        results = _load_checkpoint(ck_dir)
    processed_dates = set()
    if results:
        try:
            processed_dates = set(pd.to_datetime([r.get("date") for r in results]).dt.normalize())
        except Exception:
            processed_dates = set()
        print(f"[RESUME] Loaded {len(results)} prior prediction rows. Will skip processed dates.")
    new_since_ck = 0

    try:
        for i in range(start_eval_idx, len(all_dates)):
            pred_day = all_dates[i]
            if cfg.resume and pd.to_datetime(pred_day).normalize() in processed_dates:
                continue

            windows = cfg.window_bagging_days
            bag_total_preds, bag_sum_preds, bag_res_preds, bag_item_preds = [], [], [], []

            for W in windows:
                train_days = all_dates[(all_dates < pred_day) & (all_dates >= pred_day - pd.Timedelta(days=W))]
                if len(train_days) < cfg.min_train_days:
                    continue

                total_pred, sum_pred, res_pred, item_preds = _predict_one_window(train_days, pred_day, all_dates, exog, item_feats, y_total, cfg, Y_pivot)
                bag_total_preds.append(total_pred)
                bag_sum_preds.append(sum_pred)
                bag_res_preds.append(res_pred)
                bag_item_preds.append(item_preds)

            if not bag_total_preds:
                continue

            total_pred_today = float(np.mean(bag_total_preds))
            sum_items_pred_today = float(np.mean(bag_sum_preds))
            residual_pred_today = float(np.mean(bag_res_preds))
            per_item_today_pred = {item: np.mean([p.get(item, 0.0) for p in bag_item_preds]) for item in item_feats.keys()}

            p = total_pred_today
            hist = y_total.loc[all_dates[(all_dates < pred_day) & (all_dates >= pred_day - pd.Timedelta(days=180))]]
            if len(hist) >= 30:
                lo, hi = np.quantile(hist.values, [0.02, 0.98])
                if (p < lo) or (p > hi) or not np.isfinite(p):
                    past = y_total.loc[all_dates[all_dates < pred_day]]
                    fallback = float(past.tail(7).mean()) if len(past) > 0 else float(p)
                    total_pred_today = max(0.0, fallback if np.isfinite(fallback) else p)

            cw = int(cfg.calibration_window_days)
            if cw > 0 and len(results) >= 10:
                hist_idx = all_dates[(all_dates < pred_day) & (all_dates >= pred_day - pd.Timedelta(days=cw))]
                if len(hist_idx) >= 10:
                    y_hist = y_total.loc[hist_idx].values
                    p_hist = pd.Series({r["date"]: r["total_pred"] for r in results}).reindex(hist_idx).values
                    mask = np.isfinite(p_hist) & np.isfinite(y_hist)
                    if mask.sum() > 5:
                        A = np.vstack([p_hist[mask], np.ones(mask.sum())]).T
                        a, b = np.linalg.lstsq(A, y_hist[mask], rcond=None)[0]
                        total_pred_today = max(0.0, a * total_pred_today + b)

            y_true_today = float(y_total.loc[pred_day])

            rec: Dict[str, Any] = {"date": pred_day, "y_true": y_true_today, "sum_items_pred": sum_items_pred_today,
                                 "residual_pred": residual_pred_today, "total_pred": total_pred_today}
            for item, val in per_item_today_pred.items(): rec[f"pred_item_{item}"] = val
            results.append(rec)
            new_since_ck += 1

            if cfg.checkpoint_every and cfg.checkpoint_every > 0 and new_since_ck >= cfg.checkpoint_every:
                _save_checkpoint(ck_dir, results, tag="progress")
                new_since_ck = 0

            if (len(results) % max(1, cfg.print_progress_every)) == 0:
                arr_true, arr_pred = np.array([r["y_true"] for r in results]), np.array([r["total_pred"] for r in results])
                print(f"[PROGRESS] {len(results)} days -> R2={r2(arr_true, arr_pred):.3f} MAE={mae(arr_true, arr_pred):,.0f}")
                if len(results) >= 60:
                    hist_true = np.array([r["y_true"] for r in results[:-28]])
                    hist_pred = np.array([r["total_pred"] for r in results[:-28]])
                    recent_true = np.array([r["y_true"] for r in results[-28:]])
                    recent_pred = np.array([r["total_pred"] for r in results[-28:]])
                    base_mae = mae(hist_true, hist_pred)
                    recent_mae = mae(recent_true, recent_pred)
                    if np.isfinite(base_mae) and np.isfinite(recent_mae) and recent_mae > 1.5 * base_mae:
                        print(f"[ALERT] Drift suspected: recent28d MAE={recent_mae:,.0f} > 1.5x hist MAE={base_mae:,.0f}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] KeyboardInterrupt received. Saving checkpoint...")
        try:
            _save_checkpoint(ck_dir, results, tag="interrupt")
        except Exception:
            pass

    if not results: return pd.DataFrame(), {"message": "no predictions"}

    # final checkpoint
    try:
        _save_checkpoint(ck_dir, results, tag="final")
    except Exception:
        pass

    if not results: return pd.DataFrame(), {"message": "no predictions"}

    df_results = pd.DataFrame(results).set_index("date").sort_index()
    y_true_arr, total_arr = df_results["y_true"].values, df_results["total_pred"].values
    sum_arr = df_results["sum_items_pred"].values
    
    items_predicted = [c for c in df_results.columns if c.startswith("pred_item_")]
    days_per_item = {c.replace("pred_item_", ""): int(df_results[c].notna().sum()) for c in items_predicted}

    scores: Dict[str, Any] = {
        "R2_total": r2(y_true_arr, total_arr), "MAE_total": mae(y_true_arr, total_arr),
        "R2_sum_only": r2(y_true_arr, sum_arr), "MAE_sum_only": mae(y_true_arr, sum_arr),
        "n_days": int(len(df_results)), 
        "config": asdict(cfg),
        "items_predicted": [c.replace("pred_item_", "") for c in items_predicted],
        "days_predicted_per_item": days_per_item,
    }

    try:
        mean_diff, ci_low, ci_high = bootstrap_mae_diff_ci(y_true_arr, sum_arr, total_arr, n_boot=1000, seed=cfg.seed)
        scores["bootstrap_mae_diff_sum_minus_total"] = {"mean": mean_diff, "ci95_low": ci_low, "ci95_high": ci_high}
        print(f"[RESIDUAL] Bootstrap MAE Diff (sum - total): mean={mean_diff:,.0f} [95%CI: {ci_low:,.0f}, {ci_high:,.0f}]")
    except Exception: pass
    
    scores["lib_versions"] = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": (lgb.__version__ if _HAS_LGBM else None),
        "catboost": (catboost.__version__ if _HAS_CAT else None),
    }

    out_dir = os.getcwd()
    df_results.to_csv(os.path.join(out_dir, "res_walkforward.csv"))
    with open(os.path.join(out_dir, "scores_walkforward.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print("[SAVED] res_walkforward.csv, scores_walkforward.json")

    if _HAS_PLT:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(14, 6))
            df_results.index = pd.to_datetime(df_results.index)
            ax.plot(df_results.index, df_results["y_true"], label="Actual", color="#1f77b4", alpha=0.8, linewidth=2)
            ax.plot(df_results.index, df_results["total_pred"], label="Predicted", color="#ff7f0e", linestyle='--', linewidth=2)
            ax.set_title("Walk-forward Prediction vs Actual", fontsize=16)
            ax.set_xlabel("Date", fontsize=12); ax.set_ylabel("Weight (kg)", fontsize=12); ax.legend()
            fig.tight_layout(); plt.savefig(os.path.join(out_dir, "pred_vs_actual.png")); plt.close()
            print("[SAVED] pred_vs_actual.png")

            tmp = df_results.copy()
            tmp["dow"] = pd.to_datetime(tmp.index).weekday
            dow_mae = tmp.groupby("dow").apply(lambda g: mae(g["y_true"].values, g["total_pred"].values))
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(range(7), [dow_mae.get(d, np.nan) for d in range(7)])
            ax.set_title("MAE by Day-of-Week"); ax.set_xlabel("DOW (Mon=0)"); ax.set_ylabel("MAE")
            fig.tight_layout(); plt.savefig(os.path.join(out_dir, "mae_by_dow.png")); plt.close()

            err = (df_results["total_pred"] - df_results["y_true"]).values
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(err, bins=40)
            ax.set_title("Error Histogram (Pred - Actual)"); ax.set_xlabel("Error"); ax.set_ylabel("Count")
            fig.tight_layout(); plt.savefig(os.path.join(out_dir, "error_hist.png")); plt.close()
            print("[SAVED] mae_by_dow.png, error_hist.png")

        except Exception as e: print(f"[WARN] Failed to draw extra charts: {e}")
    return df_results, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward stacking + residual learning for inbound volume prediction")
    parser.add_argument("--raw-csv", type=str, default=None, help="Path to raw daily detail CSV")
    parser.add_argument("--reserve-csv", type=str, default=None, help="Path to reservation CSV")
    parser.add_argument("--weather-csv", type=str, default=None, help="Path to weather CSV")
    parser.add_argument("--holidays-csv", type=str, default=None, help="Path to holidays CSV")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--raw-date-col", type=str, default="伝票日付")
    parser.add_argument("--raw-item-col", type=str, default="品名")
    parser.add_argument("--raw-weight-col", type=str, default="正味重量")
    parser.add_argument("--reserve-date-col", type=str, default="予約日")
    parser.add_argument("--reserve-count-col", type=str, default="予約台数")
    parser.add_argument("--reserve-fixed-col", type=str, default="固定客")
    parser.add_argument("--weather-date-col", type=str, default="日付")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--max-train-window-days", type=int, default=365)
    parser.add_argument("--target-mode", type=str, default="log1p", choices=["raw", "log1p"])
    parser.add_argument("--use-same-day-info", action="store_true")
    parser.add_argument("--time-decay", type=str, default="exponential", choices=["none", "linear", "exponential"], nargs='?')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--print-progress-every", type=int, default=30)
    parser.add_argument("--residual-model", type=str, default="hgbr", choices=["gbr", "hgbr", "lgbm", "catboost", "ridge", "none"])
    parser.add_argument("--residual-clip-quantile", type=float, default=0.99)
    parser.add_argument("--residual-min-days", type=int, default=60)
    parser.add_argument("--no-residual-cv-guard", action="store_true")
    parser.add_argument("--residual-alpha", type=float, default=0.55)
    parser.add_argument("--meta-model", type=str, default="ridge", choices=["ridge", "auto", "elastic"])
    parser.add_argument("--calibration-window-days", type=int, default=28)
    parser.add_argument("--max-eval-days", type=int, default=None)
    parser.add_argument("--target-items", type=str, default=None, help="Comma-separated list of items to target")
    parser.add_argument("--target-items-file", type=str, default=None, help="File with list of target items")
    parser.add_argument("--window-bagging-days", type=int, nargs='+', default=None, help="複数の学習窓（日数）を指定。例: 120 240 365")
    parser.add_argument("--top-n-items", type=int, default=None, help="学習対象を上位N品目に絞る（0/未指定で無効）")
    parser.add_argument("--no-reserve", action="store_true", help="予約データを使わない")
    parser.add_argument("--no-weather", action="store_true", help="天気データを使わない")
    # checkpoint / resume
    parser.add_argument("--resume", action="store_true", help="前回チェックポイントから再開する")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="チェックポイント保存先 (未指定なら out-dir/.checkpoints)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="予測N日ごとに保存")
    args = parser.parse_args()

    colmap = ColumnMapping(raw_date=args.raw_date_col, raw_item=args.raw_item_col, raw_weight=args.raw_weight_col,
                            reserve_date=args.reserve_date_col, reserve_count=args.reserve_count_col, reserve_fixed=args.reserve_fixed_col,
                            weather_date=args.weather_date_col)

    explicit_items = None
    if args.target_items_file:
        try:
            with open(args.target_items_file, "r", encoding="utf-8") as f:
                explicit_items = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[WARN] Failed to read target-items-file: {e}")
    if args.target_items:
        explicit_items = [v.strip() for v in args.target_items.split(",") if v.strip()]

    cfg = WFConfig(colmap=colmap, n_splits=args.n_splits, gap=args.gap, min_train_days=args.min_train_days,
                   max_train_window_days=args.max_train_window_days, target_mode=args.target_mode,
                   use_same_day_info=args.use_same_day_info, time_decay=args.time_decay, seed=args.seed,
                   n_jobs=args.n_jobs, print_progress_every=args.print_progress_every,
                   residual_model=args.residual_model, residual_clip_quantile=args.residual_clip_quantile,
                   residual_min_days=args.residual_min_days, residual_cv_guard=not args.no_residual_cv_guard,
                   residual_alpha=args.residual_alpha, meta_model=args.meta_model,
                   calibration_window_days=args.calibration_window_days,
                   explicit_target_items=explicit_items,
                   window_bagging_days=args.window_bagging_days,
                   top_n_items=args.top_n_items,
                   use_reserve=not args.no_reserve,
                   use_weather=not args.no_weather,
                   resume=args.resume,
                   checkpoint_dir=args.checkpoint_dir,
                   checkpoint_every=args.checkpoint_every)

    def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
        if path is None: return None
        try: return pd.read_csv(path)
        except Exception:
            try: return pd.read_csv(path, encoding="utf-8-sig")
            except Exception as e:
                print(f"[WARN] failed to read {path}: {e}")
                return None

    df_raw = _read_csv(args.raw_csv)
    if df_raw is not None:
        try:
            df_raw = preprocess_raw_df(df_raw, args.raw_date_col, args.raw_item_col, args.raw_weight_col)
        except ValueError as e:
            print(f"[ERROR] Raw data preprocessing failed: {e}. Falling back to synthetic demo.")
            df_raw = None
    
    if df_raw is None:
        df_raw, df_reserve, df_weather, holidays = _make_synthetic_data(seed=args.seed)
        print("[INFO] Running synthetic smoke test.")
        out_dir = args.out_dir
    else:
        df_reserve = _read_csv(args.reserve_csv)
        if df_reserve is not None:
            df_reserve = preprocess_reserve_df(df_reserve, args.reserve_date_col, args.reserve_count_col, args.reserve_fixed_col)
        df_weather = _read_csv(args.weather_csv)
        hol_df = _read_csv(args.holidays_csv)
        holidays = None
        if hol_df is not None and len(hol_df) > 0:
            col0 = hol_df.columns[0]
            try:
                holidays = pd.to_datetime(hol_df[col0]).dt.normalize().tolist()
            except Exception: holidays = None
        out_dir = args.out_dir
        if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)

    if args.max_eval_days is not None and args.max_eval_days > 0:
        date_col = args.raw_date_col
        if date_col in df_raw.columns:
            latest_date = df_raw[date_col].max()
            cutoff_date = latest_date - pd.Timedelta(days=args.max_eval_days)
            df_raw = df_raw[df_raw[date_col] >= cutoff_date].copy()
            print(f"[INFO] Restricting evaluation period to last {args.max_eval_days} days.")
    
    try:
        date_col, item_col = args.raw_date_col, args.raw_item_col
        print(f"[DEBUG] Num raw records after preprocessing: {len(df_raw)}")
        if date_col in df_raw.columns: print(f"[DEBUG] Date range: {df_raw[date_col].min()} to {df_raw[date_col].max()}")
        if item_col in df_raw.columns: print(f"[DEBUG] Num unique items: {df_raw[item_col].nunique()}")
    except Exception as e: print(f"[DEBUG] Error during data check: {e}")

    start_time = time.time()
    df_results, scores = walkforward_residual_stacking(df_raw=df_raw, holidays=holidays, df_reserve=df_reserve, df_weather=df_weather, cfg=cfg)
    elapsed = time.time() - start_time
    
    scores['runtime_seconds'] = elapsed
    scores['runtime_minutes'] = elapsed / 60.0

    print("=== Summary ===")
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    print(f"[INFO] Total runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")