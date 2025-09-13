import os
import json
import time
import pickle
import hashlib
from typing import Any, Optional, Tuple, Dict
import pandas as pd

from .predict_model_v4_2_4 import full_walkforward, get_feature_list, get_target_items

# --------------------------------------------------
# Data signature utilities
# --------------------------------------------------

def _datetime_range(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        return None, None
    col = dt_cols[0]
    return (df[col].min().isoformat() if len(df) else None, df[col].max().isoformat() if len(df) else None)

def _quick_numeric_checksum(df: pd.DataFrame, limit_cols: int = 20) -> str:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return "0"
    num_cols = num_cols[:limit_cols]
    stats = {}
    for c in num_cols:
        s = df[c]
        stats[c] = [float(s.sum(skipna=True)), float(s.mean(skipna=True)), float(s.std(skipna=True) or 0.0)]
    raw = json.dumps(stats, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def dataframe_signature(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    start, end = _datetime_range(df)
    checksum = _quick_numeric_checksum(df)
    return {
        "name": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "date_start": start,
        "date_end": end,
        "num_checksum": checksum,
    }

# --------------------------------------------------
# Cache key / path
# --------------------------------------------------

def build_cache_key(params: Dict[str, Any], sigs: Dict[str, Dict[str, Any]]) -> str:
    base = {
        "params": {k: v for k, v in params.items() if k not in {"df_raw", "df_reserve", "df_weather", "holidays"}},
        "sigs": sigs,
    }
    raw = json.dumps(base, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"fw_{key}.pkl")

# --------------------------------------------------
# Public API
# --------------------------------------------------

def full_walkforward_cached(
    *,
    df_raw: pd.DataFrame,
    holidays,
    df_reserve: pd.DataFrame,
    df_weather: Optional[pd.DataFrame] = None,
    min_stage1_days: int,
    min_stage2_days: int,
    top_n: int = 5,
    allowed_features: Optional[list] = None,
    cache_dir: str = "/works/data/cache",
    force_recompute: bool = False,
    describe: bool = True,
) -> Tuple[list, list, Any, list, Dict[str, Any]]:
    """Wrapper that caches full_walkforward outputs.

    Returns
    -------
    (actual, pred, model, dates, meta)
    meta['_cache'] contains {'hit': bool, 'path': str, 'duration_sec': float, 'key': str}
    """
    os.makedirs(cache_dir, exist_ok=True)

    sig_raw = dataframe_signature(df_raw, "raw")
    sig_res = dataframe_signature(df_reserve, "reserve")
    sig_weather = dataframe_signature(df_weather, "weather") if isinstance(df_weather, pd.DataFrame) else {}

    params = dict(
        min_stage1_days=min_stage1_days,
        min_stage2_days=min_stage2_days,
        top_n=top_n,
        allowed_features=sorted(allowed_features) if allowed_features else None,
    )
    key = build_cache_key(params, {"raw": sig_raw, "reserve": sig_res, "weather": sig_weather})
    path = cache_path(cache_dir, key)

    t0 = time.time()
    if (not force_recompute) and os.path.exists(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["meta"]["_cache"] = {
            "hit": True,
            "path": path,
            "duration_sec": time.time() - t0,
            "key": key,
        }
        if describe:
            print(f"[CACHE HIT] {path} ({payload['meta'].get('feature_count')} features) load_time={payload['meta']['_cache']['duration_sec']:.2f}s")
        return payload["actual"], payload["pred"], payload["model"], payload["dates"], payload["meta"]

    # Cache miss -> compute
    if describe:
        print(f"[CACHE MISS] computing full_walkforward key={key} ...")
    actual, pred, model, dates = full_walkforward(
        df_raw=df_raw,
        holidays=holidays,
        df_reserve=df_reserve,
        df_weather=df_weather,
        min_stage1_days=min_stage1_days,
        min_stage2_days=min_stage2_days,
        top_n=top_n,
        allowed_features=allowed_features,
    )
    meta = {
        "sig_raw": sig_raw,
        "sig_reserve": sig_res,
        "sig_weather": sig_weather,
        "params": params,
        "feature_count": len(model.get('_models', {}).get('raw_feature_names', [])) if isinstance(model, dict) else None,
    }
    meta["_cache"] = {"hit": False, "path": path, "key": key, "duration_sec": time.time() - t0}
    with open(path, "wb") as f:
        pickle.dump({"actual": actual, "pred": pred, "model": model, "dates": dates, "meta": meta}, f)
    if describe:
        print(f"[CACHE SAVED] {path} time={meta['_cache']['duration_sec']:.2f}s")
    return actual, pred, model, dates, meta

__all__ = ["full_walkforward_cached", "dataframe_signature"]
