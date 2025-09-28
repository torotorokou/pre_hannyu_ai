# -*- coding: utf-8 -*-
"""
predict_model_v4_2_4_best_repro.py
 - 未来リーク無しの残差バイアス補正(曜日×直近W日median)
 - 52週サイクル(woy_sin/cos)をexogへ追加
 - ★API化向け: モデル保存(Servingバンドル: Stage1パック+Stage2モデル+直近履歴)を追加

保存物(model_bundle.joblib):
{
  "version": str,
  "created_at": ISO8601,
  "cfg": dict,
  "target_items": List[str],
  "last_date": pd.Timestamp,
  "stage1_packs": Dict[item]->pack,
  "stage2_models": Dict,            # scaler/selector/p50/p90/ls/_feature_names
  "history_tail": pd.DataFrame,     # index=date, columns=["合計"]+items（直近K日）
}
"""

import os, sys, json, argparse, warnings, re, time, platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Config
# =========================
@dataclass
class Config:
    top_n: int = 6
    min_stage1_days: int = 120
    min_stage2_rows: int = 28
    use_same_day_info: bool = True
    max_history_days: int = 600
    time_decay: str = "linear"            # none | linear | exponential
    calibration_window_days: int = 28
    calibration_window_days_tuesday: int = 56
    zero_cap_quantile: float = 0.15
    share_oof_models: int = 3
    add_dow_item_interactions: bool = True
    random_state: int = 42
    # 残差バイアス補正
    resid_bias_window_days: int = 42
    resid_bias_quantile: float = 0.5
    resid_bias_cap_pct: float = 0.85
    log_resid_adjust: bool = False
    # 保存関連
    save_bundle_path: Optional[str] = None
    serving_history_days: int = 60

# =========================
# Timer
# =========================
class Timer:
    def __init__(self): self._stack = []; self.marks: Dict[str, float] = {}
    def start(self, key: str): self._stack.append((key, time.time()))
    def stop(self):
        if not self._stack: return 0.0
        key, t0 = self._stack.pop(); dt = time.time() - t0
        self.marks[key] = self.marks.get(key, 0.0) + dt
        return dt
    def get(self): return {k: float(v) for k,v in self.marks.items()}

# =========================
# Utils
# =========================
def _mae(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[m] - yhat[m]))) if m.sum() else float("nan")

def _time_decay_weights(n: int, mode: str) -> Optional[np.ndarray]:
    if n <= 0 or mode is None or mode == "none": return None
    t = np.linspace(0, 1, n)
    if mode == "linear": w = t
    elif mode == "exponential": w = np.exp(3*t)
    else: return None
    s = w.sum()
    return w/s if s>0 else None

def _blend_weight(y_true: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    n = len(y_true); k = min(28, max(8, n//4)) if n>8 else n
    yt, at, bt = y_true[-k:], a[-k:], b[-k:]
    best_w, best_mae = 0.5, 1e18
    for w in np.linspace(0,1,41):
        m = w*at + (1-w)*bt
        v = _mae(yt, m)
        if np.isfinite(v) and v < best_mae:
            best_mae, best_w = v, float(w)
    return best_w

def _norm_col(s: str) -> str:
    if s is None: return ""
    t = str(s).replace("\u3000"," ").strip()
    t = re.sub(r"[\s\-/＿－―・:：()\[\]（）［］]+","", t)
    try:
        import unicodedata
        t = unicodedata.normalize("NFKC", t)
    except Exception:
        pass
    return t.lower()

def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path: return None
    for enc in (None, "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except Exception:
            continue
    return None

def _clean_date_string(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    s = str(x)
    s = re.sub(r"[\(（][^\)）]*[\)）]", "", s)
    s = s.replace("年","/").replace("月","/").replace("日","")
    s = s.replace("-", "/")
    return s.strip()

def _parse_date_series(sr: pd.Series) -> pd.Series:
    s = sr.astype(str).map(_clean_date_string)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().any(): return dt.dt.normalize()
    s2 = s.str.replace("/", "", regex=False)
    dt2 = pd.to_datetime(s2, format="%Y%m%d", errors="coerce")
    if dt2.notna().any(): return dt2.dt.normalize()
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _auto_map_columns(df: pd.DataFrame, want: Dict[str, List[str]]) -> Dict[str, str]:
    norm_map = {c: _norm_col(c) for c in df.columns}
    inv = {}
    for k, v in norm_map.items():
        inv.setdefault(v, []).append(k)
    out = {}
    for key, aliases in want.items():
        found = None
        for a in aliases:
            na = _norm_col(a)
            if na in inv: found = inv[na][0]; break
        if found is None:
            for na, cols in inv.items():
                if any(_norm_col(a) in na for a in aliases): found = cols[0]; break
        if found is None:
            for na, cols in inv.items():
                if any(na.startswith(_norm_col(a)) for a in aliases): found = cols[0]; break
        out[key] = found
    return out

# =========================
# Preprocess
# =========================
def preprocess_raw(df: pd.DataFrame, date_col: str, item_col: str, weight_col: str,
                   out_dir: Optional[str] = None) -> pd.DataFrame:
    want = {
        "date": [date_col, "伝票日付", "日付", "受入日", "搬入日", "計上日"],
        "item": [item_col, "品名", "商品", "銘柄", "品目", "カテゴリ"],
        "weight": [weight_col, "正味重量", "重量", "数量", "重量kg", "正味量"]
    }
    cmap = _auto_map_columns(df, want)
    miss = [k for k,v in cmap.items() if v is None]
    if miss:
        msg = f"[ERROR] 必須列の自動特定に失敗: {miss}\n実列={list(df.columns)[:30]}"
        raise ValueError(msg)
    dd = df[[cmap["date"], cmap["item"], cmap["weight"]]].copy()
    dd.columns = ["__date__", "__item__", "__weight__"]
    dd["__date__"] = _parse_date_series(dd["__date__"])
    dd["__weight__"] = pd.to_numeric(dd["__weight__"].str.replace(",", "", regex=False), errors="coerce")
    dd = dd.dropna(subset=["__date__", "__weight__"])
    if len(dd) == 0:
        _emit_preprocess_diagnostics(df, date_col, item_col, weight_col, out_dir, stage="raw->clean")
        raise ValueError("preprocess後のdf_rawが空です。日付/品目/重量の列名と値の形式を確認してください。")
    dd = dd.rename(columns={"__date__": date_col, "__item__": item_col, "__weight__": weight_col})
    dd[date_col] = pd.to_datetime(dd[date_col]).dt.normalize()
    return dd

def _emit_preprocess_diagnostics(df: pd.DataFrame, date_col: str, item_col: str, weight_col: str,
                                 out_dir: Optional[str], stage: str):
    try:
        os.makedirs(out_dir or ".", exist_ok=True)
        path = os.path.join(out_dir or ".", "preprocess_diagnostics.json")
        diag = {
            "input_columns_head": list(df.columns)[:50],
            "n_rows": int(len(df)),
            "date_sample": df.get(date_col, df.iloc[:,0]).head(20).tolist(),
            "stage": stage
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
        print(f"[DIAG] 前処理診断を書き出しました: {path}")
    except Exception as e:
        print(f"[WARN] 診断書き出しに失敗: {e}")

def preprocess_reserve(df: Optional[pd.DataFrame], date_col: str, count_col: str, fixed_col: str) -> pd.DataFrame:
    if df is None or len(df) == 0: return pd.DataFrame()
    dd = df.copy()

    def _auto_map_columns(df_, want):
        norm_map = {c: _norm_col(c) for c in df_.columns}
        inv = {}
        for k, v in norm_map.items():
            inv.setdefault(v, []).append(k)
        out = {}
        for key, aliases in want.items():
            found = None
            for a in aliases:
                na = _norm_col(a)
                if na in inv: found = inv[na][0]; break
            if found is None:
                for na, cols in inv.items():
                    if any(_norm_col(a) in na for a in aliases): found = cols[0]; break
            if found is None:
                for na, cols in inv.items():
                    if any(na.startswith(_norm_col(a)) for a in aliases): found = cols[0]; break
            out[key] = found
        return out

    cmap = _auto_map_columns(dd, {
        "date":[date_col, "予約日", "日付"],
        "count":[count_col, "台数", "予約台数", "件数"],
        "fixed":[fixed_col, "固定客", "固定"]
    })
    if cmap["date"] is None:
        raise ValueError("予約データの日付列が見つかりません。")
    dd[cmap["date"]] = _parse_date_series(dd[cmap["date"]])
    if cmap["count"] in dd.columns:
        dd[cmap["count"]] = pd.to_numeric(dd[cmap["count"]].str.replace(",","",regex=False), errors="coerce")
    if cmap["fixed"] in dd.columns:
        dd[cmap["fixed"]] = dd[cmap["fixed"]].astype(str).str.lower().isin(["1","true","yes","固定","固定客"]).astype(int)
    grp = dd.groupby(cmap["date"])
    out = pd.DataFrame({
        "reserve_count": grp.size().astype(float),
        "reserve_sum": (grp[cmap["count"]].sum() if cmap["count"] in dd.columns else grp.size()).astype(float),
        "fixed_ratio": (grp[cmap["fixed"]].mean() if cmap["fixed"] in dd.columns else 0.0)
    })
    return out

# =========================
# Feature construction
# =========================
def build_pivot(df_raw: pd.DataFrame, date_col: str, item_col: str, weight_col: str):
    if len(df_raw) == 0:
        raise ValueError("preprocess後のdf_rawが空です。")
    g = df_raw.groupby([date_col, item_col])[weight_col].sum()
    if len(g) == 0:
        raise ValueError("groupby結果が空です。列名と値を確認してください。")
    pvt = g.unstack(fill_value=0.0).sort_index()
    i_min, i_max = pvt.index.min(), pvt.index.max()
    if pd.isna(i_min) or pd.isna(i_max):
        raise ValueError("日付indexにNaTが含まれています。日付の形式を確認してください。")
    full_idx = pd.date_range(i_min, i_max, freq="D")
    pvt = pvt.reindex(full_idx, fill_value=0.0)
    total = pvt.sum(axis=1).astype(float)
    return pvt, total

def get_target_items(df_raw: pd.DataFrame, date_col: str, item_col: str,
                     weight_col: str, top_n: int, lookback_days: int = 365) -> List[str]:
    dt = pd.to_datetime(df_raw[date_col], errors="coerce")
    last = dt.max()
    if pd.isna(last):
        df = df_raw.copy()
    else:
        cut = last - pd.Timedelta(days=lookback_days)
        df = df_raw[dt >= cut].copy()
    if len(df) == 0:
        df = df_raw.copy()
    s = (pd.to_numeric(df[weight_col], errors="coerce")
            .groupby(df[item_col]).sum().sort_values(ascending=False))
    return list(s.head(top_n).index)

def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    df["dow"] = idx.weekday
    df["weekofyear"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    near = set()
    for d in idx[df["is_weekend"]==1]:
        near.add(d); near.add(d + pd.Timedelta(days=1)); near.add(d - pd.Timedelta(days=1))
    df["is_holiday_nearby"] = df.index.isin(list(near)).astype(int)
    ang = 2*np.pi*df["dow"]/7.0
    df["dow_sin"] = np.sin(ang); df["dow_cos"] = np.cos(ang)
    # 52週サイクル
    woy = df["weekofyear"].astype(float)
    df["woy_sin"] = np.sin(2*np.pi*woy/52.0)
    df["woy_cos"] = np.cos(2*np.pi*woy/52.0)
    return df

def build_exog(index: pd.DatetimeIndex, total: pd.Series,
               reserve_daily: Optional[pd.DataFrame], use_same_day_info: bool) -> pd.DataFrame:
    cal = build_calendar_features(index)
    ex = cal.copy()
    if isinstance(reserve_daily, pd.DataFrame) and len(reserve_daily)>0:
        r = reserve_daily.reindex(index).fillna(0.0)
        if not use_same_day_info:
            r = r.shift(1).fillna(0.0)
        ex = ex.join(r)
    else:
        ex[["reserve_count","reserve_sum","fixed_ratio"]] = 0.0
    total = total.reindex(index).astype(float)
    ex["total_lag1"] = total.shift(1)
    ex["total_ma3"]  = total.rolling(3,  min_periods=1).mean().shift(1)
    ex["total_ma7"]  = total.rolling(7,  min_periods=1).mean().shift(1)
    ex["total_ma14"] = total.rolling(14, min_periods=1).mean().shift(1)
    return ex.fillna(0.0)

def build_item_design(item: str, pvt: pd.DataFrame, exog: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    s = pvt[item].astype(float)
    df = exog.copy()
    df[f"{item}_lag1"] = s.shift(1)
    df[f"{item}_ma7"]  = s.rolling(7, min_periods=1).mean().shift(1)
    df[f"{item}_ma28"] = s.rolling(28, min_periods=1).mean().shift(1)
    df = df.fillna(0.0)
    return df, s

# =========================
# Stage1 stacking
# =========================
def oof_stack_for_item(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Tuple[pd.Series, Dict]:
    X = X.astype(float); y = y.astype(float)
    n = len(X)
    if n < 40 or y.nunique() <= 1:
        naive = y.rolling(7, min_periods=1).mean().shift(1).bfill()
        model = Ridge(alpha=0.5).fit(naive.values.reshape(-1,1), y.values)
        meta_oof = pd.Series(model.predict(naive.values.reshape(-1,1)), index=X.index, dtype=float)
        return meta_oof, {"base": [], "meta": ("ridge", model), "cols": ["naive"], "kept_models": []}

    base_specs = [
        ("elastic", ElasticNet(alpha=0.08, l1_ratio=0.4, max_iter=20000, random_state=cfg.random_state)),
        ("rf",      RandomForestRegressor(n_estimators=240, min_samples_leaf=3, random_state=cfg.random_state)),
        ("gbr",     GradientBoostingRegressor(n_estimators=200, learning_rate=0.06, max_depth=3, subsample=0.9, random_state=cfg.random_state)),
    ]
    tscv = TimeSeriesSplit(n_splits=5)
    oof = pd.DataFrame(index=X.index, columns=[n for n,_ in base_specs], dtype=float)

    scalers, selectors, fitted = {}, {}, {}
    for name, est in base_specs:
        fold_mae = []
        for tr, va in tscv.split(np.arange(n)):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            scaler = StandardScaler()
            selector = VarianceThreshold(1e-4)
            Xt_tr = selector.fit_transform(scaler.fit_transform(Xtr.values))
            Xt_va = selector.transform(scaler.transform(Xva.values))
            sw = _time_decay_weights(len(Xt_tr), cfg.time_decay)
            m = clone(est)
            try: m.fit(Xt_tr, ytr.values, sample_weight=sw)
            except TypeError: m.fit(Xt_tr, ytr.values)
            pv = np.ravel(m.predict(Xt_va))
            oof.loc[Xva.index, name] = pv
            fold_mae.append(_mae(yva.values, pv))
        scaler = StandardScaler(); selector = VarianceThreshold(1e-4)
        Xt_full = selector.fit_transform(scaler.fit_transform(X.values))
        sw_full = _time_decay_weights(len(Xt_full), cfg.time_decay)
        m = clone(est)
        try: m.fit(Xt_full, y.values, sample_weight=sw_full)
        except TypeError: m.fit(Xt_full, y.values)
        scalers[name], selectors[name], fitted[name] = scaler, selector, m
        print(f"[OOF] {name} mean_valMAE={np.mean(fold_mae):.3f}")

    naive = y.shift(1)
    base_mae = {c: _mae(y.values, oof[c].values) for c in oof.columns}
    naive_mae = _mae(y.values, naive.values)
    keep = [k for k,v in sorted(base_mae.items(), key=lambda kv: kv[1]) if (v+1e-12) <= 0.99*naive_mae][:cfg.share_oof_models]
    if not keep: keep = [min(base_mae, key=base_mae.get)]

    oof_used = oof[keep].ffill().bfill()
    meta = Ridge(alpha=0.5); meta.fit(oof_used.values, y.values)

    cols = list(X.columns)
    kept_models = list(keep)
    pack = {
        "base":[(k, fitted[k], scalers[k], selectors[k]) for k in keep],
        "meta":("ridge", meta),
        "cols": cols,
        "kept_models": kept_models,
    }
    meta_oof = pd.Series(meta.predict(oof_used.values), index=X.index, dtype=float)
    return meta_oof, pack

def predict_from_pack(X_today: pd.DataFrame, pack: Dict) -> float:
    preds = []
    cols = pack.get("cols", list(X_today.columns))
    X_today2 = X_today.reindex(columns=cols, fill_value=0.0)
    for name, model, scaler, selector in pack["base"]:
        Xt = selector.transform(scaler.transform(X_today2.values))
        preds.append(np.ravel(model.predict(Xt))[0])
    M = np.array(preds).reshape(1,-1)
    meta = pack["meta"][1]
    return float(np.ravel(meta.predict(M))[0])

# =========================
# Stage2 matrix (interactions)
# =========================
def _add_dow_item_interactions(X: pd.DataFrame, item_pred_cols: List[str]) -> pd.DataFrame:
    if "dow" not in X.columns: return X
    X = X.copy()
    d = pd.get_dummies(X["dow"].astype(int), prefix="dow", drop_first=False)
    X = X.join(d)
    for c in item_pred_cols:
        if c not in X.columns: continue
        for k in d.columns:
            X[f"{c}__{k}"] = X[c] * d[k]
    return X

def make_stage2_matrix(df_in: pd.DataFrame, target_items: List[str], cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    cols_pred = [f"{it}_pred" for it in target_items if f"{it}_pred" in df_in.columns]
    X = df_in.drop(columns=[c for c in ["合計"] if c in df_in.columns]).copy()
    if cfg.add_dow_item_interactions:
        X = _add_dow_item_interactions(X, cols_pred)
    feature_names = list(X.columns)
    return X.astype(float), feature_names

# =========================
# Stage2 (Total) models
# =========================
def fit_total_models(df_hist: pd.DataFrame, cfg: Config, target_items: List[str]):
    X_raw, feat_names = make_stage2_matrix(df_hist, target_items, cfg)
    if "合計" not in df_hist.columns: raise ValueError("Stage2: '合計' 列が見当たりません。")
    y = df_hist["合計"].astype(float).values

    scaler = StandardScaler(); selector = VarianceThreshold(1e-4)
    Xt = selector.fit_transform(scaler.fit_transform(X_raw.values))
    sw = _time_decay_weights(len(Xt), cfg.time_decay)

    gbdt_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=cfg.random_state)
    gbdt_p90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=cfg.random_state)
    gbdt_ls  = GradientBoostingRegressor(loss="squared_error", n_estimators=220, learning_rate=0.06, max_depth=3, subsample=0.9, random_state=cfg.random_state)

    for m in (gbdt_p50, gbdt_p90, gbdt_ls):
        try: m.fit(Xt, y, sample_weight=sw)
        except TypeError: m.fit(Xt, y)

    models = {"scaler":scaler, "selector":selector, "p50":gbdt_p50, "p90":gbdt_p90, "ls":gbdt_ls}
    models["_feature_names"] = feat_names
    return models

def predict_total(models: Dict, x_today_raw: pd.DataFrame) -> Tuple[float,float,float]:
    feat = models.get("_feature_names", list(x_today_raw.columns))
    x_aligned = x_today_raw.reindex(columns=feat, fill_value=0.0)
    Xt = models["selector"].transform(models["scaler"].transform(x_aligned.values))
    p50 = float(models["p50"].predict(Xt)[0])
    p90 = float(models["p90"].predict(Xt)[0])
    mean = float(models["ls"].predict(Xt)[0])
    return p50, p90, mean

# =========================
# 残差バイアス補正 (曜日×直近W日 median)
# =========================
def _resid_bias_adjustment(pred_day: pd.Timestamp,
                           hist_df: pd.DataFrame,
                           target_items: List[str],
                           models_total: Dict,
                           w: float,
                           cfg: Config) -> float:
    if hist_df is None or len(hist_df) == 0: return 0.0
    try:
        idx_hist = pd.DatetimeIndex(hist_df.index)
        recent_mask = idx_hist >= (pd.Timestamp(pred_day) - pd.Timedelta(days=cfg.resid_bias_window_days))

        hist_df_proc, _ = make_stage2_matrix(hist_df, target_items, cfg)
        Xt_hist = models_total["selector"].transform(
            models_total["scaler"].transform(
                hist_df_proc.reindex(columns=models_total["_feature_names"], fill_value=0.0).values
            )
        )
        direct_hist = np.ravel(models_total["ls"].predict(Xt_hist))
        sum_hist = hist_df[[c for c in hist_df.columns if c.endswith("_pred")]].sum(axis=1).values.astype(float)
        base_pred_hist = w * direct_hist + (1 - w) * sum_hist

        y_hist = hist_df["合計"].values.astype(float)
        resid = y_hist - base_pred_hist

        wd = idx_hist.weekday
        dow = int(pd.Timestamp(pred_day).weekday())
        mask = (wd == dow) & recent_mask
        r = resid[mask]
        r = r[np.isfinite(r)]
        if r.size == 0: return 0.0

        q = float(cfg.resid_bias_quantile)
        adj = float(np.median(r)) if abs(q-0.5) < 1e-9 else float(np.quantile(r, q))

        r_all = resid[recent_mask]; r_all = r_all[np.isfinite(r_all)]
        if r_all.size >= 5:
            cap = float(np.quantile(np.abs(r_all), cfg.resid_bias_cap_pct))
            adj = float(np.clip(adj, -cap, cap))

        if cfg.log_resid_adjust and abs(adj) > 1e-9:
            print(f"[RESID_ADJ] {pred_day.date()} DOW={dow}  +{adj:.1f}")
        return adj
    except Exception:
        return 0.0

# =========================
# Walk-forward
# =========================
def run_walkforward(df_raw: pd.DataFrame,
                    df_reserve: Optional[pd.DataFrame],
                    date_col: str, item_col: str, weight_col: str,
                    reserve_date_col: str, reserve_count_col: str, reserve_fixed_col: str,
                    out_dir: str, cfg: Config):

    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg.random_state)

    timer = Timer(); timer.start("total_runtime")

    # ---- Preprocess ----
    timer.start("preprocess")
    df_raw = preprocess_raw(df_raw, date_col, item_col, weight_col, out_dir=out_dir)
    timer.stop()

    print(f"[DEBUG] rows_after_preprocess={len(df_raw)} "
          f"date_min={pd.to_datetime(df_raw[date_col]).min()} "
          f"date_max={pd.to_datetime(df_raw[date_col]).max()} "
          f"unique_items={df_raw[item_col].nunique()}")

    target_items = get_target_items(df_raw, date_col, item_col, weight_col, top_n=cfg.top_n)
    print(f"[INFO] target_items={target_items}")
    if not target_items:
        fallback = list(df_raw[item_col].value_counts().head(max(5, cfg.top_n)).index)
        target_items = fallback
        print(f"[WARN] fallback target_items={target_items}")

    pvt, total = build_pivot(df_raw, date_col, item_col, weight_col)
    idx = pvt.index
    reserve_daily = preprocess_reserve(df_reserve, reserve_date_col, reserve_count_col, reserve_fixed_col)
    exog = build_exog(idx, total, reserve_daily, cfg.use_same_day_info)

    results = []
    stage2_rows: List[Dict] = []
    models_total = None

    for i, pred_day in enumerate(idx):
        if i < cfg.min_stage1_days:
            if i % 20 == 0:
                print(f"[SKIP] {pred_day.date()}  i={i} < min_stage1_days={cfg.min_stage1_days}")
            continue

        train_idx = idx[(idx < pred_day)]
        train_idx = train_idx[-cfg.max_history_days:] if cfg.max_history_days else train_idx

        today_feats = exog.loc[[pred_day]].copy()
        per_item_pred = {}
        for it in target_items:
            X_all, y_all = build_item_design(it, pvt, exog)
            X_tr, y_tr = X_all.loc[train_idx], y_all.loc[train_idx]

            timer.start("stage1_per_item_fit")
            meta_oof, pack = oof_stack_for_item(X_tr, y_tr, cfg)
            timer.stop()

            timer.start("stage1_per_item_predict")
            per_item_pred[it] = predict_from_pack(today_feats, pack)
            timer.stop()

        sum_items_today = float(np.sum(list(per_item_pred.values())))

        # Stage2 当日行
        row = {f"{it}_pred": per_item_pred[it] for it in target_items}
        for c in today_feats.columns: row[c] = float(today_feats.iloc[0][c])
        row["合計"] = float(total.loc[pred_day])
        row["__date__"] = pd.Timestamp(pred_day)  # ← 保存用
        stage2_rows.append(row)

        # ウォームアップ
        if len(stage2_rows) <= cfg.min_stage2_rows:
            print(f"[DEBUG] Stage2 warmup rows={len(stage2_rows)}/{cfg.min_stage2_rows+1}")
            y_true = float(total.loc[pred_day])
            results.append({"date": pred_day, "y_true": y_true,
                            "sum_items_pred": sum_items_today, "total_pred": sum_items_today})
            continue

        # ---- Stage2 学習
        timer.start("stage2_fit")
        hist_df = pd.DataFrame(stage2_rows[:-1]).set_index(
            pd.DatetimeIndex(idx[(idx < pred_day)][-len(stage2_rows[:-1]):])
        )
        models_total = fit_total_models(hist_df, cfg, target_items)
        timer.stop()

        # ---- Stage2 予測
        timer.start("stage2_predict")
        x_total = {f"{it}_pred":[per_item_pred[it]] for it in target_items}
        for c in today_feats.columns: x_total[c] = [float(today_feats.iloc[0][c])]
        x_total = pd.DataFrame(x_total, index=[pred_day])
        x_total_proc, _ = make_stage2_matrix(x_total, target_items, cfg)
        p50, p90, mean_pred = predict_total(models_total, x_total_proc)
        timer.stop()

        # ---- ダイナミックブレンド
        hist_df_proc, _ = make_stage2_matrix(hist_df, target_items, cfg)
        y_hist = hist_df["合計"].values.astype(float)
        sum_hist = hist_df[[c for c in hist_df.columns if c.endswith("_pred")]].sum(axis=1).values.astype(float)
        direct_hist = models_total["ls"].predict(
            models_total["selector"].transform(
                models_total["scaler"].transform(
                    hist_df_proc.reindex(columns=models_total["_feature_names"], fill_value=0.0).values
                )
            )
        )
        w = _blend_weight(y_hist, np.array(direct_hist), np.array(sum_hist))
        total_pred_today = float(w * mean_pred + (1 - w) * sum_items_today)

        # ---- ガード
        is_weekend = bool(x_total.get("is_weekend", pd.Series([0], index=[pred_day])).iloc[0])
        is_hnear   = bool(x_total.get("is_holiday_nearby", pd.Series([0], index=[pred_day])).iloc[0])
        reserve_zero = (float(x_total.get("reserve_count", pd.Series([0.0], index=[pred_day])).iloc[0]) == 0.0) and \
                       (float(x_total.get("reserve_sum",   pd.Series([0.0], index=[pred_day])).iloc[0]) == 0.0)
        if reserve_zero and (is_weekend or is_hnear):
            hist = pd.Series([r["合計"] for r in stage2_rows[:-1]]).tail(180).values
            if len(hist) >= 20:
                cap = float(np.nanpercentile(hist, 100 * cfg.zero_cap_quantile))
                sum_items_today = min(sum_items_today, cap); total_pred_today = min(total_pred_today, cap)
                p50 = min(p50, cap); p90 = min(p90, cap)

        # ---- 曜日キャリブ
        dow = int(pd.to_datetime(pred_day).weekday())
        Wc = cfg.calibration_window_days_tuesday if dow == 2 else cfg.calibration_window_days
        if len(hist_df) >= Wc:
            wd = pd.DatetimeIndex(hist_df.index).weekday
            mask = (wd == dow)
            if mask.sum() >= 6:
                base_pred_hist = (w * np.array(direct_hist) + (1 - w) * sum_hist)
                y_sub = y_hist[-Wc:][mask[-Wc:]]
                p_sub = base_pred_hist[-Wc:][mask[-Wc:]]
                A = np.vstack([p_sub, np.ones_like(p_sub)]).T
                try:
                    a, b = np.linalg.lstsq(A, y_sub, rcond=None)[0]
                    total_pred_today = float(max(0.0, a * total_pred_today + b))
                except Exception:
                    pass

        # ---- 残差バイアス補正
        adj = _resid_bias_adjustment(pred_day, hist_df, target_items, models_total, w, cfg)
        total_pred_today = float(max(0.0, total_pred_today + adj))

        # ---- 保存
        y_true = float(total.loc[pred_day])
        results.append({"date": pred_day, "y_true": y_true,
                        "sum_items_pred": sum_items_today, "total_pred": total_pred_today})

        if len(results) % 10 == 0:
            ys = np.array([r["y_true"] for r in results]); ps = np.array([r["total_pred"] for r in results])
            print(f"[PROGRESS] {len(results)} days -> R²={r2_score(ys, ps):.3f}  MAE={mean_absolute_error(ys, ps):,.0f}")

    if not results:
        raise RuntimeError("有効な評価期間がありません（min_stage1_days が大きすぎる等）")

    df_res = pd.DataFrame(results).set_index("date").sort_index()
    ys = df_res["y_true"].values; ps = df_res["total_pred"].values; ss = df_res["sum_items_pred"].values

    scores = {
        "R2_total": float(r2_score(ys, ps)), "MAE_total": float(mean_absolute_error(ys, ps)),
        "R2_sum_only": float(r2_score(ys, ss)), "MAE_sum_only": float(mean_absolute_error(ys, ss)),
        "n_days": int(len(df_res)),
        "config": asdict(cfg)
    }

    # ---- Save results ----
    df_res.to_csv(os.path.join(out_dir, "res_walkforward.csv"))
    with open(os.path.join(out_dir, "scores_walkforward.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # ---- Plots ----
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_res.index, df_res["y_true"], label="Actual")
        ax.plot(df_res.index, df_res["total_pred"], "--", label="Predicted")
        ax.set_title("Walk-forward Prediction vs Actual"); ax.legend()
        fig.tight_layout(); plt.savefig(os.path.join(out_dir, "pred_vs_actual.png")); plt.close()

        err = (df_res["total_pred"] - df_res["y_true"]).values
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(err, bins=30); ax.set_title("Error Histogram (Pred - Actual)"); ax.set_xlabel("Error")
        fig.tight_layout(); plt.savefig(os.path.join(out_dir, "error_hist.png")); plt.close()

        tmp = df_res.copy(); tmp["dow"] = pd.DatetimeIndex(tmp.index).weekday
        by = tmp.groupby("dow").apply(lambda g: mean_absolute_error(g["y_true"].values, g["total_pred"].values))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(7), [by.get(d, np.nan) for d in range(7)])
        ax.set_title("MAE by Day-of-Week"); ax.set_xlabel("DOW (Mon=0)"); ax.set_ylabel("MAE")
        fig.tight_layout(); plt.savefig(os.path.join(out_dir, "mae_by_dow.png")); plt.close()
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # ---- Feature importances
    try:
        if 'models_total' in locals() and models_total is not None:
            feat_names = models_total.get("_feature_names", [])
            importances = getattr(models_total["ls"], "feature_importances_", None)
            if importances is not None and len(feat_names)==len(importances):
                pd.Series(importances, index=feat_names).sort_values(ascending=False)\
                  .to_csv(os.path.join(out_dir, "stage2_feature_importance.csv"), encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] feature importance dump failed: {e}")

    timings = timer.get(); timer.stop(); timings = timer.get()

    # ---- API用バンドル保存（任意）
    if cfg.save_bundle_path:
        try:
            print("[SERVE] building serving bundle ...")
            # 1) Stage2 を全履歴で再学習
            stage2_full = pd.DataFrame(stage2_rows)
            if "__date__" in stage2_full.columns:
                stage2_full = stage2_full.set_index("__date__")
            final_stage2_models = fit_total_models(stage2_full, cfg, target_items)

            # 2) Stage1 packs を「最後の学習窓」で再学習
            last_date = pvt.index.max()
            train_idx_all = pvt.index[-cfg.max_history_days:] if cfg.max_history_days else pvt.index
            stage1_packs = {}
            for it in target_items:
                X_all, y_all = build_item_design(it, pvt, exog)
                X_tr, y_tr = X_all.loc[train_idx_all], y_all.loc[train_idx_all]
                _, pack = oof_stack_for_item(X_tr, y_tr, cfg)
                stage1_packs[it] = pack

            # 3) 直近履歴（合計+各品目）を同梱（移動平均窓を満たす分として余裕=60日）
            K = int(max(cfg.serving_history_days, 60))
            hist_tail = pd.DataFrame({"合計": pvt.sum(axis=1).astype(float)})
            for it in target_items:
                hist_tail[it] = pvt[it].astype(float)
            hist_tail = hist_tail.tail(K)

            bundle = {
                "version": "v4_2_4_best_repro_rbias_api1",
                "created_at": pd.Timestamp.utcnow().isoformat(),
                "cfg": asdict(cfg),
                "target_items": target_items,
                "last_date": last_date,
                "stage1_packs": stage1_packs,
                "stage2_models": final_stage2_models,
                "history_tail": hist_tail
            }
            os.makedirs(os.path.dirname(cfg.save_bundle_path), exist_ok=True)
            joblib.dump(bundle, cfg.save_bundle_path, compress=3)
            print(f"[SERVE] saved model bundle -> {cfg.save_bundle_path}")
        except Exception as e:
            print(f"[WARN] serving bundle save failed: {e}")

    meta = {
        "scores": scores,
        "timings_seconds": timings,
        "n_target_items": len(target_items),
        "target_items": target_items,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "save_bundle_path": cfg.save_bundle_path
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # run_history.csv に追記
    try:
        hist_path = os.path.join(out_dir, "run_history.csv")
        row = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "R2_total": scores["R2_total"],
            "MAE_total": scores["MAE_total"],
            "n_days": scores["n_days"],
            "top_n": cfg.top_n,
            "min_stage1_days": cfg.min_stage1_days,
            "min_stage2_rows": cfg.min_stage2_rows,
            "use_same_day_info": cfg.use_same_day_info,
            "max_history_days": cfg.max_history_days,
            "time_decay": cfg.time_decay,
            "calibration_window_days": cfg.calibration_window_days,
            "calibration_window_days_tuesday": cfg.calibration_window_days_tuesday,
            "zero_cap_quantile": cfg.zero_cap_quantile,
            "share_oof_models": cfg.share_oof_models,
            "add_dow_item_interactions": cfg.add_dow_item_interactions,
            "resid_bias_window_days": cfg.resid_bias_window_days,
            "resid_bias_quantile": cfg.resid_bias_quantile,
            "resid_bias_cap_pct": cfg.resid_bias_cap_pct,
            "serving_history_days": cfg.serving_history_days,
            "save_bundle_path": cfg.save_bundle_path or "",
            "total_runtime_sec": timings.get("total_runtime", np.nan),
            "stage1_fit_sec": timings.get("stage1_per_item_fit", np.nan),
            "stage2_fit_sec": timings.get("stage2_fit", np.nan),
        }
        df_hist = pd.DataFrame([row])
        if os.path.exists(hist_path):
            df_hist_existing = pd.read_csv(hist_path)
            df_hist = pd.concat([df_hist_existing, df_hist], ignore_index=True)
        df_hist.to_csv(hist_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] history append failed: {e}")

    print("\n=== Summary ===")
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    print(f"[SAVED] {out_dir}/res_walkforward.csv, scores_walkforward.json, run_metadata.json, run_history.csv (+ plots)")
    if cfg.save_bundle_path:
        print(f"[SAVED] model bundle -> {cfg.save_bundle_path}")
    return df_res, scores

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="品目OOF→合計direct/ブレンド（best-run再現 + 残差補正 + モデル保存）")
    ap.add_argument("--raw-csv", type=str, required=True)
    ap.add_argument("--reserve-csv", type=str, default=None)
    ap.add_argument("--raw-date-col", type=str, default="伝票日付")
    ap.add_argument("--raw-item-col", type=str, default="品名")
    ap.add_argument("--raw-weight-col", type=str, default="正味重量")
    ap.add_argument("--reserve-date-col", type=str, default="予約日")
    ap.add_argument("--reserve-count-col", type=str, default="台数")
    ap.add_argument("--reserve-fixed-col", type=str, default="固定客")
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--min-stage1-days", type=int, default=120)
    ap.add_argument("--min-stage2-rows", type=int, default=28)
    ap.add_argument("--use-same-day-info", action="store_true")
    ap.add_argument("--no-same-day-info", dest="use_same_day_info", action="store_false")
    ap.set_defaults(use_same_day_info=True)
    ap.add_argument("--max-history-days", type=int, default=600)
    ap.add_argument("--time-decay", type=str, default="linear", choices=["none","linear","exponential"])
    ap.add_argument("--calibration-window-days", type=int, default=28)
    ap.add_argument("--calibration-window-days-tuesday", type=int, default=56)
    ap.add_argument("--zero-cap-quantile", type=float, default=0.15)
    ap.add_argument("--share-oof-models", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)

    # 残差補正
    ap.add_argument("--resid-bias-window-days", type=int, default=42)
    ap.add_argument("--resid-bias-quantile", type=float, default=0.5)
    ap.add_argument("--resid-bias-cap-pct", type=float, default=0.85)
    ap.add_argument("--log-resid-adjust", action="store_true")

    # 保存
    ap.add_argument("--save-bundle", type=str, default=None,
                    help="保存先 .joblib パス（指定した場合に学習済みServingバンドルを保存）")
    ap.add_argument("--serving-history-days", type=int, default=60,
                    help="バンドルに同梱する直近履歴日数（既定60）")

    args = ap.parse_args()

    df_raw = _read_csv(args.raw_csv)
    if df_raw is None or len(df_raw) == 0:
        raise FileNotFoundError(f"raw-csv 読み込み失敗: {args.raw_csv}")
    df_res = _read_csv(args.reserve_csv) if args.reserve_csv else None

    cfg = Config(
        top_n=args.top_n,
        min_stage1_days=args.min_stage1_days,
        min_stage2_rows=args.min_stage2_rows,
        use_same_day_info=args.use_same_day_info,
        max_history_days=args.max_history_days,
        time_decay=args.time_decay,
        calibration_window_days=args.calibration_window_days,
        calibration_window_days_tuesday=args.calibration_window_days_tuesday,
        zero_cap_quantile=args.zero_cap_quantile,
        share_oof_models=args.share_oof_models,
        add_dow_item_interactions=True,
        random_state=args.random_state,
        resid_bias_window_days=args.resid_bias_window_days,
        resid_bias_quantile=args.resid_bias_quantile,
        resid_bias_cap_pct=args.resid_bias_cap_pct,
        log_resid_adjust=args.log_resid_adjust,
        save_bundle_path=args.save_bundle,
        serving_history_days=args.serving_history_days,
    )

    run_walkforward(df_raw=df_raw, df_reserve=df_res,
                    date_col=args.raw_date_col, item_col=args.raw_item_col, weight_col=args.raw_weight_col,
                    reserve_date_col=args.reserve_date_col, reserve_count_col=args.reserve_count_col, reserve_fixed_col=args.reserve_fixed_col,
                    out_dir=args.out_dir, cfg=cfg)

if __name__ == "__main__":
    main()
