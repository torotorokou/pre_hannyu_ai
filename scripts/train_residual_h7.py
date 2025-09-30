# -*- coding: utf-8 -*-
"""
train_residual_h7.py — 7日間 逐次予測（残差ブースト）完全版

目的:
- 既存ベスト日次モデルの履歴出力（res_walkforward.csv）の「残差 r = y_true - base_pred」を学習
- 直近ウィンドウのカレンダー/予約特徴から残差を推定し、将来7日のベース予測を補正して精度向上
- API化を想定し、モデル保存（joblib）にも対応

入力:
- --res-walk-csv : 既存ベストモデルの res_walkforward.csv（少なくとも date, y_true, total_pred を含む）
- （任意）--reserve-csv : 予約CSV（過去〜将来分が混在可）。日付は自動パース、当日/前日使用選択可
- （任意）--base-future-csv : 将来7日分のベース予測（date, base_pred 列）。未指定なら自動生成
- その他オプションはCLI参照

出力:
- out_dir/h7_forecast.csv           : 将来7日分の {date, base_pred, resid_pred, yhat} ほか
- out_dir/scores_residual.json      : 学習ウィンドウ内の改善度（MAE等）
- out_dir/resid_model.joblib        : 残差モデルと前処理バンドル（APIで再利用）
- 図 out_dir/backtest_resid_effect.png : 直近ウィンドウでの補正効果（任意）

要件:
- pandas, numpy, scikit-learn, joblib, matplotlib（可視化は任意）
"""

import os, re, json, argparse, warnings, io, csv, time, platform
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.copy_on_write = True


# =========================
# Utils
# =========================
def _norm_col(s: str) -> str:
    if s is None: return ""
    t = str(s).replace("\u3000"," ").strip()
    t = re.sub(r"[\s\-/＿－―・:：()\[\]（）［］]+","", t)
    try:
        import unicodedata; t = unicodedata.normalize("NFKC", t)
    except Exception: pass
    return t.lower()

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

def _read_csv_any(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"ファイルが存在しません: {fp}")
    encs = [None, "utf-8-sig", "utf-8", "cp932", "shift_jis"]
    for enc in encs:
        try:
            return pd.read_csv(fp, encoding=enc, dtype=str, low_memory=False)
        except Exception:
            continue
    # fallback: permissive reading
    with open(fp, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    return pd.read_csv(io.StringIO(text), dtype=str, low_memory=False, engine="python", on_bad_lines="skip")

def _time_decay_weights(n: int, mode: str) -> Optional[np.ndarray]:
    if n <= 0 or mode in (None, "none"): return None
    t = np.linspace(0, 1, n)
    if mode == "linear": w = t
    elif mode == "exponential": w = np.exp(3*t)
    else: return None
    s = w.sum()
    return w/s if s>0 else None


# =========================
# Reserve loader
# =========================
def load_reserve_daily(reserve_csv: Optional[str]) -> pd.DataFrame:
    if not reserve_csv: return pd.DataFrame()
    df = _read_csv_any(reserve_csv)
    cmap = {
        "date": None, "count": None, "fixed": None
    }
    norm_map = {c: _norm_col(c) for c in df.columns}
    inv = {}
    for k,v in norm_map.items():
        inv.setdefault(v, []).append(k)
    def pick(aliases):
        for a in aliases:
            na = _norm_col(a)
            if na in inv: return inv[na][0]
        for na, cols in inv.items():
            if any(_norm_col(a) in na for a in aliases): return cols[0]
        for na, cols in inv.items():
            if any(na.startswith(_norm_col(a)) for a in aliases): return cols[0]
        return None

    cmap["date"]  = pick(["予約日","日付","伝票日付","date"])
    cmap["count"] = pick(["台数","予約台数","件数","count"])
    cmap["fixed"] = pick(["固定客","固定","fixed"])

    if cmap["date"] is None:
        raise ValueError("予約データの日付列が見つかりません。")

    d = pd.DataFrame()
    d["date"] = _parse_date_series(df[cmap["date"]])

    if cmap["count"] in df.columns:
        cnt = pd.to_numeric(df[cmap["count"]].astype(str).str.replace(",","",regex=False), errors="coerce").fillna(1.0)
    else:
        cnt = pd.Series(1.0, index=df.index)
    if cmap["fixed"] in df.columns:
        fx = df[cmap["fixed"]].astype(str).str.lower().isin(["1","true","yes","固定","固定客"]).astype(int)
    else:
        fx = pd.Series(0, index=df.index)

    d["cnt"] = cnt.values
    d["fx"]  = fx.values
    grp = d.groupby("date").agg(
        reserve_count=("cnt","count"),
        reserve_sum=("cnt","sum"),
        fixed_ratio=("fx","mean")
    ).astype(float)
    return grp


# =========================
# Feature builder
# =========================
def build_feats(idx: pd.DatetimeIndex,
                hist_total: pd.Series,
                reserve_daily: Optional[pd.DataFrame],
                use_same_day_reserve: bool,
                feature_set: str = "full") -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["dow"] = df.index.weekday
    ang = 2*np.pi*df["dow"]/7.0
    df["dow_sin"] = np.sin(ang); df["dow_cos"] = np.cos(ang)
    df["is_weekend"] = (df["dow"]>=5).astype(int)
    # ラグ/移動平均（ベース: 実績の合計 or 直近値）
    s = hist_total.reindex(idx).astype(float)
    df["total_lag1"] = s.shift(1)
    df["total_ma3"]  = s.rolling(3,  min_periods=1).mean().shift(1)
    df["total_ma7"]  = s.rolling(7,  min_periods=1).mean().shift(1)
    df["total_ma14"] = s.rolling(14, min_periods=1).mean().shift(1)

    # 予約
    if reserve_daily is not None and len(reserve_daily)>0:
        r = reserve_daily.reindex(idx).fillna(0.0)
        if not use_same_day_reserve:
            r = r.shift(1).fillna(0.0)
        df = df.join(r)
    else:
        df[["reserve_count","reserve_sum","fixed_ratio"]] = 0.0
    df = df.fillna(0.0).astype(float)

    # Feature set control (proposal B)
    if feature_set == "no_reserve":
        drop_cols = [c for c in df.columns if c.startswith("reserve_") or c == "fixed_ratio"]
        df = df.drop(columns=drop_cols, errors="ignore")
    elif feature_set == "light":
        keep = [
            "dow_sin", "dow_cos", "is_weekend",
            "total_lag1", "total_ma7"
        ]
        existing = [c for c in keep if c in df.columns]
        df = df[existing]
    # else: full = use all

    return df


# =========================
# Core logic
# =========================
def run(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 履歴（ベストモデル出力）読み込み
    dfw = _read_csv_any(args.res_walk_csv)
    # date列 or インデックス対応
    date_col = None
    for cand in ["date","日付","伝票日付"]:
        if cand in dfw.columns: date_col = cand; break
    if date_col is not None:
        dfw[date_col] = _parse_date_series(dfw[date_col])
        dfw = dfw.set_index(date_col)
    else:
        # 既にindexが日付文字の可能性
        try:
            dfw.index = _parse_date_series(pd.Series(dfw.index))
        except Exception:
            raise ValueError("日付列/インデックスの特定に失敗（res_walkforward.csv）")

    # y_true / base_pred（列名ゆらぎに対応）
    def pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns: return c
        # ゆる探索
        low = {c: _norm_col(c) for c in df.columns}
        for c, n in low.items():
            if any(k in n for k in cands): return c
        return None

    y_col = pick_col(dfw, ["y_true","actual","y","truth"])
    p_col = pick_col(dfw, ["total_pred","pred","yhat","forecast"])
    if y_col is None or p_col is None:
        raise ValueError(f"必要列が見つかりません: y_col={y_col}, p_col={p_col}; cols={list(dfw.columns)[:20]}")

    dfw = dfw.sort_index()
    dfw = dfw[[y_col, p_col]].rename(columns={y_col:"y_true", p_col:"base_pred"})
    # 数値列へ強制変換（カンマ除去 / 非数はNaN）
    for c in ["y_true","base_pred"]:
        dfw[c] = pd.to_numeric(dfw[c].astype(str).str.replace(",","", regex=False), errors="coerce")
    # 欠損を除外（学習に不要な行）
    before = len(dfw)
    dfw = dfw.dropna(subset=["y_true","base_pred"])  # 数値化失敗行を排除
    if len(dfw) < before:
        print(f"[INFO] Dropped {before-len(dfw)} rows with non-numeric y_true/base_pred")
    # 残差
    dfw["resid"] = dfw["y_true"] - dfw["base_pred"]

    # 2) 予約（過去〜将来）
    reserve_daily = load_reserve_daily(args.reserve_csv)

    # 3) 残差モデル学習ウィンドウ
    #    直近 residual_window_days のデータで学習
    df_hist = dfw.dropna().copy()
    if len(df_hist)==0:
        raise RuntimeError("学習データが空です。")
    end_date = df_hist.index.max()
    start_date = end_date - pd.Timedelta(days=args.residual_window_days-1)
    df_win = df_hist.loc[(df_hist.index>=start_date)&(df_hist.index<=end_date)].copy()
    # 特徴量（実績の y_true を使ってラグ/MAを構成）
    X_win = build_feats(df_win.index, hist_total=df_hist["y_true"], reserve_daily=reserve_daily,
                        use_same_day_reserve=args.use_same_day_reserve, feature_set=args.feature_set)
    y_win_series = df_win["resid"].astype(float)

    # ホールドアウト設定（1週間など）
    holdout_days = getattr(args, "holdout_days", 0) or 0
    if holdout_days > 0 and holdout_days < len(df_win):
        holdout_start = end_date - pd.Timedelta(days=holdout_days-1)
        val_idx = df_win.index[df_win.index >= holdout_start]
        train_idx = df_win.index[df_win.index < holdout_start]
    else:
        val_idx = pd.DatetimeIndex([])
        train_idx = df_win.index

    X_train = X_win.loc[train_idx]
    y_train = y_win_series.loc[train_idx].values

    # スケーラ・モデル
    scaler = StandardScaler()
    selector = VarianceThreshold(1e-5)
    Xt = selector.fit_transform(scaler.fit_transform(X_train.values))
    sw = _time_decay_weights(len(Xt), args.time_decay)

    # 損失: huber / absolute_error / squared_error（既定: huber）
    loss = args.stage2_loss
    gbr = GradientBoostingRegressor(
        loss=loss,
        alpha=0.9 if loss=="huber" else 0.5,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.random_state
    )
    try:
        gbr.fit(Xt, y_train, sample_weight=sw)
    except TypeError:
        gbr.fit(Xt, y_train)

    # 4) バックテスト（学習窓内での改善度）
    # 学習窓内推定（訓練 + 任意で検証）
    # 訓練データ予測
    resid_pred_train = gbr.predict(selector.transform(scaler.transform(X_train.values)))
    base_pred_train = df_win.loc[train_idx, "base_pred"].astype(float).values
    yhat_train = base_pred_train + resid_pred_train

    # 全窓（訓練+検証）での推定（可視化用）
    resid_pred_full = gbr.predict(selector.transform(scaler.transform(X_win.loc[train_idx].values)))
    yhat_full = pd.Series(index=train_idx, data=base_pred_train + resid_pred_full)

    # 検証データ（ホールドアウト）
    val_metrics = {}
    if len(val_idx) > 0:
        X_val = X_win.loc[val_idx]
        resid_pred_val = gbr.predict(selector.transform(scaler.transform(X_val.values)))
        base_pred_val = df_win.loc[val_idx, "base_pred"].astype(float).values
        yhat_val = base_pred_val + resid_pred_val
        y_true_val = df_win.loc[val_idx, "y_true"].values
        mae_base_val = float(mean_absolute_error(y_true_val, base_pred_val))
        mae_boost_val = float(mean_absolute_error(y_true_val, yhat_val))
        r2_val = float(r2_score(y_true_val, yhat_val)) if len(val_idx) > 1 else None
        improve_val = float((mae_base_val - mae_boost_val) / mae_base_val) if mae_base_val > 0 else None
        val_metrics = {
            "val_days": int(len(val_idx)),
            "val_start": str(val_idx.min().date()),
            "val_end": str(val_idx.max().date()),
            "MAE_base_val": mae_base_val,
            "MAE_boosted_val": mae_boost_val,
            "R2_boosted_val": r2_val,
            "improve_ratio_val": improve_val,
        }
        # 併せて full へ拡張
        yhat_full = pd.concat([yhat_full, pd.Series(index=val_idx, data=yhat_val)])

    y_true_full = df_win["y_true"].values
    base_full = df_win["base_pred"].astype(float).values
    mae_base_full = float(mean_absolute_error(y_true_full, base_full))
    mae_boost_full = float(mean_absolute_error(y_true_full, yhat_full.loc[df_win.index].values))
    improve_full = float((mae_base_full - mae_boost_full)/mae_base_full) if mae_base_full>0 else None

    scores = {
        "train_window_days": int(len(df_win)),
        "train_start": str(df_win.index.min().date()),
        "train_end": str(df_win.index.max().date()),
        "holdout_days": int(holdout_days),
        "MAE_base_train": float(mean_absolute_error(df_win.loc[train_idx, "y_true"].values, base_pred_train)),
        "MAE_boosted_train": float(mean_absolute_error(df_win.loc[train_idx, "y_true"].values, yhat_train)),
        "improve_ratio_train": float((mean_absolute_error(df_win.loc[train_idx, "y_true"].values, base_pred_train) - mean_absolute_error(df_win.loc[train_idx, "y_true"].values, yhat_train)) / mean_absolute_error(df_win.loc[train_idx, "y_true"].values, base_pred_train)) if len(train_idx)>0 else None,
        "MAE_base_full": mae_base_full,
        "MAE_boosted_full": mae_boost_full,
        "improve_ratio_full": improve_full,
        **val_metrics
    }

    # 5) 将来7日ぶんのベース予測を用意
    future_days = args.future_days
    d1 = end_date + pd.Timedelta(days=1)
    idx_future = pd.date_range(d1, periods=future_days, freq="D")

    base_future = None
    if args.base_future_csv:
        bf = _read_csv_any(args.base_future_csv)
        c_date = None
        for cand in ["date","日付","伝票日付"]:
            if cand in bf.columns: c_date = cand; break
        if c_date is None:
            raise ValueError("base-future-csv に日付列が見つかりません。")
        bf[c_date] = _parse_date_series(bf[c_date])
        bf = bf.set_index(c_date).sort_index()
        c_base = None
        for cand in ["base_pred","total_pred","pred","yhat","forecast"]:
            if cand in bf.columns: c_base = cand; break
        if c_base is None:
            raise ValueError("base-future-csv に base_pred 相当の列が見つかりません。")
        base_future = bf.reindex(idx_future)[c_base].astype(float)
    else:
        # 自動生成: 曜日ごとの直近平均 × 移動平均のハイブリッド
        hist = df_hist.copy()
        # 直近N週間の曜日平均（実績ベース）
        weeks = max(8, args.residual_window_days//7)
        cut = end_date - pd.Timedelta(days=7*weeks-1)
        sub = hist.loc[hist.index>=cut]
        dow_mean = sub.groupby(sub.index.weekday)["y_true"].mean()
        ma7 = hist["y_true"].rolling(7, min_periods=1).mean().iloc[-1]
        preds = []
        for d in idx_future:
            m1 = dow_mean.get(d.weekday(), ma7)
            preds.append(0.5*float(m1) + 0.5*float(ma7))
        base_future = pd.Series(preds, index=idx_future, dtype=float)

    # 6) 将来7日分の特徴量（履歴は y_true、将来は直近の実績を引き伸ばしつつ生成）
    # total_lag/MA は学習時同様「実績の合計」をベースに作っていたため、
    # 将来は最新実績のラグ系列を固定（保守的）とする。
    hist_total_for_feats = df_hist["y_true"].copy()
    # 将来分は直近値を延長（MAは固定的に機能）
    ext = pd.Series([hist_total_for_feats.iloc[-1]]*future_days, index=idx_future)
    hist_ext = pd.concat([hist_total_for_feats, ext])
    X_fut_all = build_feats(hist_ext.index, hist_total=hist_ext,
                            reserve_daily=reserve_daily, use_same_day_reserve=args.use_same_day_reserve,
                            feature_set=args.feature_set)
    X_fut = X_fut_all.loc[idx_future]

    resid_pred_future = gbr.predict(selector.transform(scaler.transform(X_fut.values)))
    yhat_future = (base_future.values + resid_pred_future).clip(min=0.0)

    # ガード（極端な外れ値抑制: 直近P分位）
    if args.cap_quantile is not None:
        hist_vals = df_hist["y_true"].tail(max(90, args.residual_window_days)).values
        cap = float(np.nanpercentile(hist_vals, 100*args.cap_quantile))
        yhat_future = np.minimum(yhat_future, cap)

    # 7) 保存
    out = pd.DataFrame({
        "date": idx_future,
        "base_pred": base_future.values.astype(float),
        "resid_pred": resid_pred_future.astype(float),
        "yhat": yhat_future.astype(float)
    }).set_index("date")
    out.to_csv(os.path.join(args.out_dir, "h7_forecast.csv"), encoding="utf-8-sig")

    with open(os.path.join(args.out_dir, "scores_residual.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # 図（任意）
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_win.index, df_win["y_true"].values, label="Actual (window)")
        ax.plot(df_win.index, df_win["base_pred"].values, "--", label="Base (window)")
        ax.plot(df_win.index, yhat_full.loc[df_win.index].values, "-.", label="Boosted (window)")
        if len(val_idx)>0:
            ax.axvspan(val_idx.min(), val_idx.max(), color="#ffcc88", alpha=0.3, label="Holdout")
        ax.set_title("Residual Boost — Backtest on Training Window")
        ax.legend()
        fig.tight_layout(); plt.savefig(os.path.join(args.out_dir, "backtest_resid_effect.png")); plt.close()
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # バンドル（API用）
    if args.save_bundle:
        bundle = {
            "scaler": scaler,
            "selector": selector,
            "model": gbr,
            "cols": list(X_win.columns),
            "config": {
                "residual_window_days": args.residual_window_days,
                "time_decay": args.time_decay,
                "stage2_loss": args.stage2_loss,
                "use_same_day_reserve": args.use_same_day_reserve,
                "cap_quantile": args.cap_quantile,
                "feature_set": args.feature_set,
                "model_params": {
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "max_depth": args.max_depth,
                    "subsample": args.subsample,
                    "random_state": args.random_state,
                }
            },
            "train_scores": scores,
            "env": {
                "python": platform.python_version(),
                "numpy": __import__("numpy").__version__,
                "pandas": __import__("pandas").__version__,
                "sklearn": __import__("sklearn").__version__,
                "platform": platform.platform(),
            }
        }
        dump(bundle, args.save_bundle)
        print(f"[SAVED] bundle -> {args.save_bundle}")

    print("\n=== Residual H+7 Summary ===")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"[DONE] out_dir={args.out_dir}  elapsed={time.time()-t0:.1f}s")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="7日間残差ブースト予測（既存ベース予測の補正）")
    ap.add_argument("--res-walk-csv", type=str, required=True,
                    help="既存ベストモデルの res_walkforward.csv（少なくとも date,y_true,total_pred）")
    ap.add_argument("--reserve-csv", type=str, default=None,
                    help="予約CSV（過去〜将来混在可：予約日/台数/固定客）")
    ap.add_argument("--out-dir", type=str, required=True)

    # 学習設定
    ap.add_argument("--residual-window-days", type=int, default=84,
                    help="残差学習に使う直近日数")
    ap.add_argument("--holdout-days", type=int, default=7,
                    help="学習窓末尾からのホールドアウト日数 (0で無効)")
    ap.add_argument("--time-decay", type=str, default="exponential",
                    choices=["none","linear","exponential"])
    ap.add_argument("--stage2-loss", type=str, default="huber",
                    choices=["huber","absolute_error","squared_error"],
                    help="残差モデルの損失関数")
    ap.add_argument("--use-same-day-reserve", action="store_true",
                    help="予約を当日情報として使う（デフォルトFalseで前日情報扱い）")
    ap.add_argument("--no-same-day-reserve", dest="use_same_day_reserve", action="store_false")
    ap.set_defaults(use_same_day_reserve=False)

    # Feature set selection (proposal B)
    ap.add_argument("--feature-set", type=str, default="full",
                    choices=["full","no_reserve","light"],
                    help="特徴量セット: full=全特徴, no_reserve=予約系除外, light=最小限")

    # Model capacity hyperparameters (proposal A)
    ap.add_argument("--n-estimators", type=int, default=300, help="GBR 木数")
    ap.add_argument("--learning-rate", type=float, default=0.05, help="GBR 学習率")
    ap.add_argument("--max-depth", type=int, default=3, help="GBR 木の最大深さ")
    ap.add_argument("--subsample", type=float, default=0.9, help="GBR サブサンプル率")

    # 将来ベース予測
    ap.add_argument("--base-future-csv", type=str, default=None,
                    help="将来7日分のベース予測CSV（date, base_pred）。未指定時は自動生成")
    ap.add_argument("--future-days", type=int, default=7, help="予測日数（既定7）")

    # 外れ値抑制
    ap.add_argument("--cap-quantile", type=float, default=0.995,
                    help="最終yhatの上側分位キャップ（Noneで無効）")

    # 保存
    ap.add_argument("--save-bundle", type=str, default=None,
                    help="残差モデルの保存先（joblib）")
    ap.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
