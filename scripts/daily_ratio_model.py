# -*- coding: utf-8 -*-
"""
daily_ratio_model.py — 日次「比率」モデル（前処理強化＋学習/推論一体のWF評価＋保存対応）

目的
- 月合計(真値) × 日次比率(学習推定) で日次実数を再構成するための「比率」モデルを学習・評価。
- 月次実数が既に高精度にある前提で、日内配分のみをモデル化する。

主な仕様
- 入力CSVは1つでも複数でもOK（列名は自動推定：日付/重量/予約関連は任意）。
- 月内比率 target_ratio = day_total / month_total を教師にする（0〜1、非稼働日は任意で除外）。
- walk-forward（時系列前進）で学習期間を「直近Nヶ月」に制限（--max-train-months）。
- 時間減衰サンプル重み（none/linear/exponential）。
- モデルは GradientBoostingRegressor（損失=huber）を既定。
- 予測後に [0,1] へクリップ。さらに --hard-month-normalize で「その月の比率和=1」へ再正規化。
- 予約があれば日次外生に加える（--use-same-day-reserve / --no-same-day-reserve）。
- 特徴量：曜日/週番号/サインコサイン、月内位置（1〜末日/稼働日順位）、予約要約、
          （任意）曜日×月内位置の交互作用（--add-dow-monthpos）。

出力
- res_ratio_walkforward.csv : date, kg_true, kg_pred (= month_true * ratio_pred), ratio_true, ratio_pred 等
- scores_ratio.json         : MAEやR2（kgベース）など
- 図: pred_vs_actual_ratio.png / ratio_error_hist.png
- （任意）--save-bundle で joblib に学習済み器と前処理メタを保存

使い方（例）
python scripts/daily_ratio_model.py \
  --raw-csvs /works/data/input/2020顧客.csv /works/data/input/2022顧客.csv \
             /works/data/input/2023_all.csv /works/data/input/20240501-20250422.csv \
  --reserve-csv /works/data/input/yoyaku_data.csv \
  --out-dir /works/data/output/ratio_24m_strong \
  --max-train-months 24 \
  --time-decay exponential \
  --active-day-mask \
  --add-dow-monthpos \
  --min-month-total 1000 \
  --max-daily-kg 150000 \
  --hard-month-normalize \
  --save-bundle /works/data/output/ratio_24m_strong/ratio_model.joblib
"""

import os, json, argparse, re, time, platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# Config / helpers
# =========================

@dataclass
class Config:
    max_train_months: int = 36
    min_month_total: float = 0.0
    min_active_days: int = 8
    max_daily_kg: Optional[float] = None
    time_decay: str = "linear"  # none | linear | exponential
    use_same_day_reserve: bool = True
    add_reserve_dow_interactions: bool = False
    active_day_mask: bool = False
    hard_month_normalize: bool = False
    add_dow_monthpos: bool = False
    random_state: int = 42


def _time_decay_weights(n: int, mode: str) -> Optional[np.ndarray]:
    if n <= 0 or mode is None or mode == "none": return None
    t = np.linspace(0, 1, n)
    if mode == "linear": w = t
    elif mode == "exponential": w = np.exp(3*t)
    else: return None
    s = w.sum()
    return (w/s) if s > 0 else None


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


def _auto_map_columns(df: pd.DataFrame, want: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
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


def _clean_date_string(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    s = str(x)
    s = re.sub(r"[\(（][^\)）]*[\)）]", "", s)  # 曜日等を除去
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


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in (None, "utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except Exception:
            continue
    raise FileNotFoundError(f"CSV読み込み失敗: {path}")


# =========================
# Load & aggregate daily totals
# =========================

def load_daily_total(
    raw_csvs: List[str],
    date_alias=("伝票日付","日付","受入日","搬入日","計上日"),
    item_alias=("品名","商品","銘柄","品目","カテゴリ"),
    weight_alias=("正味重量","重量","数量","重量kg","正味量"),
    max_daily_kg: Optional[float] = None
) -> pd.DataFrame:
    """
    すべての行を読み込み -> 必須3列を自動特定 -> 日毎に合計kgで集約。
    - 「曜日等の括弧」は落としてから日付パース
    - 重量は数値化・NaN drop
    - 日次の不自然な外れ値は max_daily_kg で上限カット（任意）
    return: DataFrame(index=DatetimeIndex, columns=["kg"])
    """
    frames = []
    for fp in raw_csvs:
        df = _read_csv_any(fp)
        cmap = _auto_map_columns(df, {
            "date": list(date_alias),
            "item": list(item_alias),
            "weight": list(weight_alias),
        })
        if cmap["date"] is None or cmap["weight"] is None:
            raise ValueError(f"必須列が見つかりません: {fp}  columns={list(df.columns)[:20]}")
        sub = df[[cmap["date"], cmap["weight"]]].copy()
        sub.columns = ["__date__", "__weight__"]
        sub["__date__"] = _parse_date_series(sub["__date__"])
        sub["__weight__"] = pd.to_numeric(sub["__weight__"].astype(str).str.replace(",","",regex=False),
                                          errors="coerce")
        sub = sub.dropna(subset=["__date__","__weight__"])
        frames.append(sub)

    all_df = pd.concat(frames, ignore_index=True)
    g = all_df.groupby("__date__")["__weight__"].sum().rename("kg").to_frame()
    g.index.name = "date"
    g = g.sort_index()

    if max_daily_kg is not None and np.isfinite(max_daily_kg):
        g["kg"] = np.clip(g["kg"].values, 0.0, float(max_daily_kg))

    # 連続日付へ（欠損日は0）
    if len(g) == 0:
        raise ValueError("集計結果が空です。入力を確認してください。")
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full_idx, fill_value=0.0)
    return g


def load_reserve_daily(reserve_csv: Optional[str]) -> pd.DataFrame:
    if not reserve_csv: return pd.DataFrame()
    df = _read_csv_any(reserve_csv)
    cmap = _auto_map_columns(df, {
        "date": ["予約日","日付","伝票日付"],
        "count": ["台数","予約台数","件数","count"],
        "fixed": ["固定客","固定","fixed"],
    })
    if cmap["date"] is None:
        raise ValueError("予約データの日付列が見つかりません。")
    d = df[[cmap["date"]]].copy()
    d.columns = ["date"]
    d["date"] = _parse_date_series(d["date"])
    cnt = df[cmap["count"]] if cmap["count"] in df.columns else pd.Series(1, index=df.index)
    cnt = pd.to_numeric(cnt, errors="coerce").fillna(1.0)
    if cmap["fixed"] in df.columns:
        fx = df[cmap["fixed"]].astype(str).str.lower().isin(["1","true","yes","固定","固定客"]).astype(int)
    else:
        fx = pd.Series(0, index=df.index)
    d["cnt"] = cnt.values
    d["fx"]  = fx.values
    grp = d.groupby("date").agg(reserve_count=("cnt","count"),
                                reserve_sum=("cnt","sum"),
                                fixed_ratio=("fx","mean")).astype(float)
    return grp


# =========================
# Features
# =========================

def build_calendar_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=pd.DatetimeIndex(idx))
    d["dow"] = d.index.weekday
    d["weekofyear"] = d.index.isocalendar().week.astype(int)
    ang = 2*np.pi*d["dow"]/7.0
    d["dow_sin"] = np.sin(ang); d["dow_cos"] = np.cos(ang)
    # 月内位置
    d["day"] = d.index.day
    d["days_in_month"] = d.index.days_in_month
    d["month_pos"] = d["day"] / d["days_in_month"].replace(0,np.nan)
    return d


def add_month_order(fe: pd.DataFrame, daily_kg: pd.Series, active_mask: bool) -> pd.DataFrame:
    df = fe.copy()
    df["kg"] = daily_kg.values
    # 稼働日のみで順位（0開始正規化）
    if active_mask:
        active = df["kg"] > 0
    else:
        # 0kgも含めて順位
        active = pd.Series(True, index=df.index)
    grp = []
    for m, g in df.groupby([df.index.year, df.index.month]):
        g = g.copy()
        msk = active.loc[g.index]
        pos = np.zeros(len(g), dtype=float)
        if msk.sum() > 0:
            order_idx = np.where(msk)[0]
            # 0..(k-1) を k で割って 0-1に
            pos_vals = np.arange(len(order_idx), dtype=float)
            pos_vals = pos_vals / max(1.0, (len(order_idx)-1))
            pos[order_idx] = pos_vals
        g["month_active_pos"] = pos
        grp.append(g.drop(columns=["kg"]))
    out = pd.concat(grp).sort_index()
    return out


def make_design(
    idx: pd.DatetimeIndex,
    daily: pd.DataFrame,
    reserve: Optional[pd.DataFrame],
    cfg: Config
) -> pd.DataFrame:
    cal = build_calendar_feats(idx)
    # 稼働日順位
    cal = add_month_order(cal, daily["kg"], active_mask=cfg.active_day_mask)

    # 予約
    if reserve is not None and len(reserve) > 0:
        r = reserve.reindex(idx).fillna(0.0)
        if not cfg.use_same_day_reserve:
            r = r.shift(1).fillna(0.0)
        cal = cal.join(r)
    else:
        cal[["reserve_count","reserve_sum","fixed_ratio"]] = 0.0

    # 交互作用（曜日×月内位置など）
    if cfg.add_dow_monthpos:
        for base in ["month_pos","month_active_pos"]:
            if base in cal.columns:
                d = pd.get_dummies(cal["dow"].astype(int), prefix="dow", drop_first=False)
                # 一括生成（高速）
                Xint = d.multiply(cal[base], axis=0)
                Xint.columns = [f"{base}__{c}" for c in Xint.columns]
                cal = pd.concat([cal, Xint], axis=1)

    return cal.astype(float)


# =========================
# Train & predict (walk-forward)
# =========================

def fit_model(X: pd.DataFrame, y: np.ndarray, cfg: Config):
    scaler = StandardScaler()
    selector = VarianceThreshold(1e-5)

    Xt = selector.fit_transform(scaler.fit_transform(X.values))
    sw = _time_decay_weights(len(Xt), cfg.time_decay)

    gbr = GradientBoostingRegressor(
        loss="huber",
        alpha=0.9,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=cfg.random_state,
    )
    try:
        gbr.fit(Xt, y, sample_weight=sw)
    except TypeError:
        gbr.fit(Xt, y)

    pack = {"scaler": scaler, "selector": selector, "model": gbr, "cols": list(X.columns)}
    return pack


def predict_model(pack, X_today: pd.DataFrame) -> np.ndarray:
    X_today = X_today.reindex(columns=pack["cols"], fill_value=0.0)
    Xt = pack["selector"].transform(pack["scaler"].transform(X_today.values))
    p = pack["model"].predict(Xt)
    # 比率のため [0,1] へクリップ
    return np.clip(p, 0.0, 1.0)


def run(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load
    raw_csvs = args.raw_csvs or ([args.raw_csv] if args.raw_csv else None)
    if not raw_csvs:
        raise ValueError(" --raw-csvs もしくは --raw-csv を指定してください。")
    daily = load_daily_total(raw_csvs, max_daily_kg=args.max_daily_kg)
    reserve = load_reserve_daily(args.reserve_csv)

    # 月単位の集計（> min_month_total のみ使用）
    df = daily.copy()
    df["month"] = df.index.to_period("M")
    month_total = df.groupby("month")["kg"].sum().rename("month_total")
    # 低ボリューム月を除外
    valid_month = month_total[month_total >= args.min_month_total].index
    df = df[df["month"].isin(valid_month)].copy()
    # 活性日数フィルタ
    act_days = df.groupby("month").apply(lambda g: (g["kg"] > 0).sum())
    valid_month = act_days[act_days >= args.min_active_days].index
    df = df[df["month"].isin(valid_month)].copy()

    # 真の比率
    month_total = df.groupby("month")["kg"].sum()
    df["ratio_true"] = df["kg"] / month_total.reindex(df["month"]).values
    df["ratio_true"] = df["ratio_true"].fillna(0.0).clip(0, 1)

    # 特徴量
    X_all = make_design(df.index, df[["kg"]], reserve, cfg=Config(
        max_train_months=args.max_train_months,
        min_month_total=args.min_month_total,
        min_active_days=args.min_active_days,
        max_daily_kg=args.max_daily_kg,
        time_decay=args.time_decay,
        use_same_day_reserve=args.use_same_day_reserve,
        add_reserve_dow_interactions=args.add_reserve_dow_interactions,
        active_day_mask=args.active_day_mask,
        hard_month_normalize=args.hard_month_normalize,
        add_dow_monthpos=args.add_dow_monthpos,
        random_state=args.random_state,
    ))

    # ---- Walk-forward
    results = []
    months_sorted = sorted(df["month"].unique())  # Period('YYYY-MM')
    max_hist = args.max_train_months

    for m in months_sorted:
        # 学習対象: 直近 max_hist ヶ月（m の前まで）
        train_months = [mm for mm in months_sorted if mm < m][-max_hist:]
        if len(train_months) == 0:
            continue
        tr_idx = df["month"].isin(train_months)
        va_idx = (df["month"] == m)

        Xtr = X_all.loc[tr_idx]
        ytr = df.loc[tr_idx, "ratio_true"].values

        pack = fit_model(Xtr, ytr, cfg=Config(
            max_train_months=args.max_train_months,
            min_month_total=args.min_month_total,
            min_active_days=args.min_active_days,
            max_daily_kg=args.max_daily_kg,
            time_decay=args.time_decay,
            use_same_day_reserve=args.use_same_day_reserve,
            add_reserve_dow_interactions=args.add_reserve_dow_interactions,
            active_day_mask=args.active_day_mask,
            hard_month_normalize=args.hard_month_normalize,
            add_dow_monthpos=args.add_dow_monthpos,
            random_state=args.random_state,
        ))

        Xva = X_all.loc[va_idx]
        pred_ratio = predict_model(pack, Xva)

        # 月内ハード正規化（希望する場合のみ）
        if args.hard_month_normalize:
            s = pred_ratio.sum()
            if s > 0:
                pred_ratio = pred_ratio / s

        # kg 再構成：真の月合計 × 予測比率
        m_total = month_total.loc[m]
        kg_pred = m_total * pred_ratio

        # 集計
        tmp = pd.DataFrame({
            "date": df.loc[va_idx].index,
            "kg_true": df.loc[va_idx, "kg"].values.astype(float),
            "kg_pred": kg_pred.astype(float),
            "ratio_true": df.loc[va_idx, "ratio_true"].values.astype(float),
            "ratio_pred": pred_ratio.astype(float),
            "month": str(m)
        }).set_index("date").sort_index()
        results.append(tmp)

    if not results:
        raise RuntimeError("評価対象がありません。学習条件（min_month_total 等）を見直してください。")

    res = pd.concat(results).sort_index()
    mae = float(mean_absolute_error(res["kg_true"].values, res["kg_pred"].values))
    r2  = float(r2_score(res["kg_true"].values, res["kg_pred"].values))

    scores = {
        "MAE_kg": mae,
        "R2_kg": r2,
        "n_days": int(len(res)),
        "max_train_months": args.max_train_months,
        "time_decay": args.time_decay,
        "cfg": {
            "active_day_mask": args.active_day_mask,
            "hard_month_normalize": args.hard_month_normalize,
            "add_dow_monthpos": args.add_dow_monthpos,
            "min_month_total": args.min_month_total,
            "min_active_days": args.min_active_days,
            "max_daily_kg": args.max_daily_kg,
        },
        "env": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "sklearn": __import__("sklearn").__version__,
            "platform": platform.platform(),
        }
    }

    # ---- Save
    res.to_csv(os.path.join(args.out_dir, "res_ratio_walkforward.csv"), encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "scores_ratio.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # 図
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')

        # 直近 ~ 430日程度だけ描画（重すぎ対策）
        show = res.tail(min(430, len(res)))

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(show.index, show["kg_true"], label="Actual(kg)")
        ax.plot(show.index, show["kg_pred"], "--", label="Pred from True Monthly × Ratio")
        ax.set_title("Daily kg (True) vs (Monthly True × Pred Ratio)")
        ax.legend()
        fig.tight_layout(); plt.savefig(os.path.join(args.out_dir, "pred_vs_actual_ratio.png")); plt.close()

        err = (res["kg_pred"] - res["kg_true"]).values
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(err, bins=40)
        ax.set_title("Error Histogram (kg_pred - kg_true)")
        fig.tight_layout(); plt.savefig(os.path.join(args.out_dir, "ratio_error_hist.png")); plt.close()
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # バンドル保存
    if args.save_bundle:
        try:
            import joblib
            bundle = {
                "config": asdict(Config(
                    max_train_months=args.max_train_months,
                    min_month_total=args.min_month_total,
                    min_active_days=args.min_active_days,
                    max_daily_kg=args.max_daily_kg,
                    time_decay=args.time_decay,
                    use_same_day_reserve=args.use_same_day_reserve,
                    add_reserve_dow_interactions=args.add_reserve_dow_interactions,
                    active_day_mask=args.active_day_mask,
                    hard_month_normalize=args.hard_month_normalize,
                    add_dow_monthpos=args.add_dow_monthpos,
                    random_state=args.random_state,
                )),
                # 直近学習の pack を保存（最後の月の器）
                "last_pack": pack,
                "feature_columns": pack["cols"],
                "scores": scores,
            }
            joblib.dump(bundle, args.save_bundle)
            print(f"[SAVED] bundle -> {args.save_bundle}")
        except Exception as e:
            print(f"[WARN] bundle save failed: {e}")

    print("\n=== Ratio Model Summary ===")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"[DONE] out_dir={args.out_dir}  elapsed={time.time()-t0:.1f}s")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="日次比率モデル（月合計×比率で日次kgを再構成）")
    # 入力：複数 or 単一
    ap.add_argument("--raw-csvs", nargs="+", help="原始CSVを複数指定（推奨）")
    ap.add_argument("--raw-csv", type=str, help="原始CSVを1つだけ指定（互換）")
    ap.add_argument("--reserve-csv", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)

    # 学習設定
    ap.add_argument("--max-train-months", type=int, default=36)
    ap.add_argument("--min-month-total", type=float, default=0.0)
    ap.add_argument("--min-active-days", type=int, default=8)
    ap.add_argument("--max-daily-kg", type=float, default=None)

    ap.add_argument("--time-decay", type=str, default="linear",
                    choices=["none","linear","exponential"])

    ap.add_argument("--use-same-day-reserve", dest="use_same_day_reserve", action="store_true")
    ap.add_argument("--no-same-day-reserve", dest="use_same_day_reserve", action="store_false")
    ap.set_defaults(use_same_day_reserve=True)

    ap.add_argument("--add-reserve-dow-interactions", action="store_true",
                    help="（将来拡張用）予約×曜日の交互作用を追加")
    ap.add_argument("--no-reserve-dow-interactions", dest="add_reserve_dow_interactions",
                    action="store_false")
    ap.set_defaults(add_reserve_dow_interactions=False)

    ap.add_argument("--active-day-mask", action="store_true",
                    help="0kgなど非稼働日を比率分配/学習から除外")
    ap.add_argument("--hard-month-normalize", action="store_true",
                    help="予測比率を月内で厳密に再正規化（比率和=1.0）")
    ap.add_argument("--add-dow-monthpos", action="store_true",
                    help="曜日×月内位置の交互作用を追加")
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--save-bundle", type=str, default=None,
                    help="joblibで学習器と前処理メタを保存するパス")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
