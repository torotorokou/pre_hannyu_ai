from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
try:  # sklearn<1.0で必要だった互換ガード
    from sklearn.experimental import enable_hist_gradient_boosting  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pass
from sklearn.ensemble import HistGradientBoostingRegressor


# Optional integrations (graceful degradation if unavailable)
try:  # LightGBM
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional
    LGBMRegressor = None

try:  # XGBoost
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional
    XGBRegressor = None

try:  # CatBoost
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover - optional
    CatBoostRegressor = None


# -----------------------------
# Data utilities
# -----------------------------


def infer_feature_types(df: pd.DataFrame, target: str, date_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical columns for preprocessing. Optionally drops target/date.

    Returns:
        (numeric_cols, categorical_cols)
    """
    cols = [c for c in df.columns if c != target and (date_col is None or c != date_col)]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def add_datetime_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """
    If date_col is provided, add simple calendar features and drop original date column.
    """
    if date_col is None or date_col not in df.columns:
        return df
    ser = pd.to_datetime(df[date_col])
    feat = pd.DataFrame(
        {
            "year": ser.dt.year,
            "month": ser.dt.month,
            "day": ser.dt.day,
            "dayofweek": ser.dt.dayofweek,
            "weekofyear": ser.dt.isocalendar().week.astype(int),
        },
        index=df.index,
    )
    out = df.drop(columns=[date_col]).copy()
    for c in feat.columns:
        out[f"dt_{c}"] = feat[c]
    return out


def build_preprocessor(df: pd.DataFrame, target: str, date_col: Optional[str]) -> ColumnTransformer:
    df2 = add_datetime_features(df, date_col)
    num_cols, cat_cols = infer_feature_types(df2, target, date_col=None)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    # sklearn>=1.2 uses sparse_output; keep dense to support many estimators
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return pre


# -----------------------------
# OOF and stacking
# -----------------------------


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def oof_predict(
    model_specs: List[ModelSpec],
    df: pd.DataFrame,
    target: str,
    date_col: Optional[str] = None,
    n_splits: int = 5,
    time_series_cv: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], List[List[BaseEstimator]]]:
    """
    Create out-of-fold predictions for each base model.

    Returns:
        oof_df: DataFrame with one column per model containing OOF predictions.
        metrics: dict per model with rmse/mae/r2 on OOF.
        fitted_models_per_fold: list of list of fitted estimators per fold.
    """
    df_proc = add_datetime_features(df, date_col)
    y = df_proc[target].values
    X = df_proc.drop(columns=[target])

    pre = build_preprocessor(df, target, date_col)

    cv = TimeSeriesSplit(n_splits=n_splits) if time_series_cv else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = {spec.name: np.zeros_like(y, dtype=float) for spec in model_specs}
    fitted_models_per_fold: List[List[BaseEstimator]] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fold_models: List[BaseEstimator] = []
        for spec in model_specs:
            est = clone(spec.estimator)
            pipe = Pipeline([("pre", pre), ("est", est)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            oof[spec.name][va_idx] = pred
            fold_models.append(pipe)
        fitted_models_per_fold.append(fold_models)

    oof_df = pd.DataFrame(oof, index=df.index)
    metrics: Dict[str, Dict[str, float]] = {}
    for spec in model_specs:
        pred = oof_df[spec.name].values
        metrics[spec.name] = {
            "rmse": _rmse(y, pred),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    return oof_df, metrics, fitted_models_per_fold


def build_meta_features(oof_df: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
    feats = pd.DataFrame(index=oof_df.index)
    cols = list(oof_df.columns)
    feats["blend_mean"] = oof_df.mean(axis=1)

    # Rank-based blend
    ranks = oof_df.rank(axis=1, method="average")
    feats["blend_rank_mean"] = ranks.mean(axis=1)

    feats["blend_std"] = oof_df.std(axis=1)

    # Weighted mean: by inverse RMSE if y provided, else inverse std as proxy
    weights = []
    if y_true is not None and len(oof_df) > 1:
        for c in cols:
            rmse = math.sqrt(mean_squared_error(y_true, oof_df[c].values))
            w = 1.0 / (rmse + 1e-6)
            if not np.isfinite(w) or w <= 0:
                w = 1.0
            weights.append(w)
    else:
        # 単一行や分散0の場合は等重みへフォールバック
        for c in cols:
            std = float(oof_df[c].std())
            w = 1.0 / (std + 1e-6)
            if (not np.isfinite(w)) or (len(oof_df) <= 1) or (std == 0.0):
                w = 1.0
            weights.append(w)
    w = np.array(weights, dtype=float)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / s
    feats["blend_wmean"] = (oof_df.values @ w).astype(float)

    # pairwise interactions (limited to top 3 columns to keep it light)
    for i, ci in enumerate(cols[:3]):
        for j, cj in enumerate(cols[:3]):
            if j <= i:
                continue
            feats[f"mix_{ci}_x_{cj}"] = (oof_df[ci] + oof_df[cj]) * 0.5
    # 予防的にNaNを0で埋める（Huber等がNaN非対応のため）
    return feats.fillna(0.0)


def fit_meta_model(
    meta_estimator: BaseEstimator,
    meta_X: pd.DataFrame,
    y: np.ndarray,
) -> BaseEstimator:
    """
    メタモデルを学習する。
    - HuberRegressor の場合はスケーリング＋反復回数増で安定化
    - それでも収束しない／エラー時は Ridge にフォールバック

    Returns: 学習済み推論器（Pipeline を含むことあり）
    """
    est = clone(meta_estimator)

    # Huber を使う場合は前処理を挟んで安定化
    if isinstance(est, HuberRegressor):
        huber = HuberRegressor(max_iter=5000, tol=1e-5)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("est", huber),
        ])
        try:
            with warnings.catch_warnings():
                # 収束しない場合を例外として扱い、フォールバックへ
                warnings.simplefilter("error", ConvergenceWarning)
                pipe.fit(meta_X, y)
            return pipe
        except Exception:
            # フォールバック: Ridge
            fallback = Pipeline([
                ("scaler", StandardScaler()),
                ("est", Ridge(alpha=1.0, random_state=42)),
            ])
            fallback.fit(meta_X, y)
            return fallback

    # 通常ケース: そのまま学習
    est.fit(meta_X, y)
    return est


def fit_full_models(model_specs: List[ModelSpec], df: pd.DataFrame, target: str, date_col: Optional[str] = None) -> Dict[str, BaseEstimator]:
    df_proc = add_datetime_features(df, date_col)
    y = df_proc[target].values
    X = df_proc.drop(columns=[target])
    pre = build_preprocessor(df, target, date_col)
    fitted: Dict[str, BaseEstimator] = {}
    for spec in model_specs:
        pipe = Pipeline([("pre", pre), ("est", clone(spec.estimator))])
        pipe.fit(X, y)
        fitted[spec.name] = pipe
    return fitted


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------
# Preset configurations A-D
# -----------------------------


def _lgbm_or_histgbdt(random_state=42):
    if LGBMRegressor is not None:
        return LGBMRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    return HistGradientBoostingRegressor(random_state=42, max_depth=None)


def _xgb_or_histgbdt(random_state=42):
    if XGBRegressor is not None:
        return XGBRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist")
    return HistGradientBoostingRegressor(random_state=42, max_depth=None)


def _cat_or_rf(random_state=42):
    if CatBoostRegressor is not None:
        return CatBoostRegressor(verbose=0, random_state=random_state, depth=8, learning_rate=0.05, iterations=500)
    return RandomForestRegressor(random_state=random_state, n_estimators=400, max_depth=None, n_jobs=-1)


def preset_A(random_state=42) -> Tuple[List[ModelSpec], BaseEstimator]:
    """(A) 木系＋線形＋NNハイブリッド"""
    base = [
        ModelSpec("lgbm_like", _lgbm_or_histgbdt(random_state)),
        ModelSpec("cat_like", _cat_or_rf(random_state)),
        ModelSpec("elasticnet", ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=random_state)),
        ModelSpec("mlp_small", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", random_state=random_state, max_iter=300)),
    ]
    meta = Ridge(alpha=1.0, random_state=random_state)
    return base, meta


def preset_B(random_state=42) -> Tuple[List[ModelSpec], BaseEstimator]:
    """(B) アンサンブル・オブ・アンサンブル"""
    base = [
        ModelSpec("rf", RandomForestRegressor(random_state=random_state, n_estimators=600, n_jobs=-1)),
        ModelSpec("xgb_like", _xgb_or_histgbdt(random_state)),
        ModelSpec("lgbm_like", _lgbm_or_histgbdt(random_state)),
        ModelSpec("cat_like", _cat_or_rf(random_state)),
    ]
    meta = _xgb_or_histgbdt(random_state)
    return base, meta


def preset_C(random_state=42) -> Tuple[List[ModelSpec], BaseEstimator]:
    """(C) 多様モデル＋シンプルスタッカー"""
    base = [
        ModelSpec("lgbm_like", _lgbm_or_histgbdt(random_state)),
        ModelSpec("knn", KNeighborsRegressor(n_neighbors=8, weights="distance")),
        ModelSpec("svm", SVR(C=2.0, epsilon=0.1, kernel="rbf")),
        ModelSpec("mlp_small", MLPRegressor(hidden_layer_sizes=(64,), activation="relu", random_state=random_state, max_iter=300)),
    ]
    meta = LinearRegression()
    return base, meta


def preset_D(random_state=42) -> Tuple[List[ModelSpec], BaseEstimator]:
    """(D) 特殊アンサンブル（簡易版）
    本来は特徴ごとに異なる前処理・モデルを使い分けるが、汎用的に動作する程度に落とし込む。
    - 数値: GBDT
    - カテゴリ: 線形（OneHot）
    - 時間: MLP（add_datetime_featuresで作ったdt_系）
    ここでは全体学習に同じ前処理を使うため、モデルの多様性のみ担保する。
    """
    base = [
        ModelSpec("numeric_gbdt", _lgbm_or_histgbdt(random_state)),
        ModelSpec("categorical_linear", Ridge(alpha=0.5, random_state=random_state)),
        ModelSpec("time_mlp", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", random_state=random_state, max_iter=300)),
    ]
    meta = _xgb_or_histgbdt(random_state) if XGBRegressor is not None else Ridge(alpha=1.0, random_state=random_state)
    return base, meta


PRESETS: Dict[str, Callable[[int], Tuple[List[ModelSpec], BaseEstimator]]] = {
    "A": preset_A,
    "B": preset_B,
    "C": preset_C,
    "D": preset_D,
}


# -----------------------------
# High-level runner
# -----------------------------


def run_stacking(
    df: pd.DataFrame,
    target: str,
    date_col: Optional[str],
    preset: str = "A",
    n_splits: int = 5,
    time_series_cv: bool = False,
    robust_meta: bool = False,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    assert preset in PRESETS, f"Unknown preset: {preset}"
    base_specs, default_meta = PRESETS[preset](random_state)
    meta_est = HuberRegressor() if robust_meta else default_meta

    oof_df, base_metrics, _ = oof_predict(
        base_specs,
        df=df,
        target=target,
        date_col=date_col,
        n_splits=n_splits,
        time_series_cv=time_series_cv,
        random_state=random_state,
    )

    df_proc = add_datetime_features(df, date_col)
    y = df_proc[target].values

    meta_X = pd.concat([oof_df, build_meta_features(oof_df, y_true=y)], axis=1)
    meta_model = fit_meta_model(meta_est, meta_X, y)
    oof_meta_pred = meta_model.predict(meta_X)
    meta_metrics = evaluate_predictions(y, oof_meta_pred)

    # Full refit for deployment
    full_models = fit_full_models(base_specs, df, target, date_col)

    results = {
        "preset": preset,
        "base_metrics": base_metrics,
        "meta_metrics": meta_metrics,
        "oof_columns": list(oof_df.columns),
        "meta_feature_columns": list(set(meta_X.columns) - set(oof_df.columns)),
        "meta_model": meta_model,
        "full_models": full_models,
        "oof_df": oof_df,
        "y": y,
    }

    # Persist artifacts
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # metrics
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"base": base_metrics, "meta": meta_metrics}, f, ensure_ascii=False, indent=2)
        # oof
        oof_df.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=True)
        # meta preds
        pd.DataFrame({"y": y, "meta_oof": oof_meta_pred}, index=df.index).to_csv(
            os.path.join(output_dir, "oof_meta.csv"), index=True
        )
        # save models (optional runtime dependency)
        try:
            import joblib  # lazy import

            joblib.dump(meta_model, os.path.join(output_dir, "meta_model.joblib"))
            for name, model in full_models.items():
                joblib.dump(model, os.path.join(output_dir, f"model_{name}.joblib"))
        except Exception:
            pass

    return results
