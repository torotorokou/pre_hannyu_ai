import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from functools import lru_cache
import logging
from typing import Iterable, Optional, List

LOGGER = logging.getLogger("new_model2.walkforward")
LOGGER.setLevel(logging.INFO)

def get_target_items(df_raw, top_n=5):
    return df_raw["品名"].value_counts().head(top_n).index.tolist()

def get_feature_list(target_items, extra_features=None):
    """Return ordered feature list including target-specific lags and global stats."""
    base_features = [
        *[f"{item}_前日値" for item in target_items],
        *[f"{item}_前週平均" for item in target_items],
        "合計_前日値",
        "合計_3日平均",
        "合計_前週平均",
        "月",
        "曜日",
        "週番号",
        "1台あたり重量_過去中央値",
        "祝日フラグ",
        "祝日前フラグ",
        "祝日後フラグ",
        "連休前フラグ",
        "連休後フラグ",
        "前営業日フラグ",
        "翌営業日フラグ",
        "予約件数",
        "予約合計台数",
        "固定客予約数",
        "上位得意先予約数",
    ]
    if extra_features:
        base_features += list(extra_features)
    # 重複防止
    seen = set()
    ordered = []
    for f in base_features:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered

FALLBACK_MIN_FEATURES = ["合計_前日値","合計_前週平均","曜日","週番号"]

def ensure_date_normalized(s):
    if isinstance(s, pd.Series):
        s = pd.to_datetime(s, errors='coerce').dt.tz_localize(None).dt.floor('D')
        return s
    return pd.to_datetime(s, errors='coerce').tz_localize(None).floor('D')

def _sanitize_allowed_features(feature_list: List[str]) -> List[str]:
    return [f for f in feature_list if isinstance(f, str) and f]

def _apply_allowed_mode(original: List[str], allowed, mode: str = "whitelist") -> List[str]:
    if allowed is None:
        return original
    allowed = set(_sanitize_allowed_features(list(allowed)))
    if mode == "blacklist":
        filtered = [f for f in original if f not in allowed]
        if not filtered:
            print("[ALLOW] blacklist によりゼロ -> original にフォールバック")
            return original
        return filtered
    # whitelist
    filtered = [f for f in original if f in allowed]
    if not filtered:
        # fallback minimal set
        minimal = [f for f in FALLBACK_MIN_FEATURES if f in original]
        if minimal:
            print(f"[ALLOW] whitelist で空 -> 最小構成へフォールバック {minimal}")
            return minimal
        print("[ALLOW] whitelist で空 & 最小構成も空 -> original 使用")
        return original
    return filtered

def train_and_predict_stage1(
    df_feat_today,
    df_past_feat,
    df_past_pivot,
    base_models,
    meta_model_proto,
    feature_list,
    target_items,
    stage1_eval,
    df_pivot,
):
    results = {}
    trained_models_dict = {}
    X_train = df_past_feat[feature_list]
    for item in target_items:
        y_train = df_past_pivot[item]
        scaler = StandardScaler()
        selector = VarianceThreshold(1e-4)
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_filtered = selector.fit_transform(X_train_scaled)

        trained_models = [
            clone(model).fit(X_train_filtered, y_train) for _, model in base_models
        ]
        meta_input_train = np.column_stack(
            [m.predict(X_train_filtered) for m in trained_models]
        )
        meta_model = clone(meta_model_proto).fit(meta_input_train, y_train)

        X_target = df_feat_today[feature_list]
        X_target_filtered = selector.transform(scaler.transform(X_target))
        meta_input_target = np.column_stack(
            [m.predict(X_target_filtered) for m in trained_models]
        )
        pred = meta_model.predict(meta_input_target)[0]

        results[f"{item}_予測"] = pred
        if len(trained_models_dict) == 0:
            trained_models_dict["meta_model"] = meta_model
            trained_models_dict["raw_feature_names"] = feature_list
            trained_models_dict["selector_support_mask"] = selector.get_support()
            trained_models_dict["scaler"] = scaler
            trained_models_dict["selector"] = selector
        
        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

    results["_models"] = trained_models_dict
    return results

def train_and_predict_stage2(
    all_stage1_rows, stage1_results, df_feat_today, target_items, stage2_config=None
):
    df_hist = pd.DataFrame(all_stage1_rows[:-1])
    X_train = df_hist.drop(columns=["合計"])
    y_train = df_hist["合計"]

    scaler = StandardScaler()
    selector = VarianceThreshold(1e-4)
    X_train_filtered = selector.fit_transform(scaler.fit_transform(X_train))

    stage2_config = stage2_config or {}
    gbdt = GradientBoostingRegressor(
        n_estimators=stage2_config.get('n_estimators', 150),
        learning_rate=stage2_config.get('learning_rate', 0.05),
        max_depth=stage2_config.get('max_depth', 4),
        random_state=stage2_config.get('random_state', 42),
    )
    gbdt.fit(X_train_filtered, y_train)

    X_target = {
        f"{item}_予測": [stage1_results[f"{item}_予測"]] for item in target_items
    }
    for col in df_feat_today.columns:
        if col not in X_target:
            X_target[col] = df_feat_today.iloc[0][col]
    X_target = pd.DataFrame(X_target)
    total_pred = gbdt.predict(selector.transform(scaler.transform(X_target)))[0]
    return total_pred

def full_walkforward(
    df_raw,
    holidays,
    df_reserve,
    df_weather,
    min_stage1_days,
    min_stage2_days,
    top_n=5,
    allowed_features=None,
    allowed_mode: str = "whitelist",
    verbose: bool = False,
):
    """Walk-forward prediction pipeline with feature gating."""
    print(f"[INFO] full_walkforward start top_n={top_n} mode={allowed_mode}")
    
    # WeightFeatureBuilder, ReserveFeatureBuilder を動的にインポート
    from .feature_builder import WeightFeatureBuilder, ReserveFeatureBuilder
    
    df_raw = df_raw.copy()
    df_raw["伝票日付"] = ensure_date_normalized(df_raw["伝票日付"])
    df_raw = df_raw.dropna(subset=["伝票日付"]).sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)
    print(f"[INIT] target_items={target_items}")

    df_feat, df_pivot = WeightFeatureBuilder(df_raw, target_items, holidays).build()
    df_reserve_feat_all = ReserveFeatureBuilder(df_reserve).build()
    df_weather_feat_all = df_weather.copy() if isinstance(df_weather, pd.DataFrame) else pd.DataFrame()
    
    feature_list = get_feature_list(
        target_items,
        extra_features=["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"],
    )
    original_feature_list = feature_list.copy()
    feature_list = _apply_allowed_mode(original_feature_list, allowed_features, allowed_mode)
    if not feature_list:
        feature_list = [f for f in FALLBACK_MIN_FEATURES if f in original_feature_list] or original_feature_list[:4]
        print(f"[FEATURES][FALLBACK] empty after filtering -> using {feature_list}")
    print(f"[FEATURES] use={len(feature_list)} (orig={len(original_feature_list)}) mode={allowed_mode}")

    all_actual: List[float] = []
    all_pred: List[float] = []
    all_stage1_rows = []
    prediction_dates = []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index
    print(f"[CONFIG] dates={len(dates)} min_stage1_days={min_stage1_days} min_stage2_days={min_stage2_days}")

    base_models_conf = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),  # 高速化のため50に削減
    ]
    
    last_model = None
    for i, target_date in enumerate(dates):
        if i < min_stage1_days:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(300)  # 高速化のため300に削減
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_today = df_reserve_feat_all[df_reserve_feat_all.index <= target_date]
        df_weather_today = df_weather_feat_all[df_weather_feat_all.index <= target_date]

        df_past_feat = df_past_feat.merge(df_reserve_today, left_index=True, right_index=True, how="left")
        df_past_feat = df_past_feat.merge(df_weather_today, left_index=True, right_index=True, how="left").fillna(0)

        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(df_reserve_today, left_index=True, right_index=True, how="left")
        df_feat_today = df_feat_today.merge(df_weather_today, left_index=True, right_index=True, how="left").fillna(0)

        stage1_result = train_and_predict_stage1(
            df_feat_today,
            df_past_feat,
            df_past_pivot,
            base_models=base_models_conf,
            meta_model_proto=ElasticNet(alpha=0.1, l1_ratio=0.5),
            feature_list=feature_list,
            target_items=target_items,
            stage1_eval=stage1_eval,
            df_pivot=df_pivot,
        )
        last_model = stage1_result
        
        row = {f"{item}_予測": stage1_result[f"{item}_予測"] for item in target_items}
        for col in df_feat_today.columns:
            if col not in row:
                row[col] = df_feat_today.iloc[0][col]
        row["合計"] = df_pivot.loc[target_date, "合計"]
        all_stage1_rows.append(row)

        if len(all_stage1_rows) > min_stage2_days:
            total_pred = train_and_predict_stage2(all_stage1_rows, stage1_result, df_feat_today, target_items)
            actual_val = df_pivot.loc[target_date, "合計"]
            all_actual.append(actual_val)
            all_pred.append(total_pred)
            prediction_dates.append(target_date)
            
            if len(all_actual) % 10 == 0:  # 10件ごとに進捗表示
                r2_now = r2_score(all_actual, all_pred)
                mae_now = mean_absolute_error(all_actual, all_pred)
                print(f"[PROGRESS] n={len(all_actual)} R2={r2_now:.3f} MAE={mae_now:,.0f}kg")

    if all_actual:
        final_r2 = r2_score(all_actual, all_pred)
        final_mae = mean_absolute_error(all_actual, all_pred)
        print(f"[FINAL] R2={final_r2:.3f} MAE={final_mae:,.0f}kg n={len(all_actual)}")
    else:
        print("[FINAL] stage2 predictions absent")

    return all_actual, all_pred, last_model, prediction_dates
