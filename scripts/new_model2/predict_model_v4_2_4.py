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
# --- Logger setup (idempotent) ---
if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s','%Y-%m-%d %H:%M:%S'))
    LOGGER.addHandler(sh)

def _attach_file_handler(log_dir: str = "data/cache", filename: str = "walkforward.log"):
    """Attach a rotating-like simple FileHandler if not already.

    Parameters
    ----------
    log_dir : str
        Directory to store the log file.
    filename : str
        Log file name.
    """
    import os, time, glob
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)
        # Avoid duplicate file handlers
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path) for h in LOGGER.handlers):
            # Simple rotation: if file > 5MB rename with timestamp
            if os.path.exists(log_path) and os.path.getsize(log_path) > 5 * 1024 * 1024:
                ts = time.strftime('%Y%m%d_%H%M%S')
                os.rename(log_path, os.path.join(log_dir, f"walkforward_{ts}.log"))
                # Optional: prune older than 7 files
                old = sorted(glob.glob(os.path.join(log_dir, 'walkforward_*.log')))[:-7]
                for o in old:
                    try: os.remove(o)
                    except OSError: pass
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s','%Y-%m-%d %H:%M:%S'))
            LOGGER.addHandler(fh)
            LOGGER.debug("[LOGGER] FileHandler attached: %s", log_path)
    except Exception as e:
        # Failing to attach file logging must not break core logic
        LOGGER.warning("[LOGGER] attach file handler failed: %s", e)

_attach_file_handler()
LOGGER.setLevel(logging.INFO)

def export_reduction_history(history_df, out_dir: str = "data/cache", prefix: str = "feature_reduction"):
    """Persist feature reduction trial history to CSV & JSON.

    history_df : pd.DataFrame | None
        DataFrame produced by reduction loop (must include columns like removed_feature, mae, r2, accept_flag, etc.).
    out_dir : str
        Directory for artifacts.
    prefix : str
        Filename prefix.
    Returns
    -------
    dict with paths (csv, json) or empty if failed.
    """
    import os, json
    artifacts = {}
    if history_df is None or len(getattr(history_df, 'index', [])) == 0:
        return artifacts
    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(out_dir, f"{prefix}_{ts}.csv")
        json_path = os.path.join(out_dir, f"{prefix}_{ts}.json")
        # Ensure serialisable
        history_df.to_csv(csv_path, index=False)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
        LOGGER.info("[EXPORT] reduction history saved csv=%s json=%s", csv_path, json_path)
        artifacts = {"csv": csv_path, "json": json_path}
    except Exception as e:
        LOGGER.warning("[EXPORT] failed: %s", e)
    return artifacts

# 特徴量作成用ビルダーの読み込み（相対 import）
from .feature_builder import (
    WeightFeatureBuilder,
    ReserveFeatureBuilder,
)


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
        "曜日",
        "週番号",
        "1台あたり重量_過去中央値",
        "祝日フラグ",
        "祝日前フラグ",
        "祝日後フラグ",
        "連休前フラグ",
        "連休後フラグ",
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
_WEATHER_CACHE_KEYS = set()

def ensure_date_normalized(s):
    import pandas as _pd
    if isinstance(s, _pd.Series):
        s = _pd.to_datetime(s, errors='coerce').dt.tz_localize(None).dt.floor('D')
        return s
    return _pd.to_datetime(s, errors='coerce').tz_localize(None).floor('D')

def _sanitize_allowed_features(feature_list: List[str]) -> List[str]:
    return [f for f in feature_list if isinstance(f, str) and f]

def _apply_allowed_mode(original: List[str], allowed, mode: str = "whitelist") -> List[str]:
    if allowed is None:
        return original
    allowed = set(_sanitize_allowed_features(list(allowed)))
    if mode == "blacklist":
        filtered = [f for f in original if f not in allowed]
        if not filtered:
            LOGGER.warning("[ALLOW] blacklist によりゼロ -> original にフォールバック")
            return original
        return filtered
    # whitelist
    filtered = [f for f in original if f in allowed]
    if not filtered:
        # fallback minimal set
        minimal = [f for f in FALLBACK_MIN_FEATURES if f in original]
        if minimal:
            LOGGER.warning("[ALLOW] whitelist で空 -> 最小構成へフォールバック %s", minimal)
            return minimal
        LOGGER.warning("[ALLOW] whitelist で空 & 最小構成も空 -> original 使用")
        return original
    return filtered

_FW_RUNNING = False  # 再入防止フラグ

@lru_cache(maxsize=32)
def _cached_weather_range(start_iso: str, end_iso: str, enable_fallback: bool = True):
    try:
        from .feature_builder import WeatherFeatureBuilder
        wb = WeatherFeatureBuilder(start_iso, end_iso, enable_fallback=enable_fallback)
        dfw = wb.build()
        LOGGER.info("[Weather][cache_miss] rows=%s", len(dfw))
        return dfw
    except Exception as exc:
        LOGGER.warning("[Weather] cache build failed: %s", exc)
        import pandas as _pd
        idx = _pd.date_range(start_iso, end_iso, freq='D')
        return _pd.DataFrame(index=idx)

def get_weather_cached(start, end, enable_fallback=True):
    import pandas as _pd
    if isinstance(start, _pd.Timestamp):
        start_iso = start.date().isoformat()
    else:
        start_iso = str(start)[:10]
    if isinstance(end, _pd.Timestamp):
        end_iso = end.date().isoformat()
    else:
        end_iso = str(end)[:10]
    key = (start_iso, end_iso, enable_fallback)
    if key in _WEATHER_CACHE_KEYS:
        LOGGER.info("[Weather][cache_hit] %s -> %s", start_iso, end_iso)
    else:
        _WEATHER_CACHE_KEYS.add(key)
    dfw = _cached_weather_range(start_iso, end_iso, enable_fallback)
    return dfw


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
    trained_models_dict = {}  # 学習済みモデルを保存
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
        # 最初のアイテムのメタモデルを保存（特徴量重要度抽出用）
        if len(trained_models_dict) == 0:
            # メタモデルと共に前処理器を保存し、実際に使用された列インデックスを保持
            trained_models_dict["meta_model"] = meta_model
            trained_models_dict["raw_feature_names"] = feature_list  # フィルタ前
            # VarianceThreshold は特定列を除外する可能性があるため mask を作成
            vt_support = selector.get_support()
            trained_models_dict["selector_support_mask"] = vt_support
            trained_models_dict["scaler"] = scaler
            trained_models_dict["selector"] = selector
        
        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

    # 予測結果とモデル情報を両方返す
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


def evaluate_stage1(stage1_eval, target_items):
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        print(
            f"{item}: R² = {r2_score(y_true, y_pred):.3f}, MAE = {mean_absolute_error(y_true, y_pred):,.0f}kg"
        )


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
    min_stage2_rows: Optional[int] = None,
    fallback_min_features: Optional[Iterable[str]] = None,
    verbose: bool = False,
    model_profile: str = "full",
    disable_stage1_eval: bool = False,
    rf_estimators_fast: int = 30,
    stage2_estimators_fast: int = 60,
):
    """Walk-forward prediction pipeline with feature gating & reentrancy protection.

    Parameters
    ----------
    allowed_features : list[str] | None
        Interpreted under allowed_mode. Whitelist => keep intersection; blacklist => remove.
        Fallback strategies prevent empty feature lists.
    allowed_mode : str
        "whitelist" or "blacklist".
    min_stage2_rows : int | None
        If provided overrides min_stage2_days threshold for stage2 start.
    fallback_min_features : list[str] | None
        Minimal safe feature subset used when whitelist produces empty result.
    """
    global _FW_RUNNING
    if _FW_RUNNING:
        LOGGER.warning("[REENTRY] full_walkforward 呼び出しをスキップ (running)")
        return [], [], None, []
    _FW_RUNNING = True
    try:
        if verbose:
            print(f"[DEBUG] ▶ full_walkforward start top_n={top_n} mode={allowed_mode}")
        LOGGER.info("▶ full_walkforward(new_model2) start top_n=%s allowed_mode=%s", top_n, allowed_mode)
        df_raw = df_raw.copy()
        df_raw["伝票日付"] = ensure_date_normalized(df_raw["伝票日付"])  # 正規化
        df_raw = df_raw.dropna(subset=["伝票日付"]).sort_values("伝票日付")
        target_items = get_target_items(df_raw, top_n)
        LOGGER.info("[INIT] target_items=%s", target_items)

        df_feat, df_pivot = WeightFeatureBuilder(df_raw, target_items, holidays).build()
        df_reserve_feat_all = ReserveFeatureBuilder(df_reserve).build()
        df_weather_feat_all = df_weather.copy() if isinstance(df_weather, pd.DataFrame) else pd.DataFrame()
        
        # [COPILOT-MOD] Index date normalization with enhanced safety
        try:
            _norm_res = ensure_date_normalized(pd.Series(df_reserve_feat_all.index))
            if isinstance(_norm_res, pd.Series) and len(_norm_res) > 0:
                clean_dates = _norm_res.dropna().sort_values().drop_duplicates()
                if len(clean_dates) > 0:
                    df_reserve_feat_all = df_reserve_feat_all.reindex(clean_dates)
        except Exception as exc:
            LOGGER.debug("[DATE] reserve index normalize failed: %s", exc)
        try:
            _norm_w = ensure_date_normalized(pd.Series(df_weather_feat_all.index))
            if isinstance(_norm_w, pd.Series) and len(_norm_w) > 0:
                clean_dates = _norm_w.dropna().sort_values().drop_duplicates()
                if len(clean_dates) > 0:
                    df_weather_feat_all = df_weather_feat_all.reindex(clean_dates)
        except Exception as exc:
            LOGGER.debug("[DATE] weather index normalize failed: %s", exc)
        
        # [COPILOT-MOD] Ensure main feature dataframe index is also properly normalized
        try:
            _norm_feat = ensure_date_normalized(pd.Series(df_feat.index))
            if isinstance(_norm_feat, pd.Series) and len(_norm_feat) > 0:
                clean_dates = _norm_feat.dropna().sort_values().drop_duplicates()
                if len(clean_dates) > 0:
                    df_feat = df_feat.reindex(clean_dates)
            _norm_pivot = ensure_date_normalized(pd.Series(df_pivot.index))
            if isinstance(_norm_pivot, pd.Series) and len(_norm_pivot) > 0:
                clean_dates = _norm_pivot.dropna().sort_values().drop_duplicates()
                if len(clean_dates) > 0:
                    df_pivot = df_pivot.reindex(clean_dates)
        except Exception as exc:
            LOGGER.debug("[DATE] main dataframe index normalize failed: %s", exc)

        feature_list = get_feature_list(
            target_items,
            extra_features=["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"],
        )
        original_feature_list = feature_list.copy()
        global FALLBACK_MIN_FEATURES
        if fallback_min_features:
            FALLBACK_MIN_FEATURES = [f for f in fallback_min_features if isinstance(f, str)]
        feature_list = _apply_allowed_mode(original_feature_list, allowed_features, allowed_mode)
        if not feature_list:
            feature_list = [f for f in FALLBACK_MIN_FEATURES if f in original_feature_list] or original_feature_list[:4]
            LOGGER.warning("[FEATURES][FALLBACK] empty after filtering -> using %s", feature_list)
        LOGGER.info("[FEATURES] use=%s (orig=%s) mode=%s", len(feature_list), len(original_feature_list), allowed_mode)

        all_actual: List[float] = []
        all_pred: List[float] = []
        all_stage1_rows = []
        prediction_dates = []
        stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
        dates = df_feat.index
        effective_min_stage2 = min_stage2_rows if min_stage2_rows is not None else min_stage2_days
        LOGGER.info("[CONFIG] dates=%s min_stage1_days=%s min_stage2_days=%s eff_stage2_rows=%s", len(dates), min_stage1_days, min_stage2_days, effective_min_stage2)
        if verbose:
            print(f"[DEBUG] dates_len={len(dates)} min_stage1_days={min_stage1_days} min_stage2_days={min_stage2_days}")
        if len(dates) <= min_stage1_days + effective_min_stage2:
            LOGGER.warning("[WARN] 日数不足: stage2 予測が限定的になる可能性")

        last_reported_stage2_rows = -1
        stage2_skip_reasons_counter = {"insufficient_stage1_rows":0}
        stage2_runs = 0
        # モデルプロファイル設定
        if model_profile not in {"full","fast"}:
            LOGGER.warning("[PROFILE] unknown model_profile=%s -> full", model_profile)
            model_profile_local = "full"
        else:
            model_profile_local = model_profile
        if model_profile_local == "fast":
            base_models_conf = [
                ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ]
            stage2_conf = {"n_estimators": stage2_estimators_fast, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}
        else:
            base_models_conf = [
                ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
            stage2_conf = {}

        for i, target_date in enumerate(dates):
            if i < min_stage1_days:
                if i % 5 == 0:
                    LOGGER.debug("[SKIP] %s i=%s < min_stage1_days=%s", target_date.date(), i, min_stage1_days)
                if verbose and i % 5 == 0:
                    print(f"[SKIP] {target_date.date()} (i={i}) < min_stage1_days={min_stage1_days}")
                continue

            df_past_feat = df_feat[df_feat.index < target_date].tail(600)
            df_past_pivot = df_pivot.loc[df_past_feat.index]

            df_reserve_today = df_reserve_feat_all[df_reserve_feat_all.index <= target_date]
            df_weather_today = df_weather_feat_all[df_weather_feat_all.index <= target_date]

            df_past_feat = df_past_feat.merge(df_reserve_today, left_index=True, right_index=True, how="left")
            df_past_feat = df_past_feat.merge(df_weather_today, left_index=True, right_index=True, how="left").fillna(0)

            df_feat_today = df_feat.loc[[target_date]].copy()
            df_feat_today = df_feat_today.merge(df_reserve_today, left_index=True, right_index=True, how="left")
            df_feat_today = df_feat_today.merge(df_weather_today, left_index=True, right_index=True, how="left").fillna(0)

            LOGGER.info("[DAY] %s (%s/%s) stage1_rows=%s", target_date.date(), i, len(dates)-1, len(all_stage1_rows))
            stage1_result = train_and_predict_stage1(
                df_feat_today,
                df_past_feat,
                df_past_pivot,
                base_models=base_models_conf,
                meta_model_proto=ElasticNet(alpha=0.1, l1_ratio=0.5),
                feature_list=feature_list,
                target_items=target_items,
                stage1_eval=stage1_eval if not disable_stage1_eval else {item:{"y_true":[],"y_pred":[]} for item in target_items},
                df_pivot=df_pivot,
            )
            try:
                stage1_result['used_features'] = list(feature_list)
            except Exception:
                pass
            row = {f"{item}_予測": stage1_result[f"{item}_予測"] for item in target_items}
            for col in df_feat_today.columns:
                if col not in row:
                    row[col] = df_feat_today.iloc[0][col]
            row["合計"] = df_pivot.loc[target_date, "合計"]
            all_stage1_rows.append(row)

            if len(all_stage1_rows) <= effective_min_stage2 and len(all_stage1_rows) != last_reported_stage2_rows:
                LOGGER.debug("[STAGE2_WAIT] rows=%s/%s reasons=%s", len(all_stage1_rows), effective_min_stage2 + 1, ["insufficient_stage1_rows"]) 
                stage2_skip_reasons_counter["insufficient_stage1_rows"] += 1
                last_reported_stage2_rows = len(all_stage1_rows)

            if len(all_stage1_rows) > effective_min_stage2:
                total_pred = train_and_predict_stage2(all_stage1_rows, stage1_result, df_feat_today, target_items, stage2_config=stage2_conf)
                actual_val = df_pivot.loc[target_date, "合計"]
                all_actual.append(actual_val)
                all_pred.append(total_pred)
                prediction_dates.append(target_date)
                stage2_runs += 1
                if len(all_actual) >= 3:
                    r2_now = r2_score(all_actual, all_pred)
                    mae_now = mean_absolute_error(all_actual, all_pred)
                    LOGGER.info("[PROGRESS] n=%s R2=%.3f MAE=%skg", len(all_actual), r2_now, f"{mae_now:,.0f}")
        if verbose:
            print(f"[DEBUG] stage2_runs={stage2_runs}")

        if not all_actual and all_stage1_rows:
            burn_in_idx = min_stage1_days
            fallback_rows = all_stage1_rows[burn_in_idx:]
            if fallback_rows:
                LOGGER.warning("[FALLBACK][STAGE2] using stage1 sum predictions (rows=%s)", len(fallback_rows))
                for r in fallback_rows:
                    actual_val = r.get("合計")
                    pred_sum = 0.0
                    for k,v in r.items():
                        if k.endswith("_予測") and isinstance(v,(int,float)):
                            pred_sum += v
                    if actual_val is not None:
                        all_actual.append(actual_val)
                        all_pred.append(pred_sum)
        # Summary logging (dates length, skip counts, stage2 run count)
        try:
            skip_insufficient = stage2_skip_reasons_counter.get("insufficient_stage1_rows", 0)
        except Exception:
            skip_insufficient = 0
        LOGGER.info(
            "[SUMMARY] dates=%s stage2_runs=%s skip_insufficient_stage1=%s", len(dates), stage2_runs, skip_insufficient
        )
        if verbose:
            print(f"[DEBUG] summary: dates_len={len(dates)} stage2_runs={stage2_runs} skip_insufficient_stage1={skip_insufficient}")
        LOGGER.info("[STAGE2] evaluation")
        if all_actual:
            final_r2 = r2_score(all_actual, all_pred)
            final_mae = mean_absolute_error(all_actual, all_pred)
            LOGGER.info("[FINAL] R2=%.3f MAE=%skg n=%s", final_r2, f"{final_mae:,.0f}", len(all_actual))
        else:
            LOGGER.warning("[FINAL] stage2 predictions absent (all_actual=0) skip_reasons=%s", stage2_skip_reasons_counter)

        # ステージ1評価出力（軽量モードではスキップ）
        if not disable_stage1_eval:
            try:
                evaluate_stage1(stage1_eval, target_items)
            except Exception as exc:
                LOGGER.debug("[STAGE1][EVAL][SKIP] %s", exc)
        last_model = locals().get('stage1_result')
        return all_actual, all_pred, last_model, prediction_dates
    finally:
        _FW_RUNNING = False
