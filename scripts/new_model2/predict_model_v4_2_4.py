import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# 特徴量作成用ビルダーの読み込み（相対 import）
from .feature_builder import (
    WeightFeatureBuilder,
    ReserveFeatureBuilder,
)


def get_target_items(df_raw, top_n=5):
    return df_raw["品名"].value_counts().head(top_n).index.tolist()


def get_feature_list(target_items, extra_features=None):
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
        base_features += extra_features
    return base_features


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
    all_stage1_rows, stage1_results, df_feat_today, target_items
):
    df_hist = pd.DataFrame(all_stage1_rows[:-1])
    X_train = df_hist.drop(columns=["合計"])
    y_train = df_hist["合計"]

    scaler = StandardScaler()
    selector = VarianceThreshold(1e-4)
    X_train_filtered = selector.fit_transform(scaler.fit_transform(X_train))

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
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
):
    print("▶️ full_walkforward(new_model2) 開始")
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)
    print(f"[DEBUG] target_items={target_items}")

    df_feat, df_pivot = WeightFeatureBuilder(df_raw, target_items, holidays).build()
    df_reserve_feat_all = ReserveFeatureBuilder(df_reserve).build()
    df_weather_feat_all = df_weather.copy() if isinstance(df_weather, pd.DataFrame) else pd.DataFrame()

    feature_list = get_feature_list(
        target_items,
        extra_features=["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"],
    )
    original_feature_list = feature_list.copy()
    if allowed_features is not None:
        # preserve order while filtering
        feature_list = [f for f in feature_list if f in set(allowed_features)]
        dropped = [f for f in original_feature_list if f not in feature_list]
        if dropped:
            print(f"[INFO] allowed_features 指定により {len(dropped)} 個の特徴量を除外: {dropped}")
        if len(feature_list) == 0:
            raise ValueError("allowed_features により使用可能な特徴量が0件になりました")
    print(
        f"[DEBUG] feature_list_len={len(feature_list)} (orig={len(original_feature_list)}) df_feat_rows={len(df_feat)}"
    )

    all_actual, all_pred, all_stage1_rows = [], [], []
    prediction_dates = []  # 実際に予測を行った日付を記録
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index
    print(f"[DEBUG] dates_len={len(dates)} min_stage1_days={min_stage1_days} min_stage2_days={min_stage2_days}")
    if len(dates) <= min_stage1_days + min_stage2_days:
        print("[WARN] 有効日数が閾値合計未満のため十分なステージ2予測が得られない可能性")

    for i, target_date in enumerate(dates):
        if i < min_stage1_days:
            if i % 5 == 0:
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

        print(f"\n=== {target_date.strftime('%Y-%m-%d')} を予測中 (i={i}) ===")
        stage1_result = train_and_predict_stage1(
            df_feat_today,
            df_past_feat,
            df_past_pivot,
            base_models=[
                ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ],
            meta_model_proto=ElasticNet(alpha=0.1, l1_ratio=0.5),
            feature_list=feature_list,
            target_items=target_items,
            stage1_eval=stage1_eval,
            df_pivot=df_pivot,
        )
        row = {f"{item}_予測": stage1_result[f"{item}_予測"] for item in target_items}
        for col in df_feat_today.columns:
            if col not in row:
                row[col] = df_feat_today.iloc[0][col]
        row["合計"] = df_pivot.loc[target_date, "合計"]
        all_stage1_rows.append(row)
        if len(all_stage1_rows) <= min_stage2_days:
            print(f"[DEBUG] ステージ2未実行 rows={len(all_stage1_rows)}/{min_stage2_days+1}")

        if len(all_stage1_rows) > min_stage2_days:
            total_pred = train_and_predict_stage2(all_stage1_rows, stage1_result, df_feat_today, target_items)
            actual_val = df_pivot.loc[target_date, "合計"]
            all_actual.append(actual_val)
            all_pred.append(total_pred)
            prediction_dates.append(target_date)  # 予測日付を記録
            if len(all_actual) >= 3:
                r2_now = r2_score(all_actual, all_pred)
                mae_now = mean_absolute_error(all_actual, all_pred)
                print(f"[PROGRESS] n={len(all_actual)} R²={r2_now:.3f} MAE={mae_now:,.0f}kg")

    print("\n===== ステージ2評価結果 (合計) =====")
    if len(all_actual) > 0:
        print(f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg")
    else:
        print("評価できるデータが不足しています (all_actual=0)")

    evaluate_stage1(stage1_eval, target_items)
    
    # 最後のモデルと日付リストを返す
    last_model = stage1_result if 'stage1_result' in locals() else None
    return all_actual, all_pred, last_model, prediction_dates
