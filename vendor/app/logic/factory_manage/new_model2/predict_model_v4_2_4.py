import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# 特徴量作成用ビルダーの読み込み
from logic.factory_manage.new_model2.feature_builder import (
    WeightFeatureBuilder,
    ReserveFeatureBuilder,
    WeatherFeatureBuilder,
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
        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

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
    df_raw, holidays, df_reserve, df_weather, min_stage1_days, min_stage2_days, top_n=5
):
    print("▶️ full_walkforward 開始")
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)

    df_feat, df_pivot = WeightFeatureBuilder(df_raw, target_items, holidays).build()
    df_reserve_feat_all = ReserveFeatureBuilder(df_reserve).build()
    df_weather_feat_all = WeatherFeatureBuilder(df_weather).build()

    feature_list = get_feature_list(
        target_items, extra_features=["天気_晴れ", "天気_雨", "天気_大雨", "天気_台風"]
    )

    all_actual, all_pred, all_stage1_rows = [], [], []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index

    for i, target_date in enumerate(dates):
        if i < min_stage1_days:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(600)
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_today = df_reserve_feat_all[df_reserve_feat_all.index <= target_date]
        df_weather_today = df_weather_feat_all[df_weather_feat_all.index <= target_date]

        df_past_feat = df_past_feat.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
        )
        df_past_feat = df_past_feat.merge(
            df_weather_today, left_index=True, right_index=True, how="left"
        ).fillna(0)

        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
        )
        df_feat_today = df_feat_today.merge(
            df_weather_today, left_index=True, right_index=True, how="left"
        ).fillna(0)

        print(f"\n=== {target_date.strftime('%Y-%m-%d')} を予測中 ===")
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

        if len(all_stage1_rows) > min_stage2_days:
            total_pred = train_and_predict_stage2(
                all_stage1_rows, stage1_result, df_feat_today, target_items
            )
            all_actual.append(df_pivot.loc[target_date, "合計"])
            all_pred.append(total_pred)

    print("\n===== ステージ2評価結果 (合計) =====")
    if all_actual:
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
    else:
        print("評価できるデータが不足しています")

    evaluate_stage1(stage1_eval, target_items)
    return all_actual, all_pred
