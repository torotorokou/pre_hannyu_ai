import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# ---------- 得意先トップリスト抽出 ----------
def get_top_customers(df_reserve, target_date, window_days=90, top_n=20):
    start_date = target_date - pd.Timedelta(days=window_days)
    recent_reserve = df_reserve[
        (df_reserve["予約日"] >= start_date) & (df_reserve["予約日"] < target_date)
    ]
    customer_counts = recent_reserve["得意先名"].value_counts()
    top_customers = customer_counts.head(top_n).index.tolist()
    return top_customers


# ---------- 重量履歴から特徴量を作成する関数 ----------
def generate_weight_features(past_raw):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合A_前日値"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合B_前日値"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = daily_avg.shift(1).expanding().median()

    df_feat = df_feat.dropna()
    return df_feat, df_pivot.loc[df_feat.index]


# ---------- 予約情報から特徴量を作成する関数 ----------
def generate_reserve_features(df_reserve, target_date, top_customers):
    reserve_today = df_reserve[df_reserve["予約日"] == target_date]
    features = {
        "予約件数": len(reserve_today),
        "固定客予約件数": reserve_today["固定客"].sum(),
    }
    for customer in top_customers:
        features[f"予約_{customer}"] = int(customer in reserve_today["得意先名"].values)
    return features


# ---------- 1日分の全特徴量統合 ----------
def generate_features_for_date(
    df_raw, holidays, target_date, df_reserve=None, top_customers=[]
):
    past_raw = df_raw[df_raw["伝票日付"] < target_date]
    if len(past_raw) < 3:
        return None, None

    df_feat, df_pivot = generate_weight_features(past_raw)

    reserve_features = {"予約件数": 0, "固定客予約件数": 0}
    for customer in top_customers:
        reserve_features[f"予約_{customer}"] = 0
    if df_reserve is not None:
        reserve_features.update(
            generate_reserve_features(df_reserve, target_date, top_customers)
        )
    for key, val in reserve_features.items():
        df_feat[key] = val

    df_feat["曜日"] = target_date.weekday()
    df_feat["週番号"] = target_date.isocalendar().week
    df_feat["祝日フラグ"] = int(target_date in holidays)

    return df_feat, df_pivot


# ---------- ステージ1 (品目ごとの個別モデル) ----------
def train_stage1_models(X, y, base_models, meta_model_proto):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(X_scaled)

    trained_models = [clone(model).fit(X_filtered, y) for _, model in base_models]

    meta_model = clone(meta_model_proto)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    meta_model.fit(meta_features, y)

    return scaler, selector, trained_models, meta_model


# ---------- ステージ1 予測処理 ----------
def predict_stage1(X, scaler, selector, trained_models, meta_model):
    X_scaled = scaler.transform(X)
    X_filtered = selector.transform(X_scaled)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    return meta_model.predict(meta_features)[0]


# ---------- ステージ2 (合計補正用メタモデル) ----------
def train_stage2_model(stage1_history):
    df = pd.DataFrame(stage1_history).dropna()
    feature_cols = [c for c in df.columns if c not in ["日付", "合計_実績"]]
    X = df[feature_cols]
    y = df["合計_実績"]

    scaler = StandardScaler()
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(scaler.fit_transform(X))

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_filtered, y)
    return model, scaler, selector, feature_cols


# ---------- ステージ2 予測処理 ----------
def predict_stage2(model, scaler, selector, input_row):
    X_scaled = scaler.transform(input_row)
    X_filtered = selector.transform(X_scaled)
    return model.predict(X_filtered)[0]


# ---------- 特徴量重要度確認用（ステージ2用） ----------
def print_stage2_importance(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n===== ステージ2特徴量重要度 (GBDT) =====")
    for idx in indices:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")


# ---------- ステージ1のElasticNet係数確認用 ----------
def print_stage1_coefficients(meta_model, feature_names):
    print("\n===== ステージ1メタモデル係数 (ElasticNet) =====")
    for name, coef in zip(feature_names, meta_model.coef_):
        print(f"{name}: {coef:.4f}")


# ---------- 完全ウォークフォワード型の検証パイプライン ----------
def full_walkforward_pipeline(df_raw, holidays, df_reserve=None, top_customers=[]):
    holidays = pd.to_datetime(holidays)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))

    base_features = [
        "混合A_前日値",
        "混合B_前日値",
        "合計_前日値",
        "合計_3日平均",
        "合計_3日合計",
        "1台あたり重量_過去中央値",
        "予約件数",
        "固定客予約件数",
        "曜日",
        "週番号",
        "祝日フラグ",
    ]
    customer_features = [f"予約_{c}" for c in top_customers]
    features_all = base_features + customer_features

    target_items = ["混合廃棄物A", "混合廃棄物B"]
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

    stage1_history, all_actual, all_pred = [], [], []

    last_elastic_model = None
    last_selected_columns = None

    for target_date in all_dates[5:]:
        print(f"\n===== {target_date.date()} 処理中 =====")
        df_feat, df_pivot = generate_features_for_date(
            df_raw, holidays, target_date, df_reserve, top_customers
        )
        if df_feat is None:
            print("  履歴不足でスキップ")
            continue

        stage1_row = {}
        for item in target_items:
            use_features = [
                f
                for f in features_all
                if not (item != "混合廃棄物A" and "1台あたり" in f)
            ]
            X = df_feat[use_features]
            y = df_pivot[item]

            scaler, selector, models, meta_model = train_stage1_models(
                X, y, base_models, meta_model_proto
            )
            pred = predict_stage1(
                X.iloc[[-1]],
                scaler,
                selector,
                trained_models=models,
                meta_model=meta_model,
            )
            stage1_row[f"{item}_予測"] = pred
            print(f"    {item} 予測値: {pred:.1f}kg")

            print_stage1_coefficients(meta_model, [name for name, _ in base_models])

            # ステージ1のElasticNet弱学習器の係数保存
            elastic_model = models[0]
            selected_columns = X.columns[selector.get_support()]
            last_elastic_model = elastic_model
            last_selected_columns = selected_columns

        if len(stage1_history) >= 30:
            df_input = pd.DataFrame(
                [{**stage1_row, **df_feat.iloc[-1][features_all].to_dict()}]
            )
            model2, scaler2, selector2, feature_cols = train_stage2_model(
                stage1_history
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)
            total_actual = df_pivot["合計"].iloc[-1]
            all_pred.append(total_pred)
            all_actual.append(total_actual)
            print(
                f"  【ステージ2】 合計予測: {total_pred:.0f}kg, 実績: {total_actual:.0f}kg"
            )

        stage1_history.append(
            {
                "日付": target_date,
                **stage1_row,
                **df_feat.iloc[-1][features_all].to_dict(),
                "合計_実績": df_pivot["合計"].iloc[-1],
            }
        )

    if all_actual:
        print("\n===== 最終評価結果 =====")
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
        print_stage2_importance(model2, feature_cols)

        # ステージ1特徴量の最終係数出力
        print("\n===== ステージ1最終特徴量重要度 (ElasticNet弱学習器) =====")
        for name, coef in zip(last_selected_columns, last_elastic_model.coef_):
            print(f"{name}: {coef:.4f}")
    else:
        print("履歴不足で評価不能")
