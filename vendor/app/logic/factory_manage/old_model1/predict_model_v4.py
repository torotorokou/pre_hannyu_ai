import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# ---------------------- データ前処理 --------------------------
def generate_features(df_raw, holidays):
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    return df_feat, df_pivot


# =================== 逐次学習（高速化版） ===================
def stage1_walkforward_predict(
    df_feat, df_pivot, ab_features, target_items, base_models, meta_model_stage1
):
    results = []
    errors_stage1 = {item: [] for item in target_items}

    dates = df_feat.index[5:]  # 最初5日は特徴量が欠けるので飛ばす

    for target_date in dates:
        print(f"\U0001f552 Processing {target_date.date()} ...")
        stage1_row = {}

        for item in target_items:
            print(f"  \U0001f4e6 Training for {item}")
            df_past_feat = df_feat[df_feat.index < target_date]
            df_past_pivot = df_pivot.loc[df_past_feat.index]

            # 直近300日だけ使う
            df_past_feat = df_past_feat.tail(300)
            df_past_pivot = df_past_pivot.tail(300)

            X_train_raw = (
                df_past_feat[ab_features]
                if item == "混合廃棄物A"
                else df_past_feat[[c for c in ab_features if "1台あたり" not in c]]
            )
            y_train = df_past_pivot[item]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            selector = VarianceThreshold(threshold=1e-4)
            X_train_filtered = selector.fit_transform(X_train_scaled)

            trained_models = []
            for name, model in base_models:
                model_ = clone(model)
                model_.fit(X_train_filtered, y_train)
                trained_models.append(model_)

            train_meta = np.column_stack(
                [m.predict(X_train_filtered) for m in trained_models]
            )
            meta_model_stage1.fit(train_meta, y_train)

            X_target_raw = (
                df_feat.loc[[target_date], ab_features]
                if item == "混合廃棄物A"
                else df_feat.loc[
                    [target_date], [c for c in ab_features if "1台あたり" not in c]
                ]
            )
            X_target_scaled = scaler.transform(X_target_raw)
            X_target_filtered = selector.transform(X_target_scaled)

            base_preds = np.column_stack(
                [m.predict(X_target_filtered) for m in trained_models]
            )
            pred = meta_model_stage1.predict(base_preds)[0]

            stage1_row[f"{item}_予測"] = pred

            actual = df_pivot.loc[target_date, item]
            errors_stage1[item].append((actual, pred))

        for col in [
            "曜日",
            "週番号",
            "合計_前日",
            "1台あたり正味重量_前日中央値",
            "祝日フラグ",
        ]:
            stage1_row[col] = df_feat.loc[target_date, col]

        stage1_row["合計"] = df_pivot.loc[target_date, "合計"]
        stage1_row["日付"] = target_date
        results.append(stage1_row)

    df_stage2 = pd.DataFrame(results).set_index("日付")
    return df_stage2, errors_stage1


# =================== ステージ1評価 ===================
def evaluate_stage1(errors_stage1):
    print("\n✅ ステージ1 逐次評価:")
    for item, errors in errors_stage1.items():
        actuals, preds = zip(*errors)
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        print(f"{item}: R²={r2:.3f}, MAE={mae:.0f}kg")


# =================== ステージ2学習・評価 ===================
def stage2_train_and_eval(df_stage2):
    feature_cols = [c for c in df_stage2.columns if c not in ["合計"]]
    split_point = int(len(df_stage2) * 0.8)
    X_train_raw, X_val_raw = (
        df_stage2.iloc[:split_point][feature_cols],
        df_stage2.iloc[split_point:][feature_cols],
    )
    y_train, y_val = (
        df_stage2.iloc[:split_point]["合計"],
        df_stage2.iloc[split_point:]["合計"],
    )

    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train_raw)
    X_val_scaled = scaler2.transform(X_val_raw)

    selector2 = VarianceThreshold(threshold=1e-4)
    X_train_filtered = selector2.fit_transform(X_train_scaled)
    X_val_filtered = selector2.transform(X_val_scaled)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered, y_train)

    print("\n✅ ステージ2評価:")
    print(
        f"学習 R²={r2_score(y_train, gbdt.predict(X_train_filtered)):.3f}, MAE={mean_absolute_error(y_train, gbdt.predict(X_train_filtered)):.0f}kg"
    )
    print(
        f"検証 R²={r2_score(y_val, gbdt.predict(X_val_filtered)):.3f}, MAE={mean_absolute_error(y_val, gbdt.predict(X_val_filtered)):.0f}kg"
    )

    return gbdt


# =================== 全体実行パイプライン ===================
def run_fully_walkforward_pipeline(df_feat, df_pivot):
    ab_features = [
        "混合廃棄物A_前日",
        "混合廃棄物B_前日",
        "合計_前日",
        "合計_3日平均",
        "合計_3日合計",
        "曜日",
        "週番号",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]
    target_items = ["混合廃棄物A", "混合廃棄物B"]
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    df_stage2, errors_stage1 = stage1_walkforward_predict(
        df_feat, df_pivot, ab_features, target_items, base_models, meta_model_stage1
    )
    evaluate_stage1(errors_stage1)
    gbdt_model = stage2_train_and_eval(df_stage2)

    return gbdt_model


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# =================== 履歴長スキャンと最適履歴探索 ===================
def find_optimal_history(df_feat, df_pivot, history_windows):
    ab_features = [
        "混合廃棄物A_前日",
        "混合廃棄物B_前日",
        "合計_前日",
        "合計_3日平均",
        "合計_3日合計",
        "曜日",
        "週番号",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]
    target_items = ["混合廃棄物A", "混合廃棄物B"]

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    results_summary = []

    for window in history_windows:
        print(f"\n===== 検証: 最新 {window}日履歴 =====")
        results = []
        dates = df_feat.index[5:]

        for target_date in dates:
            stage1_row = {}
            for item in target_items:
                df_past_feat = df_feat[df_feat.index < target_date].tail(window)
                df_past_pivot = df_pivot.loc[df_past_feat.index]

                X_train_raw = (
                    df_past_feat[ab_features]
                    if item == "混合廃棄物A"
                    else df_past_feat[[c for c in ab_features if "1台あたり" not in c]]
                )
                y_train = df_past_pivot[item]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw)
                selector = VarianceThreshold(threshold=1e-4)
                X_train_filtered = selector.fit_transform(X_train_scaled)

                trained_models = []
                for name, model in base_models:
                    model_ = clone(model)
                    model_.fit(X_train_filtered, y_train)
                    trained_models.append(model_)

                train_meta = np.column_stack(
                    [m.predict(X_train_filtered) for m in trained_models]
                )
                meta_model_stage1.fit(train_meta, y_train)

                X_target_raw = (
                    df_feat.loc[[target_date], ab_features]
                    if item == "混合廃棄物A"
                    else df_feat.loc[
                        [target_date], [c for c in ab_features if "1台あたり" not in c]
                    ]
                )
                X_target_scaled = scaler.transform(X_target_raw)
                X_target_filtered = selector.transform(X_target_scaled)

                base_preds = np.column_stack(
                    [m.predict(X_target_filtered) for m in trained_models]
                )
                pred = meta_model_stage1.predict(base_preds)[0]

                stage1_row[f"{item}_予測"] = pred

            for col in [
                "曜日",
                "週番号",
                "合計_前日",
                "1台あたり正味重量_前日中央値",
                "祝日フラグ",
            ]:
                stage1_row[col] = df_feat.loc[target_date, col]

            stage1_row["合計"] = df_pivot.loc[target_date, "合計"]
            results.append(stage1_row)

        df_stage2 = pd.DataFrame(results, index=dates)
        feature_cols = [c for c in df_stage2.columns if c not in ["合計"]]
        split_point = int(len(df_stage2) * 0.8)
        X_train_raw, X_val_raw = (
            df_stage2.iloc[:split_point][feature_cols],
            df_stage2.iloc[split_point:][feature_cols],
        )
        y_train, y_val = (
            df_stage2.iloc[:split_point]["合計"],
            df_stage2.iloc[split_point:]["合計"],
        )

        scaler2 = StandardScaler()
        X_train_scaled = scaler2.fit_transform(X_train_raw)
        X_val_scaled = scaler2.transform(X_val_raw)
        selector2 = VarianceThreshold(threshold=1e-4)
        X_train_filtered = selector2.fit_transform(X_train_scaled)
        X_val_filtered = selector2.transform(X_val_scaled)

        gbdt = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        )
        gbdt.fit(X_train_filtered, y_train)

        r2_val = r2_score(y_val, gbdt.predict(X_val_filtered))
        mae_val = mean_absolute_error(y_val, gbdt.predict(X_val_filtered))

        print(f"検証 R²={r2_val:.3f}, MAE={mae_val:,.0f}kg")
        results_summary.append({"履歴長": window, "R2": r2_val, "MAE": mae_val})

    df_result = pd.DataFrame(results_summary)
    print("\n===== 履歴長検証結果 =====")
    print(df_result)

    best_row = df_result.sort_values("R2", ascending=False).iloc[0]
    best_window = int(best_row["履歴長"])
    print(f"\n✅ 最適履歴長は {best_window} 日 (R²={best_row['R2']:.3f})")

    return best_window


# =================== 運用用モデル学習 ===================
def train_final_model(df_feat, df_pivot, best_window):
    ab_features = [
        "混合廃棄物A_前日",
        "混合廃棄物B_前日",
        "合計_前日",
        "合計_3日平均",
        "合計_3日合計",
        "曜日",
        "週番号",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]
    target_items = ["混合廃棄物A", "混合廃棄物B"]

    # ステージ2用最終データ作成
    results = []
    dates = df_feat.index[5:]

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    for target_date in dates:
        stage1_row = {}
        for item in target_items:
            df_past_feat = df_feat[df_feat.index < target_date].tail(best_window)
            df_past_pivot = df_pivot.loc[df_past_feat.index]

            X_train_raw = (
                df_past_feat[ab_features]
                if item == "混合廃棄物A"
                else df_past_feat[[c for c in ab_features if "1台あたり" not in c]]
            )
            y_train = df_past_pivot[item]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            selector = VarianceThreshold(threshold=1e-4)
            X_train_filtered = selector.fit_transform(X_train_scaled)

            trained_models = []
            for name, model in base_models:
                model_ = clone(model)
                model_.fit(X_train_filtered, y_train)
                trained_models.append(model_)

            train_meta = np.column_stack(
                [m.predict(X_train_filtered) for m in trained_models]
            )
            meta_model_stage1.fit(train_meta, y_train)

            X_target_raw = (
                df_feat.loc[[target_date], ab_features]
                if item == "混合廃棄物A"
                else df_feat.loc[
                    [target_date], [c for c in ab_features if "1台あたり" not in c]
                ]
            )
            X_target_scaled = scaler.transform(X_target_raw)
            X_target_filtered = selector.transform(X_target_scaled)

            base_preds = np.column_stack(
                [m.predict(X_target_filtered) for m in trained_models]
            )
            pred = meta_model_stage1.predict(base_preds)[0]

            stage1_row[f"{item}_予測"] = pred

        for col in [
            "曜日",
            "週番号",
            "合計_前日",
            "1台あたり正味重量_前日中央値",
            "祝日フラグ",
        ]:
            stage1_row[col] = df_feat.loc[target_date, col]

        stage1_row["合計"] = df_pivot.loc[target_date, "合計"]
        results.append(stage1_row)

    df_stage2 = pd.DataFrame(results, index=dates)
    feature_cols = [c for c in df_stage2.columns if c not in ["合計"]]
    X_train_raw = df_stage2[feature_cols]
    y_train = df_stage2["合計"]

    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train_raw)
    selector2 = VarianceThreshold(threshold=1e-4)
    X_train_filtered = selector2.fit_transform(X_train_scaled)

    gbdt_final = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt_final.fit(X_train_filtered, y_train)

    print("\n✅ 運用用モデル学習 完了")

    return gbdt_final, scaler2, selector2
