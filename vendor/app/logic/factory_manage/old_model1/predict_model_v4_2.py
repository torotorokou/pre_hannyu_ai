import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


# ---------------------- 完全ウォークフォワード実行 --------------------------
def full_walkforward(df_feat, df_pivot):
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

    total_dates = df_feat.index[5:]
    all_actual, all_pred = [], []

    for i, target_date in enumerate(total_dates):
        print(f"\n===== {target_date.date()} の処理開始 =====")
        stage1_results = {}
        for item in target_items:
            print(f"  - ステージ1: {item} モデル学習")
            df_past_feat = df_feat[df_feat.index < target_date].tail(300)
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
                print(f"    - {name} モデル学習完了")

            train_meta = np.column_stack(
                [m.predict(X_train_filtered) for m in trained_models]
            )
            meta_model_stage1.fit(train_meta, y_train)
            print("    - メタモデル学習完了")

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
            stage1_results[f"{item}_予測"] = pred
            print(f"    - {item} 予測値: {pred:.0f}kg")

        # ステージ2 (逐次学習)
        if i >= 10:
            print("  - ステージ2: GBDT再学習")
            df_stage2_hist = pd.DataFrame(all_stage1_rows)
            X_train_s2 = df_stage2_hist.drop(columns=["合計"])
            y_train_s2 = df_stage2_hist["合計"]

            scaler2 = StandardScaler()
            X_train_scaled2 = scaler2.fit_transform(X_train_s2)
            selector2 = VarianceThreshold(threshold=1e-4)
            X_train_filtered2 = selector2.fit_transform(X_train_scaled2)

            gbdt = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
            )
            gbdt.fit(X_train_filtered2, y_train_s2)

            X_target_s2 = pd.DataFrame(
                {
                    f"{target_items[0]}_予測": [
                        stage1_results[f"{target_items[0]}_予測"]
                    ],
                    f"{target_items[1]}_予測": [
                        stage1_results[f"{target_items[1]}_予測"]
                    ],
                    "曜日": df_feat.loc[target_date, "曜日"],
                    "週番号": df_feat.loc[target_date, "週番号"],
                    "合計_前日": df_feat.loc[target_date, "合計_前日"],
                    "1台あたり正味重量_前日中央値": df_feat.loc[
                        target_date, "1台あたり正味重量_前日中央値"
                    ],
                    "祝日フラグ": df_feat.loc[target_date, "祝日フラグ"],
                }
            )
            X_target_scaled2 = scaler2.transform(X_target_s2)
            X_target_filtered2 = selector2.transform(X_target_scaled2)
            total_pred = gbdt.predict(X_target_filtered2)[0]

            total_actual = df_pivot.loc[target_date, "合計"]
            print(
                f"    => ステージ2 予測: {total_pred:.0f}kg / 実績: {total_actual:.0f}kg"
            )
            all_actual.append(total_actual)
            all_pred.append(total_pred)

        # ステージ2用履歴に蓄積
        stage1_row_for_s2 = {
            f"{target_items[0]}_予測": stage1_results[f"{target_items[0]}_予測"],
            f"{target_items[1]}_予測": stage1_results[f"{target_items[1]}_予測"],
            "曜日": df_feat.loc[target_date, "曜日"],
            "週番号": df_feat.loc[target_date, "週番号"],
            "合計_前日": df_feat.loc[target_date, "合計_前日"],
            "1台あたり正味重量_前日中央値": df_feat.loc[
                target_date, "1台あたり正味重量_前日中央値"
            ],
            "祝日フラグ": df_feat.loc[target_date, "祝日フラグ"],
            "合計": df_pivot.loc[target_date, "合計"],
        }
        if i == 0:
            all_stage1_rows = [stage1_row_for_s2]
        else:
            all_stage1_rows.append(stage1_row_for_s2)

    # 最終評価
    print("\n===== 完全ウォークフォワード結果 =====")
    print(
        f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
    )
    return all_actual, all_pred
