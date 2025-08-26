import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# =================== ウォークフォワード1日先評価 ===================
def walkforward_one_day_evaluation(df_feat, df_pivot, history_window):
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

    # 検証期間は最後20%
    total_dates = df_feat.index[5:]  # 先頭5日は特徴量不足で除外
    split_point = int(len(total_dates) * 0.8)
    test_dates = total_dates[split_point:]

    all_actual = []
    all_pred = []

    print("\n===== 各日の予測結果 =====")
    for target_date in test_dates:
        stage1_row = {}
        for item in target_items:
            # 学習履歴取得（直前history_window日分）
            df_past_feat = df_feat[df_feat.index < target_date].tail(history_window)
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

            # 翌日予測
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

        # 合計モデル予測（簡易版: A+B合算）
        total_pred = stage1_row["混合廃棄物A_予測"] + stage1_row["混合廃棄物B_予測"]
        total_actual = df_pivot.loc[target_date, "合計"]
        all_actual.append(total_actual)
        all_pred.append(total_pred)

        # ここで日付ごとに出力
        # print(
        #     f"{target_date.date()}  実績: {total_actual:,.0f}kg  予測: {total_pred:,.0f}kg  誤差: {total_pred - total_actual:+,.0f}kg"
        # )

    r2 = r2_score(all_actual, all_pred)
    mae = mean_absolute_error(all_actual, all_pred)

    print("\n===== ウォークフォワード1日先評価 結果 =====")
    print(f"R² = {r2:.3f}, MAE = {mae:,.0f}kg")
    return r2, mae
