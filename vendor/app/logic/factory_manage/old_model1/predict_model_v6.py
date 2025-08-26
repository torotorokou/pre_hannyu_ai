import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone
import pandas as pd
import unicodedata
import re


# ---------------------- データ前処理（A案：特徴量強化版） --------------------------
def generate_features(df_raw, holidays):
    # --- 品名正規化 ---
    df_raw["品名"] = df_raw["品名"].astype(str).apply(normalize_item_name)

    # --- ピボット生成 ---
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    # --- 特徴量生成 ---
    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot.get(
        "混合廃棄物A", pd.Series(0, index=df_pivot.index)
    ).shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot.get(
        "混合廃棄物B", pd.Series(0, index=df_pivot.index)
    ).shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()

    # 3日傾き
    def calc_slope(series):
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        return reg.coef_[0, 0]

    df_feat["合計_3日傾き"] = (
        df_pivot["合計"].shift(1).rolling(3).apply(calc_slope, raw=False)
    )

    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat["祝日前フラグ"] = (
        (df_feat["祝日フラグ"] == 0) & (df_feat.shift(-1)["祝日フラグ"] == 1)
    ).astype(int)
    df_feat["月初フラグ"] = (df_feat.index.day <= 5).astype(int)
    df_feat["月末フラグ"] = (
        df_feat.index.day >= (df_feat.index.days_in_month - 5)
    ).astype(int)

    # 欠損除去
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    return df_feat, df_pivot


# ---------------------- 汎用特徴量定義 --------------------------


# 品目ごとの特徴量選択ルール
def get_features_for_item(item, all_features):
    if item == "混合廃棄物A":
        return all_features
    else:
        return [f for f in all_features if "1台あたり" not in f]


# =================== ウォークフォワード 汎用完全版 ===================
def walkforward_v3(df_feat, df_pivot, history_window, target_items):
    # ベース特徴量定義
    base_features = [
        "混合廃棄物A_前日",
        "混合廃棄物B_前日",
        "合計_前日",
        "合計_3日平均",
        "合計_3日合計",
        "合計_3日傾き",
        "曜日",
        "週番号",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
        "祝日前フラグ",
        "月初フラグ",
        "月末フラグ",
    ]

    # ステージ1モデル定義
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    total_dates = df_feat.index[5:]
    split_point = int(len(total_dates) * 0.8)
    test_dates = total_dates[split_point:]

    all_actual = []
    all_pred = []

    print("\n===== 各日の予測結果 =====")

    for target_date in test_dates:
        stage1_row = {}
        for item in target_items:
            df_past_feat = df_feat[df_feat.index < target_date].tail(history_window)
            df_past_pivot = df_pivot.loc[df_past_feat.index]

            # 品目別特徴量選択
            features_for_this_item = get_features_for_item(item, base_features)
            X_train_raw = df_past_feat[features_for_this_item]
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

            X_target_raw = df_feat.loc[[target_date], features_for_this_item]
            X_target_scaled = scaler.transform(X_target_raw)
            X_target_filtered = selector.transform(X_target_scaled)

            base_preds = np.column_stack(
                [m.predict(X_target_filtered) for m in trained_models]
            )
            pred = meta_model_stage1.predict(base_preds)[0]

            stage1_row[f"{item}_予測"] = pred

        # ステージ2 (GBDT) 用特徴量作成
        # すべての品目のステージ1予測値を特徴量化
        stage1_pred_cols = {
            f"{item}_予測": [stage1_row[f"{item}_予測"]] for item in target_items
        }
        stage2_other_feats = {
            col: df_feat.loc[target_date, col]
            for col in base_features
            if col not in ["混合廃棄物A_前日", "混合廃棄物B_前日"]
        }
        df_stage2_row = pd.DataFrame(
            {**stage1_pred_cols, **stage2_other_feats}, index=[target_date]
        )

        # ステージ2学習データ作成
        history_rows = []
        for history_date in (
            df_feat[df_feat.index < target_date].tail(history_window).index
        ):
            hist_row = {
                **{
                    f"{item}_予測": df_pivot.loc[history_date, item]
                    for item in target_items
                },
                **{
                    col: df_feat.loc[history_date, col]
                    for col in base_features
                    if col not in ["混合廃棄物A_前日", "混合廃棄物B_前日"]
                },
                "合計": df_pivot.loc[history_date, "合計"],
            }
            history_rows.append(hist_row)

        df_stage2_hist = pd.DataFrame(history_rows)
        feature_cols = [c for c in df_stage2_hist.columns if c != "合計"]

        scaler2 = StandardScaler()
        X_train_scaled = scaler2.fit_transform(df_stage2_hist[feature_cols])
        selector2 = VarianceThreshold(threshold=1e-4)
        X_train_filtered = selector2.fit_transform(X_train_scaled)

        gbdt = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        )
        gbdt.fit(X_train_filtered, df_stage2_hist["合計"])

        # ステージ2予測
        X_target_scaled2 = scaler2.transform(df_stage2_row[feature_cols])
        X_target_filtered2 = selector2.transform(X_target_scaled2)
        total_pred = gbdt.predict(X_target_filtered2)[0]

        total_actual = df_pivot.loc[target_date, "合計"]
        all_actual.append(total_actual)
        all_pred.append(total_pred)

    r2 = r2_score(all_actual, all_pred)
    mae = mean_absolute_error(all_actual, all_pred)

    print("\n===== 完全ウォークフォワード1日先評価（完全汎用版）結果 =====")
    print(f"R² = {r2:.3f}, MAE = {mae:,.0f}kg")
    return r2, mae


def yobidasi(df_feat, df_pivot, history_window):
    target_items = [
        "混合廃棄物A",
        "混合廃棄物B",
        "GC軽鉄・スチール類",
        "選別",
        "木くず",
    ]
    r2, mae = walkforward_v3(
        df_feat, df_pivot, history_window, target_items=target_items
    )

    return r2, mae
