import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# ---------- 特徴量リスト作成 ----------
def get_features_all(target_items):
    features = [f"{item}_前日値" for item in target_items]
    features += ["合計_前日値", "合計_3日平均", "合計_3日合計", "曜日", "週番号"]
    return features


# ---------- 重量特徴量作成 ----------
def generate_weight_features(past_raw, target_items):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    for item in target_items:
        if item not in df_pivot.columns:
            df_pivot[item] = 0
    df_pivot["全品目合計"] = df_pivot.sum(axis=1)
    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
    target_sum = df_pivot[target_items].sum(axis=1)
    df_feat["合計_前日値"] = target_sum.shift(1)
    df_feat["合計_3日平均"] = target_sum.shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = target_sum.shift(1).rolling(3).sum()
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]
    return df_feat, df_pivot


# ---------- カレンダー特徴量 ----------
def generate_calendar_features(target_date):
    return {"曜日": target_date.weekday(), "週番号": target_date.isocalendar().week}


# ---------- ステージ1 ----------
def train_stage1_models(X_train, y_train, base_models, meta_model_proto):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(X_scaled)
    trained_models = [clone(model).fit(X_filtered, y_train) for _, model in base_models]
    meta_model = clone(meta_model_proto)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    meta_model.fit(meta_features, y_train)
    return scaler, selector, trained_models, meta_model


def predict_stage1(X_pred, scaler, selector, trained_models, meta_model):
    X_scaled = scaler.transform(X_pred)
    X_filtered = selector.transform(X_scaled)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    return meta_model.predict(meta_features)[0]


# ---------- ステージ2 ----------
def train_stage2_model(stage1_history, target_items):
    df = pd.DataFrame(stage1_history)
    X = df[[f"{item}_予測" for item in target_items]]
    y = df["合計_実績"]
    scaler = StandardScaler()
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(scaler.fit_transform(X))
    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_filtered, y)
    return model, scaler, selector


def predict_stage2(model, scaler, selector, input_row):
    X_scaled = scaler.transform(input_row)
    X_filtered = selector.transform(X_scaled)
    return model.predict(X_filtered)[0]


# ---------- 評価指標 ----------
def print_metrics(all_actual, all_pred):
    y_true = np.array(all_actual)
    y_pred = np.array(all_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    max_err = np.max(np.abs(y_true - y_pred))
    print("\n===== 最終評価結果 =====")
    print(f"R² = {r2:.3f}")
    print(f"MAE = {mae:,.0f} kg")
    print(f"RMSE = {rmse:,.0f} kg")
    print(f"MAPE = {mape:.2f} %")
    print(f"最大誤差 = {max_err:,.0f} kg")


def get_target_items(df):
    top_items = df["品名"].value_counts().head(5).index.tolist()
    return top_items


# ---------- ウォークフォワード全体 ----------
def full_walkforward_pipeline(df_raw, start_index=30):
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))

    target_items = get_target_items(df_raw)
    features_all = get_features_all(target_items)
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

    stage1_history, all_actual, all_pred = [], [], []

    for target_date in all_dates[start_index:]:
        past_raw = df_raw[df_raw["伝票日付"] < target_date]
        df_feat, df_pivot = generate_weight_features(past_raw, target_items)
        calendar_features = generate_calendar_features(target_date)
        for key, val in calendar_features.items():
            df_feat[key] = val

        X_full = df_feat[features_all]
        if len(X_full) < 2:
            continue
        X_train = X_full.iloc[:-1]
        X_pred = X_full.iloc[[-1]]

        stage1_row = {}
        for item in target_items:
            y_full = (
                df_pivot[item]
                if item in df_pivot.columns
                else pd.Series(0, index=X_full.index)
            )
            y_train = y_full.iloc[:-1]

            if len(y_train) < 2:
                stage1_row[f"{item}_予測"] = 0
                continue

            scaler, selector, models, meta_model = train_stage1_models(
                X_train, y_train, base_models, meta_model_proto
            )
            pred = predict_stage1(X_pred, scaler, selector, models, meta_model)
            stage1_row[f"{item}_予測"] = max(pred, 0)

        total_actual = df_pivot[target_items].sum(axis=1).iloc[-1]
        stage1_history.append(
            {"日付": target_date, **stage1_row, "合計_実績": total_actual}
        )

        if len(stage1_history) >= 30:
            # ★ 未来リーク完全排除：当日分の実績は学習に入れない
            train_history = stage1_history[:-1]
            if len(train_history) < 10:
                continue  # 学習データ不足はスキップ

            df_train = pd.DataFrame(train_history)
            X_stage2 = df_train[[f"{item}_予測" for item in target_items]]
            y_stage2 = df_train["合計_実績"]
            scaler2 = StandardScaler()
            selector2 = VarianceThreshold(1e-4)
            X_stage2_filtered = selector2.fit_transform(scaler2.fit_transform(X_stage2))
            model2 = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
            )
            model2.fit(X_stage2_filtered, y_stage2)

            df_input = pd.DataFrame(
                [{f"{item}_予測": stage1_row[f"{item}_予測"] for item in target_items}]
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)
            all_pred.append(total_pred)
            all_actual.append(total_actual)

    return all_actual, all_pred


# ---------- クロスバリデーション導入 ----------
def cross_validation_pipeline(df_raw, n_splits=3):
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))
    total_r2, total_mae, total_rmse, total_mape = [], [], [], []

    split_points = np.linspace(30, len(all_dates) - 30, n_splits, dtype=int)

    for split_start in split_points:
        print(f"\n===== クロスバリデーション分割: {split_start}日目以降 =====")
        all_actual, all_pred = full_walkforward_pipeline(
            df_raw, start_index=split_start
        )

        if len(all_actual) == 0:
            print("  -> 評価データ不足")
            continue

        y_true = np.array(all_actual)
        y_pred = np.array(all_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        )
        total_r2.append(r2)
        total_mae.append(mae)
        total_rmse.append(rmse)
        total_mape.append(mape)

        print(
            f"  R² = {r2:.3f}, MAE = {mae:,.0f} kg, RMSE = {rmse:,.0f} kg, MAPE = {mape:.2f}%"
        )

    if total_r2:
        print("\n===== クロスバリデーション全体結果 =====")
        print(f"平均 R² = {np.mean(total_r2):.3f}")
        print(f"平均 MAE = {np.mean(total_mae):,.0f} kg")
        print(f"平均 RMSE = {np.mean(total_rmse):,.0f} kg")
        print(f"平均 MAPE = {np.mean(total_mape):.2f}%")
    else:
        print("十分なデータがなくCV評価不能")


def prepare_history_data(df_raw, history_days):
    """
    df_raw から最新の履歴日数だけ抽出
    """
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))

    if len(all_dates) < history_days:
        print(f"履歴{history_days}日分のデータがそもそも不足")
        return None

    start_date = all_dates[-history_days]
    df_cut = df_raw[df_raw["伝票日付"] >= start_date]
    return df_cut
