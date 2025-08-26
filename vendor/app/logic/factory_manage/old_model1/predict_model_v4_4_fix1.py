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
            df_pivot[item] = 0  # 品目欠損補完
    df_pivot = df_pivot.sort_index()

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
def train_stage2_model(train_history, target_items):
    df = pd.DataFrame(train_history)
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
    stage1_results = {item: {"y_true": [], "y_pred": []} for item in target_items}

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
            y_full = df_pivot[item]
            y_train = y_full.iloc[:-1]

            if len(y_train) < 2:
                stage1_row[f"{item}_予測"] = 0
                continue

            scaler, selector, models, meta_model = train_stage1_models(
                X_train, y_train, base_models, meta_model_proto
            )
            pred = predict_stage1(X_pred, scaler, selector, models, meta_model)
            stage1_row[f"{item}_予測"] = max(pred, 0)

            true_val = y_full.iloc[-1]
            stage1_results[item]["y_true"].append(true_val)
            stage1_results[item]["y_pred"].append(pred)

        total_actual = df_pivot[target_items].sum(axis=1).iloc[-1]
        stage1_history.append(
            {"日付": target_date, **stage1_row, "合計_実績": total_actual}
        )

        if len(stage1_history) >= 30:
            train_history = stage1_history[:-1]  # 未来リーク排除
            if len(train_history) < 10:
                continue

            model2, scaler2, selector2 = train_stage2_model(train_history, target_items)
            df_input = pd.DataFrame(
                [{f"{item}_予測": stage1_row[f"{item}_予測"] for item in target_items}]
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)

            all_pred.append(total_pred)
            all_actual.append(total_actual)

            print_metrics(all_actual, all_pred)

    if all_actual:
        print_metrics(all_actual, all_pred)

        print("\n===== ステージ1評価結果 =====")
        for item in target_items:
            y_true_s1 = np.array(stage1_results[item]["y_true"])
            y_pred_s1 = np.array(stage1_results[item]["y_pred"])
            if len(y_true_s1) > 0:
                r2_s1 = r2_score(y_true_s1, y_pred_s1)
                mae_s1 = mean_absolute_error(y_true_s1, y_pred_s1)
                print(f"{item}: R² = {r2_s1:.3f}, MAE = {mae_s1:,.0f} kg")
    else:
        print("履歴不足で評価不能")

    return all_actual, all_pred


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
            df_pivot[item] = 0  # 品目欠損補完
    df_pivot = df_pivot.sort_index()

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
def train_stage2_model(train_history, target_items):
    df = pd.DataFrame(train_history)
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
    stage1_results = {item: {"y_true": [], "y_pred": []} for item in target_items}

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
            y_full = df_pivot[item]
            y_train = y_full.iloc[:-1]

            if len(y_train) < 2:
                stage1_row[f"{item}_予測"] = 0
                continue

            scaler, selector, models, meta_model = train_stage1_models(
                X_train, y_train, base_models, meta_model_proto
            )
            pred = predict_stage1(X_pred, scaler, selector, models, meta_model)
            stage1_row[f"{item}_予測"] = max(pred, 0)

            true_val = y_full.iloc[-1]
            stage1_results[item]["y_true"].append(true_val)
            stage1_results[item]["y_pred"].append(pred)

        total_actual = df_pivot[target_items].sum(axis=1).iloc[-1]
        stage1_history.append(
            {"日付": target_date, **stage1_row, "合計_実績": total_actual}
        )

        if len(stage1_history) >= 30:
            train_history = stage1_history[:-1]  # 未来リーク排除
            if len(train_history) < 10:
                continue

            model2, scaler2, selector2 = train_stage2_model(train_history, target_items)
            df_input = pd.DataFrame(
                [{f"{item}_予測": stage1_row[f"{item}_予測"] for item in target_items}]
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)

            all_pred.append(total_pred)
            all_actual.append(total_actual)

            print_metrics(all_actual, all_pred)

    if all_actual:
        print_metrics(all_actual, all_pred)

        print("\n===== ステージ1評価結果 =====")
        for item in target_items:
            y_true_s1 = np.array(stage1_results[item]["y_true"])
            y_pred_s1 = np.array(stage1_results[item]["y_pred"])
            if len(y_true_s1) > 0:
                r2_s1 = r2_score(y_true_s1, y_pred_s1)
                mae_s1 = mean_absolute_error(y_true_s1, y_pred_s1)
                print(f"{item}: R² = {r2_s1:.3f}, MAE = {mae_s1:,.0f} kg")
    else:
        print("履歴不足で評価不能")

    return all_actual, all_pred
