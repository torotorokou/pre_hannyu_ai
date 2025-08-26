import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone


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


# ---------------------- モデル学習と予測 --------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error


def create_stage1_features(
    df_feat, df_pivot, target_items, ab_features, base_models, meta_model_stage1
):
    tscv = TimeSeriesSplit(n_splits=5)
    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]
        X_features_all[item] = (X, y)
        train_meta = np.zeros((X.shape[0], len(base_models)))

        for i, (_, model) in enumerate(base_models):
            for train_idx, val_idx in tscv.split(X):
                model_ = clone(model)
                model_.fit(X.iloc[train_idx], y.iloc[train_idx])
                train_meta[val_idx, i] = model_.predict(X.iloc[val_idx])

        meta_model_stage1.fit(train_meta, y)
        test_meta = np.column_stack(
            [clone(model).fit(X, y).predict(X) for _, model in base_models]
        )
        stacked_preds[item] = meta_model_stage1.predict(test_meta)

    index_final = df_feat.index
    df_stage1 = pd.DataFrame(
        {f"{k}_予測": v for k, v in stacked_preds.items()}, index=index_final
    )
    for col in [
        "曜日",
        "週番号",
        "合計_前日",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]:
        df_stage1[col] = df_feat[col]

    return df_stage1, X_features_all


def evaluate_stage2_model(X_train, y_train, X_val, y_val, gbdt_model):
    gbdt_model.fit(X_train, y_train)

    y_pred_train = gbdt_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    print(f"✅ ステージ2学習データ R² = {r2_train:.3f}, MAE = {mae_train:,.0f} kg")

    y_pred_val = gbdt_model.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    print(f"✅ ステージ2検証データ R² = {r2_val:.3f}, MAE = {mae_val:,.0f} kg")

    residuals = y_train - y_pred_train
    bias = residuals.mean()
    std = residuals.std()

    return gbdt_model, bias, std


def predict_future(
    df_hist,
    df_hist_feat,
    base_models,
    meta_model_stage1,
    gbdt_model,
    X_features_all,
    ab_features,
    target_items,
    start_date,
    end_date,
    bias,
    std,
    holidays,
):
    holiday_dates = pd.to_datetime(holidays)
    last_date = df_hist_feat.index[-1]
    predict_dates = pd.date_range(start=start_date, end=end_date)
    results = []

    for predict_date in predict_dates:
        new_row = {
            "混合廃棄物A_前日": df_hist.loc[last_date, "混合廃棄物A"],
            "混合廃棄物B_前日": df_hist.loc[last_date, "混合廃棄物B"],
            "合計_前日": df_hist.loc[last_date, "合計"],
            "合計_3日平均": df_hist["合計"].shift(1).rolling(3).mean().loc[last_date],
            "合計_3日合計": df_hist["合計"].shift(1).rolling(3).sum().loc[last_date],
            "曜日": predict_date.dayofweek,
            "週番号": predict_date.isocalendar().week,
            "1台あたり正味重量_前日中央値": df_hist_feat[
                "1台あたり正味重量_前日中央値"
            ].iloc[-1],
            "祝日フラグ": int(predict_date in holiday_dates),
        }
        df_input = pd.DataFrame(new_row, index=[predict_date])

        for item in target_items:
            x_item = (
                df_input[ab_features]
                if item == "混合廃棄物A"
                else df_input[[c for c in ab_features if "1台あたり" not in c]]
            )
            X_train_item, y_train_item = X_features_all[item]
            meta_input = np.column_stack(
                [
                    clone(model).fit(X_train_item, y_train_item).predict(x_item)
                    for _, model in base_models
                ]
            )
            df_input[f"{item}_予測"] = meta_model_stage1.predict(meta_input)[0]

        stage2_input = df_input[
            [
                f"{target_items[0]}_予測",
                f"{target_items[1]}_予測",
                f"{target_items[2]}_予測",
                "曜日",
                "週番号",
                "合計_前日",
                "1台あたり正味重量_前日中央値",
                "祝日フラグ",
            ]
        ]

        y_pred = gbdt_model.predict(stage2_input)[0]
        y_adjusted = y_pred + bias
        lower = y_adjusted - 1.96 * std
        upper = y_adjusted + 1.96 * std

        results.append(
            {
                "日付": predict_date.strftime("%Y-%m-%d"),
                "予測値": y_pred,
                "補正後予測": y_adjusted,
                "下限95CI": lower,
                "上限95CI": upper,
            }
        )

        new_total = y_adjusted
        new_pivot_row = {
            "混合廃棄物A": df_input[f"{target_items[0]}_予測"].iloc[0],
            "混合廃棄物B": df_input[f"{target_items[1]}_予測"].iloc[0],
            "混合廃棄物(ｿﾌｧｰ･家具類)": df_input[f"{target_items[2]}_予測"].iloc[0],
            "合計": new_total,
        }
        df_hist.loc[predict_date] = new_pivot_row
        last_date = predict_date

    df_result = pd.DataFrame(results).set_index("日付")
    return df_result


def train_and_predict(
    df_feat, df_pivot, start_date, end_date, holidays, future_actual_df=None
):
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
    target_items = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(ｿﾌｧｰ･家具類)"]

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5)
    gbdt_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )

    df_stage1, X_features_all = create_stage1_features(
        df_feat, df_pivot, target_items, ab_features, base_models, meta_model_stage1
    )
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]

    split_point = int(len(df_stage1) * 0.8)
    X_train = df_stage1.iloc[:split_point]
    y_train = y_total_final.iloc[:split_point]
    X_val = df_stage1.iloc[split_point:]
    y_val = y_total_final.iloc[split_point:]

    gbdt_model, bias, std = evaluate_stage2_model(
        X_train, y_train, X_val, y_val, gbdt_model
    )

    df_result = predict_future(
        df_pivot.copy(),
        df_feat.copy(),
        base_models,
        meta_model_stage1,
        gbdt_model,
        X_features_all,
        ab_features,
        target_items,
        start_date,
        end_date,
        bias,
        std,
        holidays,
    )

    if future_actual_df is not None:
        df_result = df_result.join(future_actual_df.rename("実測値"))
        r2_future = r2_score(df_result["実測値"], df_result["補正後予測"])
        mae_future = mean_absolute_error(df_result["実測値"], df_result["補正後予測"])
        print(f"✅ 将来予測 R² = {r2_future:.3f}, MAE = {mae_future:,.0f} kg")

    return df_result
