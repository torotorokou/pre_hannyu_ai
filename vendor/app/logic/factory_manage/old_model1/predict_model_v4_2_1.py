import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# ---------- 上位品目抽出 ----------
def get_target_items(df_raw, top_n=5):
    top_items = df_raw["品名"].value_counts().head(top_n).index.tolist()
    return top_items


def generate_reserve_features(df_reserve, top_k_clients=10):
    """
    日別予約データから特徴量を作成する
    - df_reserve: 「予約日」「予約得意先名」「固定客」を含むDataFrame
    - top_k_clients: 上位得意先を何件抽出するか（デフォルト10）

    Returns: 日付をindexとする特徴量DataFrame
    """
    df_reserve = df_reserve.copy()
    df_reserve["予約日"] = pd.to_datetime(df_reserve["予約日"])

    # 上位得意先を抽出（頻出順）
    top_clients = df_reserve["予約得意先名"].value_counts().head(top_k_clients).index
    df_reserve["上位得意先フラグ"] = (
        df_reserve["予約得意先名"].isin(top_clients).astype(int)
    )

    # 日別に集計
    df_reserve_feat = df_reserve.groupby("予約日").agg(
        予約件数=("予約得意先名", "count"),
        固定客予約数=("固定客", lambda x: x.sum()),
        非固定客予約数=("固定客", lambda x: (~x).sum()),
        上位得意先予約数=("上位得意先フラグ", "sum"),
    )
    df_reserve_feat["固定客比率"] = (
        df_reserve_feat["固定客予約数"] / df_reserve_feat["予約件数"]
    )
    df_reserve_feat = df_reserve_feat.fillna(0)

    return df_reserve_feat


# ---------- 重量履歴＋特徴量を作成 ----------
def generate_weight_features(past_raw, target_items, holidays):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    for item in target_items:
        if item not in df_pivot.columns:
            df_pivot[item] = 0
    df_pivot = df_pivot.sort_index()
    df_pivot["合計"] = df_pivot[target_items].sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
        df_feat[f"{item}_前週平均"] = df_pivot[item].shift(1).rolling(7).mean()

    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["合計_前週平均"] = df_pivot["合計"].shift(1).rolling(7).mean()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = daily_avg.shift(1).expanding().median()

    # カレンダー特徴量付与
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat["祝日前フラグ"] = df_feat.index.map(
        lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df_feat["祝日後フラグ"] = df_feat.index.map(
        lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)

    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]
    return df_feat, df_pivot


# ---------- ステージ1学習＆予測 ----------
def train_and_predict_stage1(
    df_feat,
    df_pivot,
    target_date,
    base_models,
    meta_model_proto,
    feature_list,
    target_items,
    stage1_eval,
):
    stage1_results = {}
    df_past_feat = df_feat[df_feat.index < target_date].tail(600)
    df_past_pivot = df_pivot.loc[df_past_feat.index]

    print(f"  ステージ1: 学習履歴 {len(df_past_feat)} 日分")

    for item in target_items:
        X_train_raw = df_past_feat[feature_list]
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
        meta_model_proto.fit(train_meta, y_train)

        X_target_raw = df_feat.loc[[target_date], feature_list]
        X_target_scaled = scaler.transform(X_target_raw)
        X_target_filtered = selector.transform(X_target_scaled)
        base_preds = np.column_stack(
            [m.predict(X_target_filtered) for m in trained_models]
        )
        pred = meta_model_proto.predict(base_preds)[0]
        stage1_results[f"{item}_予測"] = pred

        true_val = df_pivot.loc[target_date, item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

    return stage1_results


# ---------- ステージ2学習＆予測 ----------
def train_and_predict_stage2(
    all_stage1_rows, stage1_results, df_feat, target_date, target_items
):
    df_stage2_hist = pd.DataFrame(all_stage1_rows[:-1])
    X_train_s2 = df_stage2_hist.drop(columns=["合計"])
    y_train_s2 = df_stage2_hist["合計"]

    print(f"  ステージ2: 学習履歴 {len(df_stage2_hist)} サンプル")

    scaler2 = StandardScaler()
    X_train_scaled2 = scaler2.fit_transform(X_train_s2)
    selector2 = VarianceThreshold(threshold=1e-4)
    X_train_filtered2 = selector2.fit_transform(X_train_scaled2)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered2, y_train_s2)

    X_target_s2 = {
        f"{item}_予測": [stage1_results[f"{item}_予測"]] for item in target_items
    }
    X_target_s2.update(
        {
            "曜日": [df_feat.loc[target_date, "曜日"]],
            "週番号": [df_feat.loc[target_date, "週番号"]],
            "合計_前日値": [df_feat.loc[target_date, "合計_前日値"]],
            "合計_前週平均": [df_feat.loc[target_date, "合計_前週平均"]],
            "1台あたり重量_過去中央値": [
                df_feat.loc[target_date, "1台あたり重量_過去中央値"]
            ],
            "祝日フラグ": [df_feat.loc[target_date, "祝日フラグ"]],
            "祝日前フラグ": [df_feat.loc[target_date, "祝日前フラグ"]],
            "祝日後フラグ": [df_feat.loc[target_date, "祝日後フラグ"]],
        }
    )
    X_target_s2 = pd.DataFrame(X_target_s2)
    X_target_scaled2 = scaler2.transform(X_target_s2)
    X_target_filtered2 = selector2.transform(X_target_scaled2)
    total_pred = gbdt.predict(X_target_filtered2)[0]
    return total_pred


# ---------- ステージ1評価出力 ----------
def evaluate_stage1(stage1_eval, target_items):
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{item}: R² = {r2:.3f}, MAE = {mae:,.0f}kg")


# ---------- 完全ウォークフォワード本体 ----------
def full_walkforward(df_raw, holidays, top_n=5):
    target_items = get_target_items(df_raw, top_n=top_n)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    df_feat, df_pivot = generate_weight_features(df_raw, target_items, holidays)

    feature_list = (
        [f"{item}_前日値" for item in target_items]
        + [f"{item}_前週平均" for item in target_items]
        + [
            "合計_前日値",
            "合計_3日平均",
            "合計_3日合計",
            "合計_前週平均",
            "曜日",
            "週番号",
            "1台あたり重量_過去中央値",
            "祝日フラグ",
            "祝日前フラグ",
            "祝日後フラグ",
        ]
    )

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    total_dates = df_feat.index
    all_actual, all_pred = [], []
    all_stage1_rows = []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}

    for i, target_date in enumerate(total_dates):
        if i < 30:
            continue

        print(f"\n=== 現在予測中の日付: {target_date.strftime('%Y-%m-%d')} ===")

        stage1_results = train_and_predict_stage1(
            df_feat,
            df_pivot,
            target_date,
            base_models,
            meta_model_stage1,
            feature_list,
            target_items,
            stage1_eval,
        )

        stage1_row_for_s2 = {
            f"{item}_予測": stage1_results[f"{item}_予測"] for item in target_items
        }
        stage1_row_for_s2.update(
            {
                "曜日": df_feat.loc[target_date, "曜日"],
                "週番号": df_feat.loc[target_date, "週番号"],
                "合計_前日値": df_feat.loc[target_date, "合計_前日値"],
                "合計_前週平均": df_feat.loc[target_date, "合計_前週平均"],
                "1台あたり重量_過去中央値": df_feat.loc[
                    target_date, "1台あたり重量_過去中央値"
                ],
                "祝日フラグ": df_feat.loc[target_date, "祝日フラグ"],
                "祝日前フラグ": df_feat.loc[target_date, "祝日前フラグ"],
                "祝日後フラグ": df_feat.loc[target_date, "祝日後フラグ"],
                "合計": df_pivot.loc[target_date, "合計"],
            }
        )
        all_stage1_rows.append(stage1_row_for_s2)

        if len(all_stage1_rows) > 30:
            total_pred = train_and_predict_stage2(
                all_stage1_rows, stage1_results, df_feat, target_date, target_items
            )
            total_actual = df_pivot.loc[target_date, "合計"]
            all_actual.append(total_actual)
            all_pred.append(total_pred)

    print("\n===== ステージ2評価結果 (全体合計) =====")
    if len(all_actual) > 0:
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
    else:
        print("十分な評価データがありません")

    evaluate_stage1(stage1_eval, target_items)
    return all_actual, all_pred
