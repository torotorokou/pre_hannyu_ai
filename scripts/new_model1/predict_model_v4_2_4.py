import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# === 特徴量作成（モジュールごとに分割） ===
from .feature_builder import (
    WeightFeatureBuilder,
    ReserveFeatureBuilder,
)


def get_target_items(df_raw, top_n=5):
    """
    データフレームから上位の品目を取得します。

    Args:
        df_raw (pd.DataFrame): 元のデータフレーム。
        top_n (int): 取得する品目の数。

    Returns:
        list: 上位の品目リスト。
    """
    return df_raw["品名"].value_counts().head(top_n).index.tolist()


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
    """
    ステージ1のモデルを訓練し、予測を行います。

    Args:
        df_feat_today (pd.DataFrame): 今日の特徴量データ。
        df_past_feat (pd.DataFrame): 過去の特徴量データ。
        df_past_pivot (pd.DataFrame): 過去のピボットデータ。
        base_models (list): 基本モデルのリスト。
        meta_model_proto (object): メタモデルのプロトタイプ。
        feature_list (list): 使用する特徴量のリスト。
        target_items (list): 予測対象の品目リスト。
        stage1_eval (dict): ステージ1の評価結果を格納する辞書。
        df_pivot (pd.DataFrame): ピボットデータ。

    Returns:
        dict: 品目ごとの予測結果。
    """
    print("\n▶️ train_and_predict_stage1 開始")
    print(f"[DEBUG] train_and_predict_stage1: X_train.shape={getattr(df_past_feat[feature_list], 'shape', None)}, df_past_pivot_cols={list(df_past_pivot.columns)[:5]}")

    results = {}
    X_train = df_past_feat[feature_list]
    for item in target_items:
        y_train = df_past_pivot[item]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        selector = VarianceThreshold(1e-4)
        X_train_filtered = selector.fit_transform(X_train_scaled)

        trained_models = []
        for name, model in base_models:
            print(f"🛠 モデル訓練中: {name} for {item}")
            m = clone(model)
            m.fit(X_train_filtered, y_train)
            trained_models.append(m)

        meta_input_train = np.column_stack(
            [m.predict(X_train_filtered) for m in trained_models]
        )
        meta_model = clone(meta_model_proto)
        meta_model.fit(meta_input_train, y_train)

        X_target = df_feat_today[feature_list]
        X_target_scaled = scaler.transform(X_target)
        X_target_filtered = selector.transform(X_target_scaled)
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
    """
    ステージ2のモデルを訓練し、予測を行います。

    Args:
        all_stage1_rows (list): ステージ1の全行データ。
        stage1_results (dict): ステージ1の予測結果。
        df_feat_today (pd.DataFrame): 今日の特徴量データ。
        target_items (list): 予測対象の品目リスト。

    Returns:
        tuple: 合計予測値、特徴量名、重要度。
    """
    print("\n▶️ train_and_predict_stage2 開始")
    print(f"[DEBUG] train_and_predict_stage2: all_stage1_rows_count={len(all_stage1_rows)} sample_keys={list(all_stage1_rows[-1].keys())[:8]}")
    df_hist = pd.DataFrame(all_stage1_rows[:-1])

    X_train = df_hist.drop(columns=["合計"])
    y_train = df_hist["合計"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    selector = VarianceThreshold(1e-4)
    X_train_filtered = selector.fit_transform(X_train_scaled)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered, y_train)

    # ←★ 特徴量名の抽出（selector を通過したものだけ）
    feature_names = np.array(X_train.columns)[selector.get_support()]
    importances = gbdt.feature_importances_

    # ターゲットデータの準備
    X_target = {
        f"{item}_予測": [stage1_results[f"{item}_予測"]] for item in target_items
    }
    for col in df_feat_today.columns:
        if col not in X_target:
            X_target[col] = df_feat_today.iloc[0][col]
    X_target = pd.DataFrame(X_target)

    X_target_scaled = scaler.transform(X_target)
    X_target_filtered = selector.transform(X_target_scaled)
    total_pred = gbdt.predict(X_target_filtered)[0]

    print(f"✅ 合計予測: {total_pred:.1f}kg")

    return total_pred, feature_names, importances


def evaluate_stage1(stage1_eval, target_items):
    """
    ステージ1の評価結果を計算し、表示します。

    Args:
        stage1_eval (dict): ステージ1の評価結果。
        target_items (list): 予測対象の品目リスト。

    Returns:
        None
    """
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        max_error = np.max(np.abs(y_true - y_pred))
        print(
            f"{item}: R² = {r2:.3f}, MAE = {mae:,.0f}kg, RMSE = {rmse:,.0f}kg, 最大誤差 = {max_error:,.0f}kg"
        )


def evaluate_stage1_feature_importance_as_table(records, top_k=15):
    """
    ステージ1の特徴量重要度をテーブル形式で表示します。

    Args:
        records (list): 特徴量重要度の記録。
        top_k (int): 表示する上位の件数。

    Returns:
        pd.DataFrame: 特徴量重要度のデータフレーム。
    """
    df = pd.DataFrame(records)

    # モデル・品目・特徴量ごとに平均を取る（複数日分あれば）
    df_summary = (
        df.groupby(["品目", "モデル", "特徴量"])["重要度"]
        .mean()
        .reset_index()
        .sort_values("重要度", ascending=False)
    )

    print("\n===== ステージ1 特徴量重要度ランキング =====")
    print(df_summary.head(top_k).to_string(index=False))

    return df_summary


def full_walkforward(
    df_raw, holidays, df_reserve, min_stage1_days, min_stage2_days, top_n=5
):
    """
    ウォークフォワード法を実行します。

    Args:
        df_raw (pd.DataFrame): 元のデータフレーム。
        holidays (list): 祝日のリスト。
        df_reserve (pd.DataFrame): 予約データ。
        df_raw (pd.DataFrame): 元のデータフレーム。
        holidays (list): 祝日のリスト。
        df_reserve (pd.DataFrame): 予約データ。
        min_stage1_days (int): ステージ1の最小日数。
        min_stage2_days (int): ステージ2の最小日数。
        top_n (int): 上位の品目数。

    Returns:
        tuple: 実際の値と予測値のリスト。
    """
    print("▶️ full_walkforward 開始")
    print(f"[DEBUG] full_walkforward: df_raw.shape={getattr(df_raw, 'shape', None)}, holidays_type={type(holidays)} df_reserve.shape={getattr(df_reserve, 'shape', None)}")

    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)

    builder = WeightFeatureBuilder(df_raw, target_items, holidays)
    df_feat, df_pivot = builder.build()

    reserve_builder = ReserveFeatureBuilder(df_reserve)
    df_reserve_feat_all = reserve_builder.build()

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    feature_list = [
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
        "第2日曜フラグ",
        "連休前フラグ",
        "連休後フラグ",
        "予約件数",
        "予約合計台数",
        "固定客予約数",
        "上位得意先予約数",
    ]

    all_actual, all_pred, all_stage1_rows = [], [], []
    stage1_feat_importances = []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index
    # 予測が1件でも生成されるために必要な最小日数を計算
    required_min_predictions = min_stage1_days + min_stage2_days + 1
    print(f"[DEBUG] full_walkforward: dates_len={len(dates)} min_stage1_days={min_stage1_days} min_stage2_days={min_stage2_days} -> required_min_predictions={required_min_predictions}")
    if len(dates) < required_min_predictions:
        print("[DEBUG] 有効日数が不足: ステージ2予測は生成されません (学習可能日数 < 必要日数)")

    for i, target_date in enumerate(dates):
        if i < min_stage1_days:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(600)
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_today = df_reserve_feat_all[df_reserve_feat_all.index <= target_date]
        df_past_feat = df_past_feat.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
        ).fillna(0)
        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
        ).fillna(0)

        print(f"\n=== {target_date.strftime('%Y-%m-%d')} を予測中 ===")
        print(f"[DEBUG] iteration i={i}: df_past_feat.shape={getattr(df_past_feat, 'shape', None)}, df_feat_today.shape={getattr(df_feat_today, 'shape', None)}")
        stage1_result = train_and_predict_stage1(
            df_feat_today,
            df_past_feat,
            df_past_pivot,
            base_models,
            meta_model_proto,
            feature_list,
            target_items,
            stage1_eval,
            df_pivot,
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
            total_actual = df_pivot.loc[target_date, "合計"]
            all_actual.append(total_actual)
            all_pred.append(total_pred)
            # ランニング指標表示
            try:
                r2_run = r2_score(all_actual, all_pred)
                mae_run = mean_absolute_error(all_actual, all_pred)
                print(f"[PROGRESS] 現在の予測件数={len(all_actual)} R²={r2_run:.3f} MAE={mae_run:,.0f}kg")
            except Exception as _e_metrics:
                print(f"[DEBUG] ランニング指標計算失敗: {_e_metrics}")
        else:
            # 進捗状況を定期的に表示（最初の数回のみ冗長）
            if len(all_stage1_rows) <= 5 or len(all_stage1_rows) == min_stage2_days:
                print(f"[DEBUG] ステージ2未実行: all_stage1_rows={len(all_stage1_rows)} / 閾値>{min_stage2_days}")

    print("\n===== ステージ2評価結果 (合計) =====")
    if all_actual:
        r2 = r2_score(all_actual, all_pred)
        mae = mean_absolute_error(all_actual, all_pred)
        rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
        max_error = np.max(np.abs(np.array(all_actual) - np.array(all_pred)))
        print(
            f"R² = {r2:.3f}, MAE = {mae:,.0f}kg, RMSE = {rmse:,.0f}kg, 最大誤差 = {max_error:,.0f}kg"
        )
    else:
        print("評価できるデータが不足しています")

    evaluate_stage1(stage1_eval, target_items)
    # ステージ2評価の後に
    evaluate_stage1_feature_importance_as_table(stage1_feat_importances, top_k=20)
    return all_actual, all_pred


def plot_feature_importances(names, importances, title="Feature Importance", top_k=15):
    """
    特徴量重要度をプロットします。

    Args:
        names (list): 特徴量名のリスト。
        importances (list): 特徴量の重要度。
        title (str): プロットのタイトル。
        top_k (int): 表示する上位の件数。

    Returns:
        None
    """
    sorted_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), np.array(names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def evaluate_feature_importance_as_table(feature_names, importances, top_k=15):
    """
    特徴量重要度をテーブル形式で表示します。

    Args:
        feature_names (list): 特徴量名のリスト。
        importances (list): 特徴量の重要度。
        top_k (int): 表示する上位の件数。

    Returns:
        pd.DataFrame: 特徴量重要度のデータフレーム。
    """
    df_importance = pd.DataFrame(
        {"特徴量": feature_names, "重要度": importances}
    ).sort_values("重要度", ascending=False)

    top_df = df_importance.head(top_k)
    print("\n===== 特徴量重要度ランキング =====")
    print(top_df.to_string(index=False))
    return top_df
