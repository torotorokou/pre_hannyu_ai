# スタッキング実験モジュール

代表的な高精度スタッキング構成（A〜D）をワンコマンドで試せる軽量フレーム。

- A: 木系 + 線形 + MLP、メタ=Ridge（既定）
- B: RF + XGB-like + LGBM-like + Cat-like、メタ=XGB-like
- C: LGBM-like + kNN + SVM + MLP、メタ=線形
- D: 特徴別を意識した多様構成（簡易版）、メタ=XGB-like or Ridge

CatBoost/LightGBM/XGBoost が未インストールの場合、sklearnの代替（HistGBDT/RandomForest）に自動フォールバックします。

## 使い方（例）

```bash
python -m scripts.stacking.run_experiments \
  --csv data/input/2023_all.csv \
  --target 目的変数列名 \
  --date-col 日付列名 \
  --preset A \
  --n-splits 5 \
  --ts-cv \
  --robust-meta
```

出力は `outputs/stack_<Preset>_<csv名>/` 以下に保存されます。

すべてのプリセット（A, B, C, D）を一括で試すには、

```bash
python -m scripts.stacking.run_experiments --csv <path> --target <col> --date-col <date> --preset ALL
```

## 依存関係

requirements.txt の scikit-learn で動作。任意で以下を追加すると高精度な実行が可能です。

- lightgbm
- xgboost
- catboost

```bash
pip install lightgbm xgboost catboost
```

## 実装メモ

- OOFを作成し、その列に加えて mean / weighted mean / rank mean / std / 簡易ペアをメタ特徴量化
- メタモデルはRidge（--robust-metaでHuberにも切替）
- TimeSeriesSplitにも対応（--ts-cv）
- 学習後はフルデータでベースモデル再学習し、デプロイ用に joblib で保存