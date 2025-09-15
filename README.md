# 搬入量予測API

## 概要
日付と予約情報から廃棄物の搬入量を予測するFastAPI-ベースのWebAPIです。機械学習モデルを使用して、256品目の詳細な搬入量予測を提供します。

## 🚀 API機能

### エンドポイント一覧
- `GET /health` - ヘルスチェック（モデル読み込み状態確認）
- `POST /preprocess/features` - 特徴量生成（日付+予約情報→機械学習用特徴量）
- `POST /predict/with-features` - 搬入量予測（特徴量→品目別搬入量予測）
- `GET /docs` - Swagger UI（API仕様とテスト画面）

## 📁 プロジェクト構成
```
./
├── docker-compose.yml          # サービス定義（API/Jupyter）
├── Dockerfile                  # Jupyter環境用
├── Dockerfile.api              # API環境用
├── requirements.txt            # 依存パッケージ
├── move/                       # メインプロジェクト
│   ├── api/
│   │   └── app_predict_simple.py  # FastAPIアプリケーション
│   ├── data/                   # モデルファイル・特徴量定義
│   ├── notebooks/              # 開発用Jupyter Notebook
│   └── scripts/                # モデル・特徴量生成スクリプト
└── test_api_flow.sh           # エンドツーエンドテストスクリプト
```

## 🔧 セットアップ

### 前提条件
- Docker Desktop
- Docker Compose

### API環境の起動
```bash
# APIサーバーの起動
docker compose --profile api up --build

# バックグラウンドで起動する場合
docker compose --profile api up -d --build
```

### Jupyter開発環境の起動（開発者向け）
```bash
# Jupyter Labの起動
docker compose --profile jupyter up --build
```

### 両方同時起動
```bash
# API + Jupyter環境を同時に起動
docker compose --profile api --profile jupyter up --build
```

## 📡 API利用方法

### 1. ヘルスチェック
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. 特徴量生成
```bash
curl -X POST "http://localhost:8000/preprocess/features" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-09-15",
    "yoyaku_count": 30,
    "yoyaku_total": 110,
    "fixed_customer_count": 10,
    "top_customer_count": 5
  }'
```

### 3. 搬入量予測
```bash
curl -X POST "http://localhost:8000/predict/with-features" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-09-15",
    "features": {
      "曜日": 0.0,
      "週番号": 38.0,
      "祝日フラグ": 1.0,
      "予約件数": 30.0,
      "予約合計台数": 110.0,
      "固定客予約数": 10.0,
      "上位得意先予約数": 5.0,
      "天気_晴れ": 1.0,
      "天気_雨": 0.0,
      "天気_大雨": 0.0,
      "天気_台風": 0.0
    }
  }'
```

### 4. エンドツーエンドテスト
```bash
./test_api_flow.sh
```

## 🌐 Swagger UI
APIの詳細仕様とインタラクティブなテスト画面：
- URL: http://localhost:8000/docs
- 全エンドポイントの仕様確認
- リアルタイムでのAPIテスト
- リクエスト/レスポンス例の表示

## 📊 予測結果の形式

### 特徴量生成レスポンス
```json
{
  "code": "PREPROCESS_OK",
  "detail": "特徴量生成完了",
  "result": {
    "date": "2025-09-15",
    "features": {
      "曜日": 0.0,
      "週番号": 38.0,
      "祝日フラグ": 1.0,
      "予約件数": 30.0,
      "予約合計台数": 110.0,
      "固定客予約数": 10.0,
      "上位得意先予約数": 5.0,
      "天気_晴れ": 1.0
    }
  }
}
```

### 搬入量予測レスポンス
```json
{
  "code": "PREDICT_OK",
  "detail": "推論完了",
  "result": {
    "date": "2025-09-15",
    "per_item": {
      "混合廃棄物A": 92.57,
      "混合廃棄物B": 22.37,
      "GC 軽鉄･ｽﾁｰﾙ類": 17.31
    },
    "total": 195.0,
    "used_features": ["曜日", "週番号", "祝日フラグ"]
  }
}
```

## 🔍 トラブルシューティング

### コンテナが起動しない
```bash
# ログ確認
docker compose --profile api logs

# 完全なクリーンアップ後に再起動
docker compose down --remove-orphans
docker system prune -f
docker compose --profile api up --build
```

### モデルが読み込まれない
- `/health`エンドポイントで`"model_loaded": false`の場合
- `move/data/`ディレクトリにモデルファイルが存在するか確認
- コンテナのログでモデル読み込みエラーを確認

### APIレスポンスが遅い
- 初回リクエスト時はモデル読み込みで時間がかかる場合があります
- 2回目以降は高速化されます

## 🚀 本番環境への展開

### 環境変数設定例
```bash
# 本番用ポート設定
export PORT=8000
export HOST=0.0.0.0

# CORS設定
export CORS_ALLOW_ORIGINS="https://your-frontend-domain.com"

# モデルファイルパス
export PREDICTOR_MODEL_PATH="/path/to/your/model.pkl"
```

### Docker環境変数
```yaml
environment:
  - PORT=8000
  - HOST=0.0.0.0
  - CORS_ALLOW_ORIGINS=https://your-frontend-domain.com
```

## 📝 開発情報
- FastAPI: 最新版
- Python: 3.10
- 機械学習フレームワーク: scikit-learn
- 品目数: 256種類
- 特徴量数: 15種類（基本）+ 動的特徴量

---

## 📞 サポート
- Swagger UI: http://localhost:8000/docs
- OpenAPI仕様: http://localhost:8000/openapi.json
- ReDoc: http://localhost:8000/redoc

