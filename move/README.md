# 搬入量予測 API（学習済みモデル推論）

本フォルダは「最小構成」で、学習済みモデルを用いて搬入量を推論する FastAPI サーバーです。未来日付でも、特徴量を明示すれば per_item と total を返します。

## 構成
```
./
  requirements.txt
  api/
    app_predict.py                       # FastAPI アプリ本体
  data/
    final_stage1_model_api.pkl           # 既定で読み込む学習済みモデル（推論器を内包）
    final_stage1_model_predictable.pkl   # 予備の推論器（不足時の注入用）
    selected_features_final.txt          # 使用する特徴量一覧（改行区切り）
  notebooks/
    api_response/                        # レスポンススキーマ（ApiResponse）
  scripts/
    new_model2/
      __init__.py
      feature_builder.py                 # 学習時と互換の Predictor 実装
```

## 起動方法（ローカル）
以下で FastAPI を起動します。
（カレントディレクトリは move フォルダ想定）
```bash
python -m uvicorn api.app_predict:app --host 0.0.0.0 --port 8000
```
ブラウザ: http://localhost:8000/docs （Swagger UI）

環境変数（任意）
- `PREDICTOR_MODEL_PATH`: 既定以外のモデル pkl を読み込む場合に指定（既定: `data/final_stage1_model_api.pkl`）
- `CORS_ALLOW_ORIGINS`: CORS で許可するオリジン（カンマ区切り）。既定は `*`

## エンドポイント
- GET `/health`
  - モデルロードの状態を返します。
- POST `/predict/with-features`
  - 未来日付を含む任意日付と、特徴量マップを受け取り、予測結果（per_item 内訳と total）を返します。
  - リクエスト例:
    ```json
    {
      "date": "2099-01-01",
      "features": {
        "曜日": 2,
        "週番号": 35,
        "祝日フラグ": 0,
        "予約件数": 12,
        "予約合計台数": 20,
        "固定客予約数": 3,
        "上位得意先予約数": 5,
        "天気_晴れ": 1
      }
    }
    ```
  - レスポンス例（成功・抜粋）:
    ```json
    {
      "status": "success",
      "code": "PREDICT_OK",
      "result": {
        "date": "2099-01-01",
        "per_item": { "木くず": 3.98, "混合廃棄物": 1.49 },
        "total": 89.0,
        "used_features": ["曜日", "週番号", "祝日フラグ", "予約件数", "天気_晴れ"]
      }
    }
    ```

## モデルについて（重要）
- 学習済みの実モデルを使用します（ダミーではありません）。
- 既定では `data/final_stage1_model_api.pkl` をロードします。
  - Unpickle 後に以下を自動補完：
    - allowed_features が未設定なら `selected_features_final.txt` から補完
    - 推論器 `_model` が見つからない場合、`final_stage1_model_predictable*.pkl` から注入
    - 品目内訳 `_target_items` が未設定の場合、`df_raw` から品目一覧と平均シェアを推定
  - これにより、将来日でも、特徴量が与えられれば `per_item` と `total` を返せます。

注意:
- scikit-learn のバージョン差による InconsistentVersionWarning が出ることがあります。推論は可能ですが、長期運用では現行環境での再pickleを推奨します。

## 特徴量
- 使用する特徴量は `data/selected_features_final.txt` に列挙（1行1特徴量）。
- 未指定の特徴量は 0 で補完。余分なキーは無視。
- 天気などは one-hot（例: `天気_晴れ`）を想定。

## 簡易テスト（任意）
Python からの簡易実行例：
```python
import requests, random
base='http://127.0.0.1:8000'
print(requests.get(base+'/health').json())

af=[ln.strip() for ln in open('data/selected_features_final.txt',encoding='utf-8') if ln.strip()]
random.seed(0)
features={}
for col in af:
    if col=='曜日': features[col]=random.randint(0,6)
    elif col=='週番号': features[col]=random.randint(1,53)
    elif col.startswith('天気_') or col.endswith('フラグ'): features[col]=random.randint(0,1)
    else: features[col]=random.randint(0,50)
weather_cols=[c for c in af if c.startswith('天気_')]
if weather_cols:
    ch=random.choice(weather_cols)
    for c in weather_cols:
        features[c]=1 if c==ch else 0

payload={'date':'2099-01-01','features':features}
print(requests.post(base+'/predict/with-features', json=payload).json())
```

## よくある質問
- Q. ダミーモデルですか？
  - A. いいえ。学習済みの実モデルを復元して予測します。推論器が不足している場合も別 pkl から注入し、確実に推論できるようにしています。
- Q. 既知実績日の予測APIは？
  - A. 未来予測に特化し、特徴量を直接指定する `/predict/with-features` のみ提供します。


