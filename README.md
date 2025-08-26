# 搬入量予測AI 用 Docker / Notebook 環境（リポジトリ現状版）

## 目的
このリポジトリ内の Notebook（例: `notebooks/pre_ryou_ai2.ipynb`）や分析スクリプトをコンテナ上で再現・実行するための軽量環境を提供します。

## リポジトリ構成（現状）
```
./
  Dockerfile               # ルートに配置されたコンテナ定義
  docker-compose.yml       # サービス定義（Notebook 等）
  requirements.txt         # 依存パッケージ
  notebooks/               # Jupyter Notebook を置く場所
  data/                    # Notebook が参照するデータ（ローカルに置くかマウント）
  scripts/                 # 起動やコピー等の補助スクリプト
  vendor/                  # アプリ本体やユーティリティコード（参照可能）
```

## 使い方
### 1) Notebook / データを準備
手元の Notebook と参照データを `notebooks/` `data/` にコピーします。手動か補助スクリプトを使ってください。

手動例:
```bash
cp pre_ryou_ai2.ipynb notebooks/
cp vendor/app/data/factory_manage/weight_data.db data/   # 必要ならパスを調整
cp vendor/app/data/input/*.csv data/                     # 必要な CSV をコピー
```

補助スクリプトがある場合（存在しない場合は手動で）:
```bash
python scripts/copy_assets.py --help
```

### 2) コンテナのビルドと起動
リポジトリルートで以下を実行します（`docker` と `docker compose` がインストール済みであること）：
```bash
docker compose build
docker compose up -d
```
Jupyter にアクセスする場合はログ出力に表示される URL（通常 http://localhost:8888）を参照してください。

### 3) 既存アプリコードへの参照
`docker-compose.yml` で `vendor/` や `logic/` 等をボリュームマウントしている場合、Notebook 側でそのままアプリのモジュールを import できます。パスが合わない場合は `PYTHONPATH` を調整してください。

### 4) よくあるトラブルと対処
- 文字化け: コンテナ内で `ja_JP.UTF-8` が利用可能か確認してください（Dockerfile でロケールを作ることを推奨）。
- モジュールが見つからない: `PYTHONPATH` の設定を確認。Notebook 内で `!echo $PYTHONPATH` で確認できます。
- 追加ライブラリ: `requirements.txt` に追記後、`docker compose build --no-cache` を実行してください。

### 5) クリーンアップ
```bash
docker compose down -v
```

## 次の改善案（任意）
- 大きなデータを扱う場合は `data/` を Git 管理から外し、代わりに Git LFS や外部ストレージを使う
- CI（GitHub Actions 等）で依存テストを自動化
- Notebook の再現性のために環境を `requirements.txt` で固定化

---
変更点: README をリポジトリの現状（ルートに Dockerfile/docker-compose がある構成）に合わせて簡潔化・誤字修正しました。

