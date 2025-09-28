FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /works

COPY requirements.txt ./requirements.txt

# 代表的なアンサンブル・スタッキング用パッケージをrequirements.txtに記載していない場合でも直接インストールできるようにする例（必要に応じてrequirements.txtも修正推奨）
# RUN pip install --no-cache-dir lightgbm catboost xgboost scikit-learn torch torchvision torchaudio tensorflow

# 書き込み権限付与（/works配下に全ユーザー書き込み可）
RUN chmod -R a+w /works

# VSCode 内部実行用: 外部ポート公開しない

# 起動時インストール後アイドル (VSCode が attach してノートブック実行)
ENTRYPOINT ["bash", "-c", "set -e; (pip install --upgrade pip >/dev/null 2>&1 || true); (pip install --no-cache-dir -r requirements.txt || echo 'WARNING: dependency install failed'); exec sleep infinity"]
