FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /works

COPY requirements.txt ./requirements.txt

# VSCode 内部実行用: 外部ポート公開しない

# 起動時インストール後アイドル (VSCode が attach してノートブック実行)
ENTRYPOINT ["bash", "-c", "set -e; (pip install --upgrade pip >/dev/null 2>&1 || true); (pip install --no-cache-dir -r requirements.txt || echo 'WARNING: dependency install failed'); exec sleep infinity"]
