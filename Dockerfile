FROM python:3.10-slim

# ============== Build-time configuration ==============
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /works

# 依存関係をビルド時に固定インストール (提案B)
COPY requirements.txt ./requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

# 非 root ユーザー追加 (提案C)
RUN groupadd -g ${GROUP_ID} app && useradd -m -u ${USER_ID} -g app app \
    && chown -R app:app /works

# /work 互換シンボリックリンク (既存コード対策)
RUN ln -s /works /work 2>/dev/null || true

USER app

# シンプルな待機 (VSCode attach 前提)。必要に応じて jupyter 起動をここに書き換える。
ENTRYPOINT ["bash", "-c", "exec sleep infinity"]
