from __future__ import annotations
import os
import time
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Predict Model API", version="0.1.0")

MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "/works/models/model_bundle.joblib")
_model_bundle = None
_model_mtime = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_mtime: Optional[float]

class PredictRequest(BaseModel):
    dates: List[str] = Field(..., description="予測対象日付 (YYYY-MM-DD) の配列")
    # 追加で exogenous features を直接受ける場合はここにフィールドを定義

class PredictItemResponse(BaseModel):
    date: str
    yhat: float

class PredictResponse(BaseModel):
    predictions: List[PredictItemResponse]
    elapsed_ms: float


def _load_model_if_needed(force: bool = False):
    global _model_bundle, _model_mtime
    try:
        if not os.path.exists(MODEL_BUNDLE_PATH):
            return False
        mtime = os.path.getmtime(MODEL_BUNDLE_PATH)
        if force or _model_bundle is None or _model_mtime != mtime:
            _model_bundle = joblib.load(MODEL_BUNDLE_PATH)
            _model_mtime = mtime
        return True
    except Exception:
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    _load_model_if_needed(force=True)

@app.get("/health", response_model=HealthResponse)
async def health():
    exists = _load_model_if_needed()
    return HealthResponse(
        status="ok",
        model_loaded=exists and _model_bundle is not None,
        model_path=MODEL_BUNDLE_PATH,
        model_mtime=_model_mtime,
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not _load_model_if_needed():
        raise HTTPException(status_code=500, detail="Model bundle not found")
    if _model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    t0 = time.time()

    # モデルバンドル仕様に合わせて日付→特徴量変換を実装
    # ここではサンプルとして単純に日付を index に持つDataFrameを生成し、
    # bundle 内に 'stage2_models' に total 予測モデルがある前提で処理
    dates = pd.to_datetime(req.dates, errors='coerce')
    if dates.isna().any():
        raise HTTPException(status_code=400, detail="Invalid date format in request")

    # 簡易例: 過去履歴からラグ特徴を再構築できる情報が bundle['history_tail'] に入っている想定
    hist = _model_bundle.get("history_tail") if isinstance(_model_bundle, dict) else None
    if hist is None or len(hist) == 0:
        raise HTTPException(status_code=500, detail="history_tail missing in bundle")

    # ここでは最も単純に: 直近総量の平均を予測値にするダミー
    # 実運用では predict_model_v4_2_4_best_repro.py のロジックを関数化して再利用する
    avg_val = float(np.mean(hist.select_dtypes(include=[float,int]).sum(axis=1))) if len(hist) else 0.0
    preds = []
    for d in dates:
        preds.append(PredictItemResponse(date=d.strftime('%Y-%m-%d'), yhat=avg_val))

    elapsed_ms = (time.time() - t0) * 1000.0
    return PredictResponse(predictions=preds, elapsed_ms=elapsed_ms)

@app.post("/reload")
async def reload_model():
    ok = _load_model_if_needed(force=True)
    if not ok:
        raise HTTPException(status_code=500, detail="Reload failed")
    return {"status": "reloaded", "model_mtime": _model_mtime}
