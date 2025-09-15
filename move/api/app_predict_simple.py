"""FastAPI prediction API with feature generation and prediction endpoints"""

import os
import pickle
import sys
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional, Any, TypeVar, Generic
import numpy as np

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jpholiday

# --- Response wrapper ---
T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    """API response wrapper"""
    code: str
    detail: str
    result: Optional[T] = None
    
    @classmethod
    def success(cls, code: str, detail: str, result: T = None):
        return cls(code=code, detail=detail, result=result)
    
    @classmethod
    def error(cls, code: str, detail: str):
        return cls(code=code, detail=detail, result=None)

# --- Request/Response Models for Feature Generation ---
class PreprocessFeaturesRequest(BaseModel):
    date: _date = Field(..., description="Target date in YYYY-MM-DD")
    yoyaku_count: int = Field(..., description="予約件数")
    yoyaku_total: int = Field(..., description="予約合計台数")
    fixed_customer_count: int = Field(..., description="固定客予約数")
    top_customer_count: int = Field(..., description="上位得意先予約数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-09-15",
                "yoyaku_count": 12,
                "yoyaku_total": 20,
                "fixed_customer_count": 3,
                "top_customer_count": 5
            }
        }
    }

class PreprocessFeaturesResult(BaseModel):
    date: str
    features: Dict[str, float]

# --- Request/Response Models for Prediction ---
class PredictWithFeaturesRequest(BaseModel):
    """Request body for /predict/with-features."""
    date: _date = Field(..., description="Target date in YYYY-MM-DD")
    features: Dict[str, float] = Field(
        ..., description="Feature name to value map; missing features will be treated as 0 by the model"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2099-01-01",
                "features": {
                    "曜日": 2,
                    "週番号": 35,
                    "祝日フラグ": 0,
                    "予約件数": 12,
                    "予約合計台数": 20,
                    "固定客予約数": 3,
                    "上位得意先予約数": 5,
                    "天気_晴れ": 1,
                },
            }
        }
    }

class PredictResult(BaseModel):
    """Result payload returned by predict endpoints."""
    date: str
    per_item: Dict[str, float]
    total: float
    used_features: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-09-13",
                "per_item": {"ITEM_A": 123.4, "ITEM_B": 56.7},
                "total": 180.1,
                "used_features": ["weekday", "is_holiday", "lag_7d_sum"],
            }
        }
    }

class PredictRequest(BaseModel):
    date: _date = Field(..., description="Target date in YYYY-MM-DD")
    yoyaku_count: int = Field(..., description="予約件数")
    yoyaku_total: int = Field(..., description="予約合計台数")
    fixed_customer_count: int = Field(..., description="固定客予約数")
    top_customer_count: int = Field(..., description="上位得意先予約数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-09-15",
                "yoyaku_count": 30,
                "yoyaku_total": 110,
                "fixed_customer_count": 10,
                "top_customer_count": 5
            }
        }
    }

# --- Workspace detection ---
def detect_workspace_root() -> Path:
    """Detect workspace root"""
    # Check if we're in /works (container environment)
    if Path("/works").exists():
        return Path("/works")
    # Otherwise use current directory structure
    here = Path(__file__).resolve().parent
    p = here
    for _ in range(5):
        if (p / "data").exists() or (p / "notebooks").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd()

# --- Global vars ---
WS_ROOT = detect_workspace_root()

# Load model once at startup
_model = None
_features_list = None

# --- Predictor Provider (from app_predict.py) ---
class PredictorProvider:
    """Provide a loaded predictor instance from disk."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = (WS_ROOT / "data" / "final_stage1_model_api.pkl").as_posix()
        self._model_path = model_path
        self._predictor = None  # lazy-loaded
        
        # Ensure scripts path for imports
        self._ensure_scripts_on_path()

    def _ensure_scripts_on_path(self) -> None:
        """Ensure /scripts is importable so that new_model* packages can be imported."""
        candidates = [
            WS_ROOT.parent.resolve(),
            (WS_ROOT.parent / "scripts").resolve(),
            WS_ROOT.resolve(),
            (WS_ROOT / "scripts").resolve(),
        ]
        # 追加順序を逆順にして、candidates 先頭のものが最優先で先頭に来るようにする
        for d in reversed(candidates):
            if d.exists():
                p = d.as_posix()
                if p not in sys.path:
                    sys.path.insert(0, p)

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        """Load predictor if present and not yet loaded."""
        if self._predictor is not None:
            return
        if not os.path.exists(self._model_path):
            return
        
        # Try normal pickle load first
        try:
            with open(self._model_path, "rb") as f:
                self._predictor = pickle.load(f)
            return
        except Exception:
            pass

        # Compatibility unpickler for class resolution issues
        import importlib
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                target_path = os.getenv(
                    "PREDICTOR_CLASS",
                    "scripts.new_model2.feature_builder.NewModel2Predictor",
                )
                if name == "NewModel2Predictor" or (
                    module == "__main__" and name.endswith("Predictor")
                ):
                    try:
                        mod_name, cls_name = target_path.rsplit(".", 1)
                        mod = importlib.import_module(mod_name)
                        return getattr(mod, cls_name)
                    except Exception:
                        # Fallback: try common locations
                        for cand in [
                            "scripts.new_model2.feature_builder.NewModel2Predictor",
                            "works.scripts.new_model2.feature_builder.NewModel2Predictor",
                            "scripts.new_model2.predict_model_v4_2_4.NewModel2Predictor",
                            "new_model2.predict_model_v4_2_4.NewModel2Predictor",
                        ]:
                            try:
                                m, c = cand.rsplit(".", 1)
                                mod = importlib.import_module(m)
                                return getattr(mod, c)
                            except Exception:
                                continue
                return super().find_class(module, name)

        with open(self._model_path, "rb") as f:
            self._predictor = _CompatUnpickler(f).load()

        # --- Post-load fixups: ensure real estimator and features are present ---
        try:
            pred = self._predictor
            # 1) allowed_features 補完
            if getattr(pred, "allowed_features", None) in (None, [], ()):
                sel_path_candidates = [
                    (WS_ROOT / "data" / "selected_features_final.txt").resolve(),
                    (WS_ROOT.parent / "data" / "selected_features_final.txt").resolve(),
                ]
                for pth in sel_path_candidates:
                    try:
                        if pth.exists():
                            with open(pth.as_posix(), encoding="utf-8") as f:
                                pred.allowed_features = [ln.strip() for ln in f if ln.strip()]
                            break
                    except Exception:
                        continue

            # 2) 推論器注入（predictを持たない場合）
            need_estimator = False
            try:
                m = getattr(pred, "_model", None)
                if not hasattr(m, "predict"):
                    need_estimator = True
            except Exception:
                need_estimator = True
            if need_estimator:
                cand_paths = [
                    (WS_ROOT / "data" / "final_stage1_model_predictable.pkl").resolve(),
                    (WS_ROOT / "data" / "final_stage1_model.pkl").resolve(),
                    (WS_ROOT.parent / "data" / "final_stage1_model_predictable.pkl").resolve(),
                    (WS_ROOT.parent / "data" / "final_stage1_model.pkl").resolve(),
                ]
                for mp in cand_paths:
                    try:
                        if mp.exists():
                            with open(mp.as_posix(), "rb") as mf:
                                est = pickle.load(mf)
                            if hasattr(est, "predict"):
                                setattr(pred, "_model", est)
                                break
                    except Exception:
                        continue

            # 3) 属性の安全な既定化（unpickle元で未定義のケースに対応）
            # _target_items は存在しないと AttributeError になるので None で初期化
            if not hasattr(pred, "_target_items"):
                # meta から復元を試みる
                items = None
                try:
                    meta = getattr(pred, "_meta", None)
                    if isinstance(meta, dict):
                        for k in ("target_items", "items", "labels"):
                            v = meta.get(k)
                            if isinstance(v, (list, tuple)) and len(v) > 0:
                                items = list(v)
                                break
                except Exception:
                    items = None
                setattr(pred, "_target_items", items)
            # _item_shares も未定義なら None で初期化
            if not hasattr(pred, "_item_shares"):
                setattr(pred, "_item_shares", None)

            # 4) _target_items が空/None の場合、df_raw から推定して補完し、_item_shares も算出
            try:
                if not getattr(pred, "_target_items", None):
                    df_raw = getattr(pred, "df_raw", None)
                    if df_raw is not None and {"伝票日付", "品名", "正味重量"}.issubset(set(df_raw.columns)):
                        import pandas as _pd
                        import numpy as _np
                        df = df_raw[["伝票日付", "品名", "正味重量"]].copy()
                        df["伝票日付"] = _pd.to_datetime(df["伝票日付"])
                        pivot = df.groupby(["伝票日付", "品名"])['正味重量'].sum().unstack(fill_value=0)
                        items = [str(c) for c in pivot.columns]
                        setattr(pred, "_target_items", items)
                        total = pivot.sum(axis=1)
                        with _np.errstate(divide='ignore', invalid='ignore'):
                            shares = (pivot.T / _np.where(total.values == 0, 1, total.values)).T
                        mean_shares = shares.replace([_np.inf, -_np.inf], 0).fillna(0).mean(axis=0)
                        s = mean_shares.values
                        if s.sum() <= 0 and len(items) > 0:
                            s = _np.ones(len(items)) / len(items)
                        else:
                            s_sum = s.sum()
                            s = s / (s_sum if s_sum != 0 else 1)
                        setattr(pred, "_item_shares", s)
            except Exception:
                pass
        except Exception:
            # フィックスに失敗してもAPI起動は継続
            pass

    def get(self):
        return self._predictor

# Initialize provider
_provider_singleton = PredictorProvider()

def generate_features(date: _date, yoyaku_count: int, yoyaku_total: int, 
                     fixed_customer_count: int, top_customer_count: int) -> Dict[str, float]:
    """Generate features from input data"""
    
    # Try to load features list
    sel_path = WS_ROOT / "data" / "selected_features_final.txt"
    if sel_path.exists():
        with open(sel_path, encoding="utf-8") as f:
            features_list = [ln.strip() for ln in f if ln.strip()]
    else:
        # Default features
        features_list = [
            "曜日", "週番号", "祝日フラグ", "予約件数", "予約合計台数", 
            "固定客予約数", "上位得意先予約数", "天気_晴れ", "天気_雨", 
            "天気_大雨", "天気_台風"
        ]
    
    # 日付情報
    weekday = date.weekday()  # 0=月, 6=日
    weeknum = date.isocalendar()[1]
    is_holiday = int(jpholiday.is_holiday(date) or weekday >= 5)

    # 天気はダミー（晴れのみ1, 他0）
    weather_keys = [k for k in features_list if k.startswith("天気_")]
    weather = {k: (1 if k == "天気_晴れ" else 0) for k in weather_keys}

    # 特徴量ベクトル生成
    feats = {}
    for k in features_list:
        if k == "曜日": 
            feats[k] = float(weekday)
        elif k == "週番号": 
            feats[k] = float(weeknum)
        elif k == "祝日フラグ": 
            feats[k] = float(is_holiday)
        elif k == "予約件数": 
            feats[k] = float(yoyaku_count)
        elif k == "予約合計台数": 
            feats[k] = float(yoyaku_total)
        elif k == "固定客予約数": 
            feats[k] = float(fixed_customer_count)
        elif k == "上位得意先予約数": 
            feats[k] = float(top_customer_count)
        elif k in weather:
            feats[k] = float(weather[k])
        else:
            feats[k] = 0.0
    
    return feats

# --- FastAPI app ---
app = FastAPI(
    title="Prediction API",
    version="1.0.0",
    description="予測API - 特徴量生成と搬入量予測",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    _provider_singleton.load()

# --- Routes ---
@app.get("/health")
def health():
    """Health check"""
    model_status = "loaded" if _provider_singleton.is_loaded else "not_loaded"
    return ApiResponse.success(
        code="HEALTH_OK", 
        detail=f"API is running. Model: {model_status}",
        result={
            "model_loaded": _provider_singleton.is_loaded
        }
    )

@app.post(
    "/preprocess/features",
    response_model=ApiResponse[PreprocessFeaturesResult],
    summary="日付＋予約情報から特徴量ベクトルを自動生成",
    tags=["features"],
)
def preprocess_features(
    payload: PreprocessFeaturesRequest = Body(...),
):
    """日付＋予約情報から特徴量ベクトルを自動生成して返すAPI"""
    try:
        features = generate_features(
            payload.date, payload.yoyaku_count, payload.yoyaku_total,
            payload.fixed_customer_count, payload.top_customer_count
        )
        
        result = PreprocessFeaturesResult(date=payload.date.isoformat(), features=features)
        return ApiResponse.success(code="PREPROCESS_OK", detail="特徴量生成完了", result=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"特徴量生成エラー: {str(e)}")

@app.post(
    "/predict/volume",
    response_model=ApiResponse[PredictResult],
    summary="搬入量予測API",
    tags=["prediction"],
)
def predict_volume(
    payload: PredictRequest = Body(...),
):
    """日付＋予約情報から搬入量を予測するAPI"""
    try:
        # 特徴量生成
        features = generate_features(
            payload.date, payload.yoyaku_count, payload.yoyaku_total,
            payload.fixed_customer_count, payload.top_customer_count
        )
        
        # 予測実行
        if _model is None:
            # モデルがない場合はダミー値
            predicted_volume = float(payload.yoyaku_total * 1.2)  # 予約台数の1.2倍をダミー予測値
            confidence_score = 0.0
        else:
            # 特徴量をモデル用に配列変換
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            predicted_volume = float(_model.predict(feature_array)[0])
            
            # 信頼度スコア（モデルによって異なる）
            try:
                confidence_score = float(_model.predict_proba(feature_array).max())
            except:
                confidence_score = None
        
        result = PredictResult(
            date=payload.date.isoformat(),
            predicted_volume=predicted_volume,
            confidence_score=confidence_score,
            features_used=features
        )
        
        return ApiResponse.success(code="PREDICT_OK", detail="予測完了", result=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

@app.post(
    "/predict/with-features",
    response_model=ApiResponse[PredictResult],
    summary="Predict for a given date with explicit feature map (future ok)",
    tags=["prediction"],
)
def predict_with_features(
    payload: PredictWithFeaturesRequest = Body(...),
):
    """Predict using user-provided feature map. Missing features are treated as zero by the model."""
    try:
        predictor = _provider_singleton.get()
        if predictor is None:
            raise HTTPException(status_code=500, detail="モデル未ロード")
        
        # Ensure numeric feature values (coerce to float)
        feats = {str(k): float(v) for k, v in payload.features.items()}
        out = predictor.predict_with_features(payload.date.isoformat(), feats)
        result = PredictResult(**out)
        return ApiResponse.success(code="PREDICT_OK", detail="推論完了", result=result)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"予測に失敗: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)