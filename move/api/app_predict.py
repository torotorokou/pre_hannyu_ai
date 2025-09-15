from __future__ import annotations

"""FastAPI prediction API (PEP8 + SOLID).

- Domain: predictions served from a saved predictor object
- Response contract: notebooks/api_response.ApiResponse[T]
- Layers:
  * Config: constants, paths
  * Provider: predictor loader (single responsibility)
  * Service: use-case orchestration (depends on provider)
  * API: FastAPI routes (depends on service via DI)

This module keeps imports explicit and only adjusts sys.path for
notebooks/api_response at startup without polluting global state elsewhere.
"""
import os
import pickle
import sys
from datetime import date as _date
import importlib
from pathlib import Path
from typing import Dict, List

from fastapi import Body, Depends, FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import calendar
import jpholiday
# --- Result/Request Models (OpenAPI schema) -------------------------------
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-09-15",
                "features": {
                    "曜日": 1,
                    "週番号": 37,
                    "祝日フラグ": 0,
                    "予約件数": 12,
                    "予約合計台数": 20,
                    "固定客予約数": 3,
                    "上位得意先予約数": 5,
                    "天気_晴れ": 1,
                    "天気_雨": 0,
                    "天気_大雨": 0,
                    "天気_台風": 0
                }
            }
        }
    }
# --- App/Routes (Controller Layer) ----------------------------------------
@app.post(
    "/preprocess/features",
    response_model=ApiResponse[PreprocessFeaturesResult],
    summary="日付＋予約情報から特徴量ベクトルを自動生成",
    tags=["predict"],
)
def preprocess_features(
    payload: PreprocessFeaturesRequest = Body(...),
):
    """日付＋予約情報から特徴量ベクトルを自動生成して返すAPI"""
    # 特徴量リスト取得
    sel_path = _WS_ROOT / "data" / "selected_features_final.txt"
    if not sel_path.exists():
        raise HTTPException(status_code=500, detail="selected_features_final.txt not found")
    with open(sel_path, encoding="utf-8") as f:
        features_list = [ln.strip() for ln in f if ln.strip()]

    # 日付情報
    dt = payload.date
    weekday = dt.weekday()  # 0=月, 6=日
    weeknum = dt.isocalendar()[1]
    is_holiday = int(jpholiday.is_holiday(dt) or weekday >= 5)

    # 天気はダミー（晴れのみ1, 他0）
    weather_keys = [k for k in features_list if k.startswith("天気_")]
    weather = {k: (1 if k == "天気_晴れ" else 0) for k in weather_keys}

    # 特徴量ベクトル生成
    feats = {}
    for k in features_list:
        if k == "曜日": feats[k] = float(weekday)
        elif k == "週番号": feats[k] = float(weeknum)
        elif k == "祝日フラグ": feats[k] = float(is_holiday)
        elif k == "予約件数": feats[k] = float(payload.yoyaku_count)
        elif k == "予約合計台数": feats[k] = float(payload.yoyaku_total)
        elif k == "固定客予約数": feats[k] = float(payload.fixed_customer_count)
        elif k == "上位得意先予約数": feats[k] = float(payload.top_customer_count)
        elif k in weather:
            feats[k] = float(weather[k])
        else:
            feats[k] = 0.0

    result = PreprocessFeaturesResult(date=dt.isoformat(), features=feats)
    return ApiResponse.success(code="PREPROCESS_OK", detail="特徴量生成完了", result=result)


# --- Workspace/Path detection --------------------------------------------
def _detect_workspace_root(max_up: int = 5) -> Path:
    """Detect a plausible workspace root by walking up to find known folders.

    Priorities:
    1) WORKSPACE_ROOT env
    2) Parent chain containing 'notebooks' or 'data' or 'vendor'
    3) Current working directory as a fallback
    """
    env_root = os.getenv("WORKSPACE_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    p = here
    for _ in range(max_up):
        if (p / "notebooks").exists() or (p / "data").exists() or (p / "vendor").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()


def _ensure_api_response_import(workspace_root: Path) -> None:
    """Try importing api_response; if fails, attempt to add candidate sys.path entries.

    Candidates:
    - NOTEBOOKS_DIR env
    - <workspace_root>/notebooks
    - <workspace_root>
    """
    try:
        # Try direct import first
        from api_response import ApiResponse  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    candidates: list[Path] = []
    env_nb = os.getenv("NOTEBOOKS_DIR")
    if env_nb:
        candidates.append(Path(env_nb).resolve())
    candidates.append((workspace_root / "notebooks").resolve())
    candidates.append((workspace_root / "works" / "notebooks").resolve())
    candidates.append(workspace_root.resolve())

    for d in candidates:
        pkg_dir = d / "api_response"
        if pkg_dir.exists() and (pkg_dir / "__init__.py").exists():
            if d.as_posix() not in sys.path:
                sys.path.insert(0, d.as_posix())
            try:
                from api_response import ApiResponse  # type: ignore  # noqa: F401
                return
            except Exception:
                continue

    # Final attempt failed: raise a clear error
    raise RuntimeError(
        "Failed to import 'api_response'. Set NOTEBOOKS_DIR env to the folder that "
        "contains the 'api_response' package, or ensure it is in PYTHONPATH."
    )


def _ensure_scripts_on_path(workspace_root: Path) -> None:
    """Ensure /scripts is importable so that new_model* packages can be imported.

    Adds both <ws>/scripts and <ws> (for 'scripts.' prefix) if needed.
    """
    # 'scripts' パッケージは <workspace_root>/scripts 配下が正とする。
    # そのため最初に <workspace_root> を追加し、次に <workspace_root>/scripts を追加する。
    # 互換のために <workspace_root>/works とその配下も後順位で追加。
    # 親ディレクトリ（/works）を最優先にし、次に /works/scripts を追加。
    # その後、互換のために workspace_root（/works/works）系も追加。
    candidates = [
        workspace_root.parent.resolve(),
        (workspace_root.parent / "scripts").resolve(),
        workspace_root.resolve(),
        (workspace_root / "scripts").resolve(),
    ]
    for d in candidates:
        pass
    # 追加順序を逆順にして、candidates 先頭のものが最優先で先頭に来るようにする
    for d in reversed(candidates):
        if d.exists():
            p = d.as_posix()
            if p not in sys.path:
                sys.path.insert(0, p)


# Initialize workspace and import contract
_WS_ROOT = _detect_workspace_root()
_ensure_api_response_import(_WS_ROOT)
_ensure_scripts_on_path(_WS_ROOT)
from api_response import ApiResponse  # type: ignore  # noqa: E402


# --- Config ---------------------------------------------------------------
APP_TITLE = "Prediction API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "予測モデルの推論API。\n\n"
    "主なエンドポイント:\n"
    "- GET /health: モデルロード状態の確認\n"
    "- POST /predict/with-features: 特徴量マップを明示して任意日付（未来含む）を予測\n\n"
    "Swagger UI は /docs、OpenAPI JSON は /openapi.json、ReDoc は /redoc から参照できます。"
)

def _default_model_path(workspace_root: Path) -> str:
    p = (workspace_root / "data" / "final_stage1_model_api.pkl").resolve()
    return p.as_posix()


MODEL_PATH = os.getenv("PREDICTOR_MODEL_PATH", _default_model_path(_WS_ROOT))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]


# --- Result/Request Models (OpenAPI schema) -------------------------------
class PredictResult(BaseModel):
    """Result payload returned by predict endpoints."""

    date: str
    per_item: Dict[str, float]
    total: float
    used_features: List[str]

    # Pydantic v2: スキーマ例
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

# --- Provider: Loads the predictor (Single Responsibility) ----------------
class PredictorProvider:
    """Provide a loaded predictor instance from disk.

    Hides persistence detail from the service layer.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self._model_path = model_path
        self._predictor = None  # lazy-loaded

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        """Load predictor if present and not yet loaded."""
        if self._predictor is not None:
            return
        if not os.path.exists(self._model_path):
            # Keep unloaded; service will report via ApiResponse
            return
    # Try normal pickle load first; if it fails due to class resolution,
    # fall back to a compatibility unpickler that remaps classes.
        try:
            with open(self._model_path, "rb") as f:
                self._predictor = pickle.load(f)
            return
        except Exception:
            pass

        # Compatibility: map '__main__.NewModel2Predictor' (or similar) to the
        # real class in our code. Try current feature_builder first, then legacy.
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):  # type: ignore[override]
                # Allow override via env var (full dotted path)
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
                        # Fallback: try a few common locations
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
            if getattr(pred, "allowed_features", None) in (None, [], ()):  # type: ignore[attr-defined]
                sel_path_candidates = [
                    (_WS_ROOT / "data" / "selected_features_final.txt").resolve(),
                    (_WS_ROOT.parent / "data" / "selected_features_final.txt").resolve(),
                ]
                for pth in sel_path_candidates:
                    try:
                        if pth.exists():
                            with open(pth.as_posix(), encoding="utf-8") as f:
                                pred.allowed_features = [ln.strip() for ln in f if ln.strip()]  # type: ignore[attr-defined]
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
                    (_WS_ROOT / "data" / "final_stage1_model_predictable.pkl").resolve(),
                    (_WS_ROOT / "data" / "final_stage1_model.pkl").resolve(),
                    (_WS_ROOT.parent / "data" / "final_stage1_model_predictable.pkl").resolve(),
                    (_WS_ROOT.parent / "data" / "final_stage1_model.pkl").resolve(),
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
                        df["伝票日付"] = _pd.to_datetime(df["伝票日付"])  # type: ignore
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

    def get(self):  # -> NewModel2Predictor at runtime
        return self._predictor


# --- Service: Business/use-case logic (Depends on Provider) ---------------
class PredictionService:
    """Application service orchestrating prediction use-cases."""

    def __init__(self, provider: PredictorProvider) -> None:
        self._provider = provider

    def health(self) -> ApiResponse[None]:
        if not self._provider.is_loaded:
            return ApiResponse.error(code="MODEL_NOT_LOADED", detail="モデル未ロード")
        return ApiResponse.success(code="OK", detail="healthy")

    # 未来日付含む予測は predict_with_features のみを提供

    def predict_with_features(self, target_date: _date, features: Dict[str, float]) -> ApiResponse[PredictResult]:
        """Predict for any date (including future) using provided feature map."""
        predictor = self._provider.get()
        if predictor is None:
            return ApiResponse.error(code="MODEL_NOT_LOADED", detail="モデル未ロード")
        try:
            # Ensure numeric feature values (coerce to float)
            feats = {str(k): float(v) for k, v in features.items()}
            out = predictor.predict_with_features(target_date.isoformat(), feats)
            result = PredictResult(**out)
            return ApiResponse.success(code="PREDICT_OK", detail="推論完了", result=result)
        except Exception as exc:
            return ApiResponse.error(code="PREDICT_FAILED", detail=f"予測に失敗: {exc}")


# --- Dependency Injection --------------------------------------------------
_provider_singleton = PredictorProvider()


def get_prediction_service() -> PredictionService:
    """Provide a service instance for request handling."""
    return PredictionService(provider=_provider_singleton)


# --- App/Routes (Controller Layer) ----------------------------------------
TAGS_METADATA = [
    {"name": "health", "description": "ヘルスチェック（モデルロード状態）"},
    {"name": "predict", "description": "予測エンドポイント群（特徴量直指定）"},
]

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    docs_url=os.getenv("DOCS_URL", "/docs"),
    redoc_url=os.getenv("REDOC_URL", "/redoc"),
    openapi_url=os.getenv("OPENAPI_URL", "/openapi.json"),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_load_predictor() -> None:
    """Load predictor once on startup (fail-soft if missing)."""
    _provider_singleton.load()


@app.get("/health", response_model=ApiResponse[None], summary="Health check", tags=["health"])
def health(service: PredictionService = Depends(get_prediction_service)):
    """Health endpoint returning unified ApiResponse schema."""
    return service.health()




@app.post(
    "/predict/with-features",
    response_model=ApiResponse[PredictResult],
    summary="Predict for a given date with explicit feature map (future ok)",
    tags=["predict"],
)
def predict_with_features(
    payload: PredictWithFeaturesRequest = Body(...),
    service: PredictionService = Depends(get_prediction_service),
):
    """Predict using user-provided feature map. Missing features are treated as zero by the model."""
    return service.predict_with_features(payload.date, payload.features)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "0"))),
    )

# 便利リンク（任意）
@app.get("/", summary="Service root")
def root():
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "redoc": "/redoc",
    }
