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
from __future__ import annotations

import os
import pickle
import sys
from datetime import date as _date
from pathlib import Path
from typing import Dict, List

from fastapi import Body, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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


# Initialize workspace and import contract
_WS_ROOT = _detect_workspace_root()
_ensure_api_response_import(_WS_ROOT)
from api_response import ApiResponse  # type: ignore  # noqa: E402


# --- Config ---------------------------------------------------------------
APP_TITLE = "Prediction API"
APP_VERSION = "1.0.0"

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


class PredictRequest(BaseModel):
    """Request body for /predict."""

    date: _date = Field(..., description="Target date in YYYY-MM-DD")


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
        with open(self._model_path, "rb") as f:
            self._predictor = pickle.load(f)

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

    def predict(self, target_date: _date) -> ApiResponse[PredictResult]:
        predictor = self._provider.get()
        if predictor is None:
            return ApiResponse.error(
                code="MODEL_NOT_LOADED",
                detail="予測器がロードされていません。",
                hint=f"存在するか確認: {MODEL_PATH}",
            )
        try:
            out = predictor.predict(target_date.isoformat())
            result = PredictResult(**out)
            return ApiResponse.success(
                code="PREDICT_OK", detail="推論完了", result=result
            )
        except ValueError as exc:
            return ApiResponse.error(
                code="PREDICT_OUT_OF_RANGE",
                detail=str(exc),
                hint="学習期間内の日付を指定してください。",
            )
        except Exception as exc:  # pragma: no cover
            return ApiResponse.error(code="PREDICT_FAILED", detail=f"予測に失敗: {exc}")

    def predict_last(self) -> ApiResponse[PredictResult]:
        predictor = self._provider.get()
        if predictor is None:
            return ApiResponse.error(code="MODEL_NOT_LOADED", detail="モデル未ロード")
        try:
            out = predictor.predict_last()
            result = PredictResult(**out)
            return ApiResponse.success(
                code="PREDICT_OK", detail="推論完了", result=result
            )
        except Exception as exc:  # pragma: no cover
            return ApiResponse.error(code="PREDICT_FAILED", detail=str(exc))


# --- Dependency Injection --------------------------------------------------
_provider_singleton = PredictorProvider()


def get_prediction_service() -> PredictionService:
    """Provide a service instance for request handling."""
    return PredictionService(provider=_provider_singleton)


# --- App/Routes (Controller Layer) ----------------------------------------
app = FastAPI(title=APP_TITLE, version=APP_VERSION)
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


@app.get("/health", response_model=ApiResponse[None], summary="Health check")
def health(service: PredictionService = Depends(get_prediction_service)):
    """Health endpoint returning unified ApiResponse schema."""
    return service.health()


@app.post(
    "/predict",
    response_model=ApiResponse[PredictResult],
    summary="Predict for a given date",
)
def predict(
    payload: PredictRequest = Body(...),
    service: PredictionService = Depends(get_prediction_service),
):
    """Predict for the provided date."""
    return service.predict(payload.date)


@app.get(
    "/predict/last",
    response_model=ApiResponse[PredictResult],
    summary="Predict for the latest available date",
)
def predict_last(service: PredictionService = Depends(get_prediction_service)):
    """Predict for the latest date available in the model."""
    return service.predict_last()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "0"))),
    )
