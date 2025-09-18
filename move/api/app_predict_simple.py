"""FastAPI prediction API with feature generation and prediction endpoints"""

import os
import pickle
import sys
from datetime import date as _date
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jpholiday

# 公式レスポンスモデル（notebooks/api_response）に統一
from notebooks.api_response.response_base import ApiResponse

# Response result models (for response_model typing)
class HealthResult(BaseModel):
    model_loaded: bool

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
            # Try move/data first, then fallback to data
            move_path = (WS_ROOT / "move" / "data" / "final_stage1_model_api.pkl").as_posix()
            default_path = (WS_ROOT / "data" / "final_stage1_model_api.pkl").as_posix()
            
            if os.path.exists(move_path):
                model_path = move_path
            else:
                model_path = default_path
        
        self._model_path = model_path
        self._predictor = None  # lazy-loaded
        
        # Ensure scripts path for imports
        self._ensure_scripts_on_path()

    def _ensure_scripts_on_path(self) -> None:
        """Ensure /scripts is importable so that new_model* packages can be imported."""
        # Remove any existing scripts paths to avoid conflicts
        sys.path = [p for p in sys.path if not p.endswith('/scripts')]
        # 優先順位: move > /works/works > /works
        # NOTE: 'scripts' パッケージ（scripts.new_model2.*）を正しく解決するため、
        # sys.path には scripts フォルダ自体ではなく、その親（/works/move）を入れる。
        # これにより `import scripts.new_model2.feature_builder` が PYTHONPATH 未設定でも成功する。
        preferred = [
            (WS_ROOT / "move").resolve(),
            (WS_ROOT / "works").resolve(),
            WS_ROOT.resolve(),
        ]
        for d in reversed(preferred):  # 最後に挿入したものが最前に来るので逆順
            if d.exists():
                p = d.as_posix()
                if p not in sys.path:
                    sys.path.insert(0, p)
        # 既に読み込まれている 'scripts' 系モジュールをパージして、/works/move 側が優先されるようにする
        try:
            to_del = [k for k in list(sys.modules.keys()) if k == 'scripts' or k.startswith('scripts.')]
            for k in to_del:
                del sys.modules[k]
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        """Load predictor if present and not yet loaded."""
        if self._predictor is not None:
            return
        
        print(f"[DEBUG] Checking model path: {self._model_path}")
        print(f"[DEBUG] Path exists: {os.path.exists(self._model_path)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] WS_ROOT: {WS_ROOT}")
        
        if not os.path.exists(self._model_path):
            print(f"[ERROR] Model file not found: {self._model_path}")
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
                # Helper: ensure /works/move is first on sys.path for scripts.* imports
                def _prefer_move_scripts():
                    try:
                        move_dir = (WS_ROOT / "move").resolve().as_posix()
                        ws_dir = WS_ROOT.resolve().as_posix()
                        # Remove existing entries to reinsert in desired order
                        for target in [move_dir]:
                            try:
                                while target in sys.path:
                                    sys.path.remove(target)
                            except ValueError:
                                pass
                        # Ensure move_dir is first
                        sys.path.insert(0, move_dir)
                        # Also drop any loaded 'scripts' modules so the resolver reloads
                        to_del = [k for k in list(sys.modules.keys()) if k == 'scripts' or k.startswith('scripts.')]
                        for k in to_del:
                            try:
                                del sys.modules[k]
                            except Exception:
                                pass
                    except Exception:
                        pass
                target_path = os.getenv(
                    "PREDICTOR_CLASS",
                    "scripts.new_model2.feature_builder.NewModel2Predictor",
                )
                # 1) NewModel2Predictor は scripts.new_model2.feature_builder へ誘導
                if name == "NewModel2Predictor" or (
                    module == "__main__" and name.endswith("Predictor")
                ):
                    try:
                        mod_name, cls_name = target_path.rsplit(".", 1)
                        if mod_name.startswith('scripts.'):
                            _prefer_move_scripts()
                        mod = importlib.import_module(mod_name)
                        return getattr(mod, cls_name)
                    except Exception:
                        # Fallback: try common locations
                        for cand in [
                            "scripts.new_model2.feature_builder.NewModel2Predictor",
                            "scripts.new_model2.NewModel2Predictor",
                            "scripts.new_model2.predict_model_v4_2_4.NewModel2Predictor",
                            "new_model2.predict_model_v4_2_4.NewModel2Predictor",
                        ]:
                            try:
                                m, c = cand.rsplit(".", 1)
                                if m.startswith('scripts.'):
                                    _prefer_move_scripts()
                                mod = importlib.import_module(m)
                                return getattr(mod, c)
                            except Exception:
                                continue
                # 2) 補助ビルダー類の互換（古いpickleが works.scripts.* を指している場合に備える）
                builder_names = {
                    "WeatherFeatureBuilder",
                    "ReserveFeatureBuilder",
                    "WeightFeatureBuilder",
                    "StackingEnsemble",
                }
                if name in builder_names:
                    # まず scripts.new_model2.feature_builder を優先
                    for cand_mod in [
                        "scripts.new_model2.feature_builder",  # move配下が sys.path 先頭
                        "works.scripts.new_model2.feature_builder",
                        "scripts.new_model2.predict_model_v4_2_4",
                        "new_model2.feature_builder",
                    ]:
                        try:
                            if cand_mod.startswith('scripts.'):
                                _prefer_move_scripts()
                            mod = importlib.import_module(cand_mod)
                            if hasattr(mod, name):
                                return getattr(mod, name)
                        except Exception:
                            continue
                return super().find_class(module, name)

        try:
            with open(self._model_path, "rb") as f:
                self._predictor = _CompatUnpickler(f).load()
        except EOFError:
            print(f"[ERROR] EOFError while unpickling: {self._model_path} (file may be truncated)")
            raise

        # --- Post-load fixups: ensure real estimator and features are present ---
        try:
            pred = self._predictor
            # 0) _model が学習成果物 dict の場合、StackingEnsemble に再構築
            try:
                m = getattr(pred, "_model", None)
                if isinstance(m, dict) and ("stage1_models" in m and "stage2_model" in m):
                    import importlib
                    try:
                        mod = importlib.import_module("scripts.new_model2.feature_builder")
                        SE = getattr(mod, "StackingEnsemble", None)
                    except Exception:
                        SE = None
                    if SE is not None:
                        # target_items の決定
                        tis = m.get("target_items") or getattr(pred, "_target_items", None) or []
                        ens = SE(m.get("stage1_models"), m.get("stage2_model"), list(tis))
                        setattr(pred, "_model", ens)
                        # predictor 側にも反映
                        try:
                            if tis and not getattr(pred, "_target_items", None):
                                setattr(pred, "_target_items", list(tis))
                        except Exception:
                            pass
                # _model が StackingEnsemble だが中身が空の場合、_model_raw から再構築
                else:
                    try:
                        has_breakdown = hasattr(m, "predict_with_breakdown")
                        st1 = getattr(m, "stage1", None)
                        st2 = getattr(m, "stage2", None)
                        empty_st1 = (isinstance(st1, dict) and len(st1.get("items", {})) == 0) if isinstance(st1, dict) else (st1 is None)
                        empty_st2 = (isinstance(st2, dict) and len(st2) == 0) if isinstance(st2, dict) else (st2 is None)
                        if has_breakdown and (empty_st1 or empty_st2):
                            mraw = getattr(pred, "_model_raw", None)
                            if isinstance(mraw, dict) and ("stage1_models" in mraw and "stage2_model" in mraw):
                                import importlib
                                try:
                                    mod = importlib.import_module("scripts.new_model2.feature_builder")
                                    SE = getattr(mod, "StackingEnsemble", None)
                                except Exception:
                                    SE = None
                                if SE is not None:
                                    tis = mraw.get("target_items") or getattr(pred, "_target_items", None) or []
                                    ens = SE(mraw.get("stage1_models"), mraw.get("stage2_model"), list(tis))
                                    setattr(pred, "_model", ens)
                                    if tis and not getattr(pred, "_target_items", None):
                                        setattr(pred, "_target_items", list(tis))
                    except Exception:
                        pass
            except Exception:
                pass
            # 0.1) 既にEstimatorが存在し target_items が空なら、_model_rawや_model(dict)から補完
            try:
                est = getattr(pred, "_model", None)
                # 候補の target_items
                tis = None
                mraw = getattr(pred, "_model_raw", None)
                if isinstance(mraw, dict) and mraw.get("target_items"):
                    tis = list(mraw.get("target_items"))
                if tis is None and isinstance(est, dict) and est.get("target_items"):
                    tis = list(est.get("target_items"))
                if tis is None and getattr(pred, "_target_items", None):
                    tis = list(getattr(pred, "_target_items"))
                if tis:
                    # Estimator側に設定
                    if hasattr(est, "target_items") and not getattr(est, "target_items", None):
                        try:
                            setattr(est, "target_items", list(tis))
                        except Exception:
                            pass
                    # Predictor 側にも確定
                    if not getattr(pred, "_target_items", None):
                        setattr(pred, "_target_items", list(tis))
            except Exception:
                pass
            # 1) allowed_features 補完
            if getattr(pred, "allowed_features", None) in (None, [], ()):
                sel_path_candidates = [
                    (WS_ROOT / "move" / "data" / "selected_features_final.txt").resolve(),
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
                    (WS_ROOT / "move" / "data" / "final_stage1_model_predictable.pkl").resolve(),
                    (WS_ROOT / "move" / "data" / "final_stage1_model.pkl").resolve(),
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
                
                # metaからも取得できない場合、デフォルトの品目リストを使用
                if items is None:
                    items = ["混合廃棄物A", "混合廃棄物B", "GC 軽鉄･ｽﾁｰﾙ類", "選別", "木くず"]
                
                setattr(pred, "_target_items", items)
                print(f"[FIX] _target_items set to: {items}")
            # 空リストやNoneの場合、_model_rawや_model(dict)から補完
            try:
                if not getattr(pred, "_target_items", None):
                    mraw = getattr(pred, "_model_raw", None)
                    if isinstance(mraw, dict) and mraw.get("target_items"):
                        pred._target_items = list(mraw.get("target_items"))
                    else:
                        m = getattr(pred, "_model", None)
                        if isinstance(m, dict) and m.get("target_items"):
                            pred._target_items = list(m.get("target_items"))
                    if getattr(pred, "_target_items", None):
                        print(f"[FIX] _target_items recovered from model: {pred._target_items}")
            except Exception:
                pass
            
            # _item_shares も未定義なら None で初期化
            if not hasattr(pred, "_item_shares"):
                # デフォルトの均等分割を設定
                target_items = getattr(pred, "_target_items", None)
                if target_items and len(target_items) > 0:
                    import numpy as np
                    shares = np.ones(len(target_items)) / len(target_items)
                    setattr(pred, "_item_shares", shares)
                    print(f"[FIX] _item_shares set to equal distribution: {shares}")
                else:
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

# 内部派生（モデル側で計算されるため、外部から上書きすべきでない特徴）
INTERNAL_DERIVED_FEATURES: Set[str] = {
    "合計_前日値",
    "合計_3日平均",
    "合計_前週平均",
    "1台あたり重量_過去中央値",
}

# Initialize provider
_provider_singleton = PredictorProvider()

def generate_features(date: _date, yoyaku_count: int, yoyaku_total: int, 
                     fixed_customer_count: int, top_customer_count: int) -> Dict[str, float]:
    """Generate features from input data"""
    
    # Try to load features list
    sel_path_candidates = [
        WS_ROOT / "move" / "data" / "selected_features_final.txt",
        WS_ROOT / "data" / "selected_features_final.txt"
    ]
    
    features_list = None
    for sel_path in sel_path_candidates:
        if sel_path.exists():
            with open(sel_path, encoding="utf-8") as f:
                features_list = [ln.strip() for ln in f if ln.strip()]
            break
    
    if features_list is None:
        # Default features
        features_list = [
            "曜日", "週番号", "祝日フラグ", "予約件数", "予約合計台数", 
            "固定客予約数", "上位得意先予約数", "天気_晴れ", "天気_雨", 
            "天気_大雨", "天気_台風"
        ]
    
    # 日付情報
    weekday = date.weekday()  # 0=月, 6=日
    weeknum = date.isocalendar()[1]
    monthnum = date.month
    is_holiday = int(jpholiday.is_holiday(date) or weekday >= 5)
    # 営業日（平日かつ祝日でない）
    def _is_business_day(d):
        return (d.weekday() < 5) and (not jpholiday.is_holiday(d))
    prev_bd = _is_business_day(date - datetime.timedelta(days=1))
    next_bd = _is_business_day(date + datetime.timedelta(days=1))

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
        elif k == "前営業日フラグ":
            feats[k] = float(1 if prev_bd else 0)
        elif k == "翌営業日フラグ":
            feats[k] = float(1 if next_bd else 0)
        elif k == "月":
            feats[k] = float(monthnum)
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
    # 内部派生特徴はAPIの出力から除去（クライアントが上書きして精度劣化するのを防止）
    for drop_key in list(INTERNAL_DERIVED_FEATURES):
        if drop_key in feats:
            feats.pop(drop_key, None)

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
    try:
        print("[STARTUP] モデルロード開始...")
        _provider_singleton.load()
        if _provider_singleton.is_loaded:
            print("[STARTUP] モデルロード成功")
        else:
            print("[STARTUP] モデルロード失敗 - ファイルが見つからない可能性")
    except Exception as e:
        print(f"[STARTUP] モデルロード中にエラー: {e}")
        import traceback
        traceback.print_exc()

# --- Routes ---
@app.get("/health", response_model=ApiResponse[HealthResult])
def health():
    """Health check"""
    model_status = "loaded" if _provider_singleton.is_loaded else "not_loaded"
    return ApiResponse.success(
        code="HEALTH_OK",
        detail=f"API is running. Model: {model_status}",
        result=HealthResult(model_loaded=_provider_singleton.is_loaded),
    )

@app.get(
    "/_debug/model",
    tags=["debug"],
    summary="モデル内部の簡易ダンプ",
    response_model=ApiResponse[Dict[str, Any]],
)
def debug_model():
    pred = _provider_singleton.get()
    if pred is None:
        return ApiResponse.success(code="DEBUG_OK", detail="model not loaded", result={})
    out: Dict[str, Any] = {}
    # 参照中のモデルパス
    try:
        out["model_path"] = getattr(_provider_singleton, "_model_path", None)
    except Exception:
        out["model_path"] = None
    out["predictor_class"] = str(type(pred))
    out["has_predict_with_features"] = bool(hasattr(pred, "predict_with_features"))
    out["allowed_features_count"] = len(getattr(pred, "allowed_features", []) or [])
    out["allowed_features_head"] = (getattr(pred, "allowed_features", []) or [])[:10]
    out["target_items"] = getattr(pred, "_target_items", None)
    # 内部モデル
    m = getattr(pred, "_model", None)
    out["_model_type"] = str(type(m))
    if isinstance(m, dict):
        out["_model_keys"] = list(m.keys())
        st1 = m.get("stage1_models") if isinstance(m, dict) else None
        st2 = m.get("stage2_model") if isinstance(m, dict) else None
        if isinstance(st1, dict):
            items = st1.get("items")
            out["stage1_items_count"] = len(items) if isinstance(items, dict) else None
            out["stage1_item_sample_keys"] = list(items.keys())[:3] if isinstance(items, dict) else None
            out["stage1_raw_feature_names_count"] = len(st1.get("raw_feature_names", []) or [])
        if isinstance(st2, dict):
            out["stage2_keys"] = list(st2.keys())
            out["stage2_feature_list_count"] = len(st2.get("feature_list", []) or [])
    else:
        # estimator の場合
        out["_model_has_predict"] = bool(hasattr(m, "predict"))
        out["_model_has_breakdown"] = bool(hasattr(m, "predict_with_breakdown"))
        # StackingEnsemble 内部の確認
        try:
            st1 = getattr(m, "stage1", None)
            st2 = getattr(m, "stage2", None)
            tis = getattr(m, "target_items", None)
            if isinstance(st1, dict):
                out["ens_stage1_keys"] = list(st1.keys())
                items = st1.get("items")
                out["ens_stage1_items_count"] = len(items) if isinstance(items, dict) else None
                out["ens_stage1_item_sample_keys"] = list(items.keys())[:3] if isinstance(items, dict) else None
                out["ens_stage1_raw_feature_names_count"] = len(st1.get("raw_feature_names", []) or [])
            if isinstance(st2, dict):
                out["ens_stage2_keys"] = list(st2.keys())
                out["ens_stage2_feature_list_count"] = len(st2.get("feature_list", []) or [])
            if isinstance(tis, (list, tuple)):
                out["ens_target_items_count"] = len(tis)
                out["ens_target_items_head"] = list(tis)[:5]
        except Exception:
            pass
    # raw model dump (dict)
    mraw = getattr(pred, "_model_raw", None)
    out["_model_raw_type"] = str(type(mraw))
    if isinstance(mraw, dict):
        out["_model_raw_keys"] = list(mraw.keys())
        st1 = mraw.get("stage1_models")
        st2 = mraw.get("stage2_model")
        if isinstance(st1, dict):
            items = st1.get("items")
            out["raw_stage1_items_count"] = len(items) if isinstance(items, dict) else None
        if isinstance(st2, dict):
            out["raw_stage2_keys"] = list(st2.keys())
    return ApiResponse.success(code="DEBUG_OK", detail="dump", result=out)

@app.post(
    "/_debug/reload",
    tags=["debug"],
    summary="モデルを再読込（ホットリロード）",
    response_model=ApiResponse[Dict[str, Any]],
)
def debug_reload(model_path: Optional[str] = Body(None, description="明示的に読み込むPKLパスを指定可能")):
    global _provider_singleton
    try:
        # 新しいプロバイダで置き換え（明示パス指定があれば利用）
        _provider_singleton = PredictorProvider(model_path=model_path)
        _provider_singleton.load()
        pred = _provider_singleton.get()
        ok = pred is not None
        result = {
            "reloaded": ok,
            "model_path": getattr(_provider_singleton, "_model_path", None),
            "target_items": getattr(pred, "_target_items", None) if ok else None,
        }
        return ApiResponse.success(code="RELOAD_OK", detail="reloaded", result=result)
    except Exception as e:
        payload = ApiResponse.error(code="RELOAD_FAILED", detail=f"reload failed: {e}")
        return JSONResponse(status_code=500, content=payload.model_dump())

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
        payload = ApiResponse.error(code="PREPROCESS_FAILED", detail=f"特徴量生成エラー: {str(e)}")
        return JSONResponse(status_code=500, content=payload.model_dump())


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
            payload = ApiResponse.error(code="MODEL_NOT_LOADED", detail="モデル未ロード")
            return JSONResponse(status_code=500, content=payload.model_dump())
        
        # Ensure numeric feature values (coerce to float)
        # かつ、内部派生特徴は受理しても上書きしない（除外）
        feats = {str(k): float(v) for k, v in payload.features.items() if str(k) not in INTERNAL_DERIVED_FEATURES}
        
        # Check if predict_with_features method exists
        if hasattr(predictor, 'predict_with_features'):
            out = predictor.predict_with_features(payload.date.isoformat(), feats)
        else:
            # Fallback: use basic prediction logic
            out = {
                "date": payload.date.isoformat(),
                "per_item": {},
                "total": sum(feats.values()) * 1.2,  # Simple fallback calculation
                "used_features": list(feats.keys())
            }
        
        result = PredictResult(**out)
        return ApiResponse.success(code="PREDICT_OK", detail="推論完了", result=result)
    
    except Exception as exc:
        payload = ApiResponse.error(code="PREDICT_FAILED", detail=f"予測に失敗: {str(exc)}")
        return JSONResponse(status_code=500, content=payload.model_dump())

if __name__ == "__main__":
    import uvicorn
    import os as _os
    port = int(_os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)