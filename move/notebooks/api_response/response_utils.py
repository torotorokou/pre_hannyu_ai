from fastapi.responses import JSONResponse
from typing import Any, Optional
from .response_base import BaseApiResponse, SuccessApiResponse, ErrorApiResponse


# 旧形式（関数型）も互換用に残しておく
def api_response(
    *,
    status_code: int,
    status_str: str,
    code: str,
    detail: str,
    result: Optional[Any] = None,
    hint: Optional[str] = None,
) -> JSONResponse:
    """非推奨：クラス版に移行推奨"""
    content = {
        "status": status_str,
        "code": code,
        "detail": detail,
    }
    if result is not None:
        content["result"] = result
    if hint is not None:
        content["hint"] = hint
    return JSONResponse(status_code=status_code, content=content)
