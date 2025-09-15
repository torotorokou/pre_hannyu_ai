"""
API基底レスポンス（Pydantic v2 対応）

目的:
- 既存の SuccessApiResponse / ErrorApiResponse の呼び出し互換を維持
- 中身は Pydantic モデルに委譲し、/docs で正しいスキーマを公開
- 新規エンドポイントは response_model=ApiResponse[...] を利用可能に

# 変更メモ（リファクタリング対象）
# - Pydantic v1 の `GenericModel` を廃止し、v2 推奨の `BaseModel` + `Generic[T]` に統一
# - `from pydantic.generics import GenericModel` を除去
# - v1/v2 互換のための分岐・フラグを削除してシンプル化
# - dict 変換は v2 の `model_dump()` を使用
"""

from typing import Any, Optional, Generic, TypeVar, Literal
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # Pydantic v2 を前提に統一


# --- 共通レスポンス契約（/docs に出す唯一のスキーマ） -------------------------
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):  # GenericModel -> BaseModel に置換
    """
    FastAPI に公開する共通レスポンスモデル（契約）。
    - /docs へ自動反映
    - response_model=ApiResponse[YourResultModel] として使用
    """

    status: Literal["success", "error"]
    code: str
    detail: str
    result: Optional[T] = None
    hint: Optional[str] = None

    @classmethod
    def success(
        cls,
        *,
        code: str,
        detail: str,
        result: Optional[T] = None,
        hint: Optional[str] = None,
    ) -> "ApiResponse[T]":
        return cls(status="success", code=code, detail=detail, result=result, hint=hint)

    @classmethod
    def error(
        cls,
        *,
        code: str,
        detail: str,
        result: Optional[Any] = None,
        hint: Optional[str] = None,
    ) -> "ApiResponse[Any]":
        return cls(status="error", code=code, detail=detail, result=result, hint=hint)


# 下記からは廃止予定
# --- Pydanticモデル -> dict の互換ヘルパ --------------------------------------
def _model_to_dict(model: BaseModel) -> dict:
    """
    Pydantic v2 で dict へ変換。
    """
    return model.model_dump()  # v2 標準


# --- 互換レイヤー: 既存クラス（内部で ApiResponse に委譲） -------------------
class BaseApiResponse:
    """
    【互換用】既存の使用箇所を壊さずに維持するための薄いラッパ。
    実体は ApiResponse に委譲し、to_json_response() もそのまま使える。
    直接のインスタンス化は非推奨（success/error のサブクラスを使用）。
    """

    # 各サブクラスで "success" or "error" を必ず上書き
    status: Literal["success", "error"]

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        result: Optional[Any] = None,
        hint: Optional[str] = None,
        status_code: int = 200,
    ):
        """
        互換コンストラクタ
        - 旧コードの引数はそのまま受ける
        - 内部で ApiResponse に委譲
        """
        # サブクラスで status がセットされている前提
        if self.status not in ("success", "error"):
            raise ValueError(
                "BaseApiResponse.status は 'success' か 'error' を指定してください。"
            )

        if self.status == "success":
            self.payload = ApiResponse.success(
                code=code, detail=detail, result=result, hint=hint
            )
        else:
            self.payload = ApiResponse.error(
                code=code, detail=detail, result=result, hint=hint
            )

        self.status_code = status_code

    def to_json_response(self) -> JSONResponse:
        """
        旧コード互換: JSONResponse を返す。
        ※ 新規コードでは JSONResponse を直接返さず、
           FastAPI にモデル/辞書を返すことを推奨。
        """
        content = _model_to_dict(self.payload)
        return JSONResponse(status_code=self.status_code, content=content)


class SuccessApiResponse(BaseApiResponse):
    """
    【互換用】成功レスポンスクラス
    旧来の呼び出しを維持しつつ、内部は ApiResponse に委譲する。
    """

    status: Literal["success", "error"] = "success"

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        result: Optional[Any] = None,
        hint: Optional[str] = None,
        status_code: int = 200,
    ):
        super().__init__(
            code=code,
            detail=detail,
            result=result,
            hint=hint,
            status_code=status_code,
        )


class ErrorApiResponse(BaseApiResponse):
    """
    【互換用】エラーレスポンスクラス
    旧来の呼び出しを維持しつつ、内部は ApiResponse に委譲する。
    """

    status: Literal["success", "error"] = "error"

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        result: Optional[Any] = None,
        hint: Optional[str] = None,
        status_code: int = 422,
    ):
        super().__init__(
            code=code,
            detail=detail,
            result=result,
            hint=hint,
            status_code=status_code,
        )


# --- 旧名: ApiResponseModel を残したい場合のエイリアス（任意） ---------------
# 旧コードに ApiResponseModel が出てくる場合の下位互換。
# 新規コードでは ApiResponse[...] を直接使うことを推奨。
ApiResponseModel = ApiResponse  # 下位互換用の別名
