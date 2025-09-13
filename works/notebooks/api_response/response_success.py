from typing import Any, Optional

from .response_base import SuccessApiResponse


class TransportersSuccessResponse(SuccessApiResponse):
    """
    運搬業者リスト返却用の成功レスポンス
    """

    def __init__(
        self,
        *,
        code: str = "TRANSPORTERS_LIST_SUCCESS",
        detail: str = "運搬業者リストの取得に成功しました。",
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
