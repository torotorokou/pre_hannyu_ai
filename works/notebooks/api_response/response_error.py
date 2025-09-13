from .response_base import ErrorApiResponse
from fastapi import UploadFile
from typing import Any


class NoFilesUploadedResponse(ErrorApiResponse):
    def __init__(self):
        super().__init__(
            code="no_files",
            detail="ファイルが1つもアップロードされていません。",
            hint="すべての必須CSVファイルをアップロードしてください。",
        )


class CSVReadErrorResponse(ErrorApiResponse):
    def __init__(self, file_name: str, exception: Exception):
        super().__init__(
            code="csv_read_error",
            hint=f"{file_name}ファイルが正しいCSV形式か確認してください。",
            detail=f"{file_name} のCSV読み込みに失敗しました: {str(exception)}",
        )


class ValidationFailedResponse(ErrorApiResponse):
    def __init__(self, file_name: str, errors: list[dict]):
        super().__init__(
            code="csv_validation_failed",
            detail=f"{file_name} のCSVバリデーションに失敗しました。",
            hint="ファイルのフォーマットと内容を確認してください。",
            result={"file": file_name, "errors": errors},
        )


class MissingColumnsResponse(ErrorApiResponse):
    def __init__(
        self,
        csv_type: str,
        missing_columns: list[str],
        file: UploadFile,
        df_columns: list[str],
    ):
        super().__init__(
            code="MISSING_COLUMNS",
            detail=f"{csv_type}ファイルの必須カラムが不足しています: {missing_columns}",
            result={
                csv_type: {
                    "filename": file.filename,
                    "columns": df_columns,
                    "status": "error",
                    "code": "MISSING_COLUMNS",
                    "detail": f"必須カラムが不足しています: {missing_columns}",
                }
            },
        )


class MissingDateFieldResponse(ErrorApiResponse):
    def __init__(self, missing_type: str, file: UploadFile, df_columns: list[str]):
        super().__init__(
            code="MISSING_DATE_FIELD",
            detail=f"{missing_type}ファイルに『伝票日付』カラムがありません。",
            result={
                missing_type: {
                    "filename": file.filename,
                    "columns": df_columns,
                    "status": "error",
                    "code": "MISSING_DATE_FIELD",
                    "detail": "『伝票日付』カラムがありません",
                }
            },
        )


class DateMismatchResponse(ErrorApiResponse):
    def __init__(self, dates_info: Any):
        super().__init__(
            code="DATE_MISMATCH",
            detail="伝票日付が一致しません。",
            hint="すべてのファイルの「伝票日付」が同じ範囲になっているか確認してください。",
            result={"dates": dates_info},
        )
