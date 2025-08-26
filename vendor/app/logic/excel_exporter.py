# Excelへの出力処理
import pandas as pd
from io import BytesIO


def export_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="データ")
    return output.getvalue()
