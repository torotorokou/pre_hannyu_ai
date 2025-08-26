from io import BytesIO
import pandas as pd


def convert_df_to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    """
    DataFrameをExcel形式のBytesIOに変換

    - 中項目のNaNは空白に
    - 游ゴシックフォント
    - 単価は小数点2桁表示
    - 全列同じ幅に揃える
    - 罫線なし
    """
    output = BytesIO()

    # --- NaNや文字列'nan'などを空白に変換（中項目のみ）
    if "中項目" in df.columns:
        df = df.copy()
        df["中項目"] = (
            df["中項目"]
            .replace(["nan", "NaN", "None"], "")  # ← 文字列としてのnanも空白に
            .fillna("")  # ← 本物のNaNも空白に
            .astype(str)  # ← 念のためすべて文字列化
        )

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=1, header=False)

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        # --- フォント定義（游ゴシック、罫線なし）
        header_format = workbook.add_format(
            {"font_name": "游ゴシック", "bold": True, "bg_color": "#F2F2F2"}
        )

        cell_format = workbook.add_format({"font_name": "游ゴシック"})

        unit_price_format = workbook.add_format(
            {"font_name": "游ゴシック", "num_format": "#,##0.00"}
        )

        # --- ヘッダー書き込み
        for col_num, column_name in enumerate(df.columns):
            worksheet.write(0, col_num, column_name, header_format)

        # --- データ書き込み（単価だけフォーマットを分ける）
        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
                col_name = df.columns[col_num]
                value = df.iat[row_num, col_num]

                if col_name == "単価":
                    worksheet.write(row_num + 1, col_num, value, unit_price_format)
                else:
                    worksheet.write(row_num + 1, col_num, value, cell_format)

        # --- 列幅を個別に指定（列名 → 幅）
        column_widths = {
            "大項目": 15,
            "中項目": 10,
            "合計正味重量": 10,
            "合計金額": 10,
            "単価": 7,
            "台数": 7,
        }

        for i, col_name in enumerate(df.columns):
            width = column_widths.get(col_name, 20)  # 未定義なら幅20に
            worksheet.set_column(i, i, width)

    output.seek(0)
    return output
