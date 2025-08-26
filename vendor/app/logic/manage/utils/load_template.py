from pathlib import Path
import pandas as pd
import os


def load_master_and_template(master_path: str | Path) -> pd.DataFrame:
    """
    テンプレート設定内の master_csv_path（相対パス）を受け取り、DataFrameとして返す。
    """
    base_dir = Path(os.getenv("BASE_DIR", "/work/app"))
    full_path = base_dir / Path(master_path)

    dtype_spec = {
        "大項目": str,
        "小項目1": str,
        "小項目2": str,
        "セル": str,
        "値": "object",
    }

    return pd.read_csv(full_path, encoding="utf-8-sig", dtype=dtype_spec)
