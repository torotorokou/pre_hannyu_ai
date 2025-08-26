import numpy as np
import json


def load_expected_dtypes(config):
    path = config["main_paths"]["expected_dtypes"]
    with open(path, encoding="utf-8") as f:
        dtype_str_map = json.load(f)

    # 型マップを拡張
    type_map = {
        "float": float,
        "float64": np.float64,
        "int": int,
        "int64": np.int64,
        "str": str,
        "bool": bool,
        "datetime": "datetime64[ns]",  # pandasで読み取り可能な形式
    }

    return {k: type_map.get(v, str) for k, v in dtype_str_map.items()}
