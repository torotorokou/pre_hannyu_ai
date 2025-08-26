def resolve_dtype(dtype_str):
    import numpy as np

    dtype_map = {
        "int": int,
        "float": float,
        "str": str,
        "datetime": "datetime64[ns]",
        "datetime64": "datetime64[ns]",
        "np.int64": np.int64,
        "np.float64": np.float64,
    }

    return dtype_map.get(dtype_str, str)  # デフォルトはstrにしておく


# def get_expected_dtypes_by_template(template_key: str) -> dict:
#     """
#     YAMLなどから取得した expected_dtypes を Python 型に解決して返す
#     """
#     raw_config = load_yaml("expected_dtypes.yaml")  # ← 実際の読み込み方法に応じて調整
#     template_map = raw_config.get(template_key, {})

#     resolved_map = {}

#     for file_key, col_dtypes in template_map.items():
#         resolved_map[file_key] = {
#             col: resolve_dtype(dtype) for col, dtype in col_dtypes.items()
#         }

#     return resolved_map
