def check_missing_files(
    validated_files: dict, required_keys: list[str]
) -> tuple[bool, list[str]]:
    """
    必要ファイルのうち、アップロードされていないものをチェックする

    Returns:
        all_uploaded (bool): すべてのファイルが揃っているかどうか
        missing_keys (list): 欠損しているキーの一覧
    """
    missing_keys = [k for k in required_keys if validated_files.get(k) is None]
    all_uploaded = len(missing_keys) == 0
    return all_uploaded, missing_keys
