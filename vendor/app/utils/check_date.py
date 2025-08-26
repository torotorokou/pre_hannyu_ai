import utils.file_loader as load_uploaded_csv_files


def check_csv_manegement(dfs: dict, uploaded_files: dict) -> dict:
    """
    CSVファイルの読み込みと日付チェックを行う関数。
    """
    # アップロードされたCSVファイルを取得
    dfs = load_uploaded_csv_files(uploaded_files)
