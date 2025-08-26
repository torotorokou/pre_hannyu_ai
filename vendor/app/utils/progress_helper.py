import time


def update_progress(progress_bar, percent: int, message: str, delay: float = 0.3):
    """
    プログレスバーを更新し、指定時間スリープする

    Args:
        progress_bar (st.progress): Streamlitのprogressオブジェクト
        percent (int): 進捗の割合（0〜100）
        message (str): 表示するメッセージ
        delay (float): スリープ時間（秒）
    """
    progress_bar.progress(percent, message)
    time.sleep(delay)
