# 汎用的な処理関数
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
