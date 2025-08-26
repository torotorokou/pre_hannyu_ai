import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_jp_font():
    """
    日本語フォント(NotoSansCJK)をmatplotlibのデフォルトに設定する。
    """
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if not fm.os.path.exists(font_path):
        raise FileNotFoundError(f"フォントファイルが見つかりません: {font_path}")

    font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams["font.family"] = font_name
    print(f"✅ 日本語フォントを設定しました: {font_name}")
