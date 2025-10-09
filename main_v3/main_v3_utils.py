"""
概要:
    v3 パイプラインで共通的に用いる軽量ユーティリティ関数群をまとめたモジュール。
    - 例: 指定した年月範囲から "YYYYMM" 形式の月キー配列を生成する等

注意:
    - 本モジュールは外部依存が少なく、他モジュールからの再利用を想定
"""
from datetime import datetime

def get_available_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """
    概要:
        開始年月から終了年月までを両端含みで走査し、"YYYYMM" 形式の文字列リストを生成して返す。

    入力:
        - start_year (int): 開始年（例: 2014）
        - start_month (int): 開始月（1-12）
        - end_year (int): 終了年（例: 2023）
        - end_month (int): 終了月（1-12）

    処理:
        - datetime(start_year, start_month, 1) から end_year/end_month の月初まで、1か月刻みで進める
        - 各月の年月を current.strftime("%Y%m") で "YYYYMM" に整形しリストへ追加
        - 月繰上がりは 12月→翌年1月 にロールオーバー

    出力:
        - months (List[str]): "YYYYMM" 形式の文字列リスト（開始・終了を含む昇順）
    """
    months = []
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    while current <= end:
        months.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return months

__all__ = ["get_available_months"]
