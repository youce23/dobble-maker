import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def imread_japanese(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imreadの非ASCIIパス対応版

    - filenameがASCIIのみで構成されていればcv2.imread()をそのまま実行
    - 非ASCIIが含まれていれば、tempfileモジュールの仕様に従って定まる一時フォルダに
    ファイルをコピーしてからcv2.imread()で読み込み
        - 一時フォルダはASCIIのみで構成されているという前提
        - 一方でWindowsのユーザ名に日本語が含まれる場合など、上記が成り立たない可能性がある
    - 一時フォルダにも非ASCIIが含まれる場合は、カレントの直下にtempファイルを作成し、処理が終了したら削除
    """
    # ASCII文字のみのファイル名の場合、通常のcv2.imreadを使用
    if filename.isascii():
        return cv2.imread(filename, flags=flags)

    # 非ASCIIファイルの場合の処理、一時フォルダでの処理にトライ
    with tempfile.TemporaryDirectory() as temp_dir:
        # 一時ディレクトリがASCII文字のみの場合はその中にファイルをコピーしてから読み込み
        if temp_dir.isascii():
            temp_file = Path(temp_dir) / ("temp" + Path(filename).suffix)
            shutil.copyfile(filename, temp_file)
            return cv2.imread(str(temp_file), flags=flags)

    # 一時ディレクトリも非ASCIIを含む場合、カレントディレクトリ直下にコピー
    temp_file = "temp" + Path(filename).suffix
    shutil.copyfile(filename, temp_file)
    img = cv2.imread(temp_file, flags=flags)
    os.remove(temp_file)  # 一時ファイルを削除
    return img


def imwrite_japanese(filename: str, img: np.ndarray, params: Sequence[int] = []) -> bool:
    """cv2.imwriteの非ASCIIパス対応版

    処理フローはimread_japanese()と同様
    """
    if filename.isascii():
        return cv2.imwrite(filename, img, params=params)

    with tempfile.TemporaryDirectory() as temp_dir:
        if temp_dir.isascii():
            temp_file = Path(temp_dir) / ("temp" + Path(filename).suffix)
            ret = cv2.imwrite(str(temp_file), img, params=params)
            if ret:
                shutil.move(temp_file, filename)
            return ret
    temp_file = "temp" + Path(filename).suffix
    ret = cv2.imwrite(str(temp_file), img, params=params)
    shutil.move(temp_file, filename)
    return ret
