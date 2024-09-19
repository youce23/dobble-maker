import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def cv2_putText(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: str,
    font_scale: None | int,
    color: str | tuple,
    anchor: str,
    *,
    text_w: None | int = None,
    text_h: None | int = None,
    align: Literal["left", "center", "right"] = "left",
    font_index: int = 0,
    stroke_width: int = 0,
    stroke_fill: None | tuple = None,
):
    """日本語テキストの描画

    参考:
        OpenCVで日本語フォントを描画する を関数化する を最新にする,
        https://qiita.com/mo256man/items/f07bffcf1cfedf0e42e0

    Args:
        img: 対象画像 (上書きされる)
        text: 描画テキスト
        org: 描画位置 (x, y), 意味はanchorにより変わる
        font: フォントファイルパス (*.ttf | *.ttc)
        font_scale:
            フォントサイズ,
            Noneの場合はtext_wあるいはtext_hの指定が必須
        color: 描画色
        anchor (str):
            2文字で指定する, 1文字目はx座標について, 2文字目はy座標について以下のルールで指定する
            1文字目: "l", "m", "r" のいずれかを指定. それぞれテキストボックスの 左端, 中心, 右端 を意味する
            2文字目: "t", "m", "b" のいずれかを指定. それぞれテキストボックスの 上端, 中心, 下端 を意味する
        text_w:
            描画するテキスト幅 [pix]
            このサイズ以下になるギリギリのサイズで描画する
            (font_scaleがNoneの場合のみ有効)
        text_h:
            描画するテキスト高さ [pix]
            このサイズ以下になるギリギリのサイズで描画する
            (font_scaleがNoneの場合のみ有効)
        align:
            text に複数行(改行文字"\n"を含む文字列)を指定した場合の描画方法
            "left": 左揃え (default)
            "center": 中央揃え
            "right": 右揃え
        font_index: フォントファイルがttcの場合は複数フォントを含む。その中で使用するフォントのインデックスを指定する。
        stroke_width: 文字枠の太さ
        stroke_fill: 文字枠の色
    """
    assert len(anchor) == 2

    # デフォルトの行間は広いので調整
    spacing: int = 0  # 行間のピクセル数

    # テキスト描画域を取得
    x, y = org
    if font_scale is not None:
        fontPIL = ImageFont.truetype(font=font, size=font_scale, index=font_index)
        dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
        xL, yT, xR, yB = dummy_draw.multiline_textbbox(
            (x, y), text, font=fontPIL, align=align, stroke_width=stroke_width, spacing=spacing
        )
    else:
        assert type(text_w) is int or type(text_h) is int
        font_scale = 1
        while True:
            fontPIL = ImageFont.truetype(font=font, size=font_scale, index=font_index)
            dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
            xL, yT, xR, yB = dummy_draw.multiline_textbbox(
                (0, 0), text, font=fontPIL, align=align, stroke_width=stroke_width, spacing=spacing
            )
            bb_w = xR - xL
            bb_h = yB - yT
            if type(text_w) is int and bb_w > text_w:
                break
            elif type(text_h) is int and bb_h > text_h:
                break
            font_scale += 1
        font_scale -= 1
        if font_scale < 1:
            raise ValueError("指定のサイズでは描画できない")
        fontPIL = ImageFont.truetype(font=font, size=font_scale, index=font_index)
        dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
        xL, yT, xR, yB = dummy_draw.multiline_textbbox(
            (x, y), text, font=fontPIL, align=align, stroke_width=stroke_width, spacing=spacing
        )

    # 少なくともalignを"center"にした場合にxL, xRがfloatになることがあったため、intにキャスト
    xL, yT, xR, yB = int(np.floor(xL)), int(np.floor(yT)), int(np.ceil(xR)), int(np.ceil(yB))

    # anchorによる座標の変換
    img_h, img_w = img.shape[:2]
    if anchor[0] == "l":
        offset_x = xL - x
    elif anchor[0] == "m":
        offset_x = (xR + xL) // 2 - x
    elif anchor[0] == "r":
        offset_x = xR - x
    else:
        raise NotImplementedError

    if anchor[1] == "t":
        offset_y = yT - y
    elif anchor[1] == "m":
        offset_y = (yB + yT) // 2 - y
    elif anchor[1] == "b":
        offset_y = yB - y
    else:
        raise NotImplementedError

    x0, y0 = x - offset_x, y - offset_y
    xL, yT = xL - offset_x, yT - offset_y
    xR, yB = xR - offset_x, yB - offset_y

    # 画面外なら何もしない
    if xR <= 0 or xL >= img_w or yB <= 0 or yT >= img_h:
        print("out of bounds")
        return img

    # ROIを取得する
    x1, y1 = max([xL, 0]), max([yT, 0])
    x2, y2 = min([xR, img_w]), min([yB, img_h])
    roi = img[y1:y2, x1:x2]

    # ROIをPIL化してテキスト描画しCV2に戻る
    roiPIL = Image.fromarray(roi)
    draw = ImageDraw.Draw(roiPIL)
    draw.text(
        (x0 - x1, y0 - y1),
        text,
        color,
        fontPIL,
        align=align,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        spacing=spacing,
    )
    roi = np.array(roiPIL, dtype=np.uint8)
    img[y1:y2, x1:x2] = roi
