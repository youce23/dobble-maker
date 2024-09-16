import glob
import math
import os
import random
import tempfile
from enum import Enum
from typing import Any, Literal

import cv2
import img2pdf
import numpy as np
import pypdf

from card_drawer.voronoi import cvt
from cv2_image_utils import cv2_putText, imread_japanese, imwrite_japanese

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OVERLAP_VAL = 127


class CARD_SHAPE(str, Enum):  # Enumのみではjson.load, dumpに失敗するため型を指定
    CIRCLE = "CIRCLE"
    RECTANGLE = "RECTANGLE"


class VoronoiError(Exception):
    # ボロノイ分割失敗の例外
    pass


class LayoutSymbolImageError(Exception):
    # 画像をカード領域に描画するのに失敗
    pass


class DrawTextError(Exception):
    # テキスト描画に関するエラー
    pass


def _make_canvas(
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    is_canvas_for_overlap: bool,
    *,
    draw_black_in_rendering_area: bool = False,
) -> np.ndarray:
    """空のキャンバスを作成

    Args:
        shape (CARD_SHAPE): カード形状
        width (int): カード幅
        height (int): カード高さ
        margin (int): カード余白
        is_canvas_for_overlap (bool): 重複確認用のキャンバス（マスク）ならTrue, 最終出力するカード画像を作るキャンバスならFalse
        draw_black_in_rendering_area (bool, optional):
            not is_canvas_for_overlap の場合のみ有効.
            カード内の余白を除く描画可能エリアを黒塗りするならTrue.
            Defaults to False.

    Returns:
        np.ndarray: 画像
    """
    if shape == CARD_SHAPE.CIRCLE:
        center = (height // 2, width // 2)
        radius = width // 2 - margin

        if not is_canvas_for_overlap:
            # 描画キャンバス
            canvas = np.full((height, width, 3), WHITE, dtype=np.uint8)
            if draw_black_in_rendering_area:
                cv2.circle(canvas, center, radius, BLACK, thickness=-1)
        else:
            # 重複確認用キャンバス
            canvas = np.full((height, width), OVERLAP_VAL, dtype=np.uint8)
            cv2.circle(
                canvas,
                center,
                radius,
                0,  # type: ignore
                thickness=-1,
            )  # type: ignore
    elif shape == CARD_SHAPE.RECTANGLE:
        if not is_canvas_for_overlap:
            # 描画キャンバス
            if not draw_black_in_rendering_area:
                canvas = np.full((height, width, 3), WHITE, dtype=np.uint8)
            else:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), WHITE, thickness=margin, lineType=cv2.LINE_4)
        else:
            # 重複確認用キャンバス
            canvas = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(
                canvas,
                (0, 0),
                (width - 1, height - 1),
                OVERLAP_VAL,  # type: ignore
                thickness=margin,
                lineType=cv2.LINE_4,
            )  # type: ignore
    else:
        raise NotImplementedError

    return canvas


def _calc_drawing_params_by_random(
    images: list[np.ndarray],
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    show: bool,
) -> list[dict[str, Any]]:
    """画像をランダム配置するパラメータを計算

    Args:
        images (list[np.ndarray]): 配置する画像のリスト
        shape (CARD_SHAPE): カードの形状（円または矩形）
        width (int): カードの幅
        height (int): カードの高さ（円の場合は無視される）
        margin (int): カード外縁の余白
        show (bool): 計算中の画像を表示するかどうか

    Returns:
        list[dict[str, Any]]: 各画像の配置パラメータのリスト
    """
    # 描画位置を決定するための小さなキャンバス(短辺180)を準備
    scl_canv = 180 / min(width, height)
    w_scl_canv = int(width * scl_canv)
    h_scl_canv = int(height * scl_canv)
    m_scl_canv = int(np.ceil(margin * scl_canv))  # margin > 0 の場合には1以上になるようにceil

    # 画像1枚当たりの基本サイズを指定
    n_img_in_card = len(images)
    img_base_size = int(np.ceil(max(h_scl_canv, w_scl_canv) / np.ceil(np.sqrt(n_img_in_card))))

    while True:
        params: list[dict[str, Any]] = []

        # マスク画像に描画するパラメータ
        # キャンバスの作成
        canvas = _make_canvas(shape, w_scl_canv, h_scl_canv, m_scl_canv, True)
        for img in images:
            # 元画像の余白を除去して外接矩形でトリミング
            _img = _trim_bb_image(img)
            h_im_trim, w_im_trim = _img.shape[:2]
            h_im_trim = np.ceil(h_im_trim * scl_canv)
            w_im_trim = np.ceil(w_im_trim * scl_canv)

            # ランダムにリサイズ、回転、配置し、重複するならパラメータを変えて最大n_max回やり直し
            # (n_max回試してもダメなら最初からやり直し)
            n_max = 100
            cur_canv = canvas.copy()
            ok = False
            for _ in range(n_max):
                _canv = cur_canv.copy()

                # リサイズ, 移動, 回転のパラメータをランダムに決定
                _scale_base_size = img_base_size / max(h_im_trim, w_im_trim)  # 標準サイズを基準にさらにリサイズ
                scale = random.uniform(0.5, 0.8) * _scale_base_size
                h_im_resized = h_im_trim * scale
                w_im_resized = w_im_trim * scale
                center = [
                    random.randint(0, int(w_scl_canv - w_im_resized)) + w_im_resized / 2,
                    random.randint(0, int(h_scl_canv - h_im_resized)) + h_im_resized / 2,
                ]
                angle = random.randint(0, 360)

                # リサイズ, 回転, 平行移動を矩形の4頂点に対して計算
                lt = _rotate_2d((-w_im_resized / 2, -h_im_resized / 2), angle)
                rt = _rotate_2d((+w_im_resized / 2, -h_im_resized / 2), angle)
                lb = _rotate_2d((+w_im_resized / 2, +h_im_resized / 2), angle)
                rb = _rotate_2d((-w_im_resized / 2, +h_im_resized / 2), angle)

                lt = (lt[0] + center[0], lt[1] + center[1])
                rt = (rt[0] + center[0], rt[1] + center[1])
                lb = (lb[0] + center[0], lb[1] + center[1])
                rb = (rb[0] + center[0], rb[1] + center[1])

                # 画像貼り付け位置にマスクを描画
                mask_img = np.zeros(_canv.shape[:2], dtype=np.uint8)
                _pts = np.array([lt, rt, lb, rb], dtype=int)
                cv2.fillConvexPoly(mask_img, _pts, OVERLAP_VAL, lineType=cv2.LINE_4)  # type: ignore

                # キャンバスに重畳
                _canv += mask_img

                if show:
                    cv2.imshow("canv_overlap", _canv)
                    cv2.waitKey(1)

                # 重なりの確認
                if (_canv > OVERLAP_VAL).sum() == 0:
                    ok = True
                    canvas = _canv

                    p = {"scale": scale, "center_xy": (np.array(center) / scl_canv).tolist(), "rotation_deg": angle}
                    params.append(p)

                    break

            if not ok:
                break

        if ok:
            break

    return params


def _layout_random(
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    images: list[np.ndarray],
    show: bool,
) -> np.ndarray:
    """画像を重ならないようにランダムに配置する

    Args:
        shape (CARD_SHAPE): 出力画像の形
        width (int): 出力画像の幅
        height (int): 出力画像の高さ
        margin (int): 出力画像の外縁につける余白サイズ
        images (list[np.ndarray]): 配置画像ソース
        show (bool): 計算中の画像を画面表示するならTrue
    """
    # 画像毎の描画位置決め
    draw_params_images = _calc_drawing_params_by_random(
        images,
        shape,
        width,
        height,
        margin,
        show,
    )

    # 描画
    card_image = _render_images_with_params(
        images,
        shape,
        width,
        height,
        margin,
        draw_params_images,
        show,
    )

    return card_image


def _trim_bb_image(image: np.ndarray) -> np.ndarray:
    """imageの余白を除去してトリミング"""
    assert image.shape[2] == 3  # 3チャネルのカラー画像とする

    thr = 5  # 二値化の閾値

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, im_bin = cv2.threshold(im_gray, 255 - thr, OVERLAP_VAL, cv2.THRESH_BINARY_INV)
    wh = np.where(im_bin == OVERLAP_VAL)
    y0 = min(wh[0])
    x0 = min(wh[1])
    y1 = max(wh[0])
    x1 = max(wh[1])

    trim = image[y0:y1, x0:x1, :]

    return trim


def _rotate_2d(pt: tuple[float, float], degree: float, *, org: tuple[float, float] = (0, 0)) -> tuple[float, float]:
    """2次元座標をorgを原点として回転"""
    x = pt[0] - org[0]
    y = pt[1] - org[1]
    rad = math.radians(degree)
    rot_x = x * math.cos(rad) - y * math.sin(rad)
    rot_y = x * math.sin(rad) + y * math.cos(rad)

    return (rot_x + org[0], rot_y + org[1])


def _make_drawing_region_in_card(shape: CARD_SHAPE, width: int, height: int, margin: int) -> list[tuple[float, float]]:
    """カード内の描画範囲を取得

    Args:
        shape (CARD_SHAPE): カードの形
        width (int): カードの幅
        height (int): カードの高さ (円なら無視)
        margin (int): カード外縁の余白

    Returns:
        list[tulpe[float, float]]: 描画範囲を凸包表現した際の座標リスト
    """
    if shape == CARD_SHAPE.CIRCLE:
        c = width / 2
        r = width / 2 - margin
        bnd_pts = [(c + r * np.cos(x / 180.0 * np.pi), c + r * np.sin(x / 180.0 * np.pi)) for x in range(0, 360, 5)]
    elif shape == CARD_SHAPE.RECTANGLE:
        bnd_pts = [
            (margin, margin),
            (width - margin, margin),
            (width - margin, height - margin),
            (margin, height - margin),
        ]
        bnd_pts = [(float(x), float(y)) for x, y in bnd_pts]  # cast
    else:
        raise NotImplementedError

    return bnd_pts


def _calc_drawing_params_by_vorooni(
    images: list[np.ndarray],
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    radius_p: float,
    n_iters: int,
    min_image_size_rate: float,
    max_image_size_rate: float,
    show: bool,
) -> list[dict[str, Any]]:
    """画像をボロノイ分割に基づいて配置するためのパラメータを計算する

    Args:
        images (list[np.ndarray]): 配置する画像のリスト
        shape (CARD_SHAPE): カードの形状（円または矩形）
        width (int): カードの幅
        height (int): カードの高さ（円の場合は無視される）
        margin (int): カード外縁の余白
        radius_p (float): ボロノイ分割の半径パラメータ
        n_iters (int): ボロノイ分割の反復回数
        min_image_size_rate (float): 画像の最小サイズの割合
        max_image_size_rate (float): 画像の最大サイズの割合
        show (bool): 計算中の画像を表示するかどうか

    Raises:
        VoronoiError: ボロノイ分割に失敗した場合に発生
        LayoutSymbolImageError: シンボル画像のレイアウトに失敗した場合に発生

    Returns:
        list[dict[str, Any]]: 各画像の配置パラメータのリスト
    """
    # 描画位置を決定するための小さなキャンバス(短辺180)を準備
    scl_canv = 180 / min(width, height)
    width = int(width * scl_canv)
    height = int(height * scl_canv)
    margin = int(np.ceil(margin * scl_canv))  # margin > 0 の場合には1以上になるようにceil

    # 余白とカード形状を考慮した画像の描画可能範囲を取得
    bnd_pts = _make_drawing_region_in_card(shape, width, height, margin)

    # 重心ボロノイ分割で各画像の中心位置と範囲を取得
    try:
        # NOTE: ここのshowは毎回止まってしまうのでデバッグ時に手動でTrueにする
        center_images, rgn_images = cvt(bnd_pts, len(images), radius_p=radius_p, n_iters=n_iters, show_step=None)
    except Exception as e:
        # radius_p を設定した場合に初期値によって例外が生じることがある(0.0指定時は生じたことはない)
        raise VoronoiError(f"ボロノイ分割が失敗 (radius_p({radius_p:.2f})が大きすぎる可能性が高い)") from e

    # 重なりチェックをするためのマスク画像 (1ch) を作成
    # 描画後に(0またはOVERLAP_VAL)以外の値があったら、その画素は重なりがあったことを意味する
    canvas = _make_canvas(shape, width, height, margin, True)

    # 各画像をpos_imagesに配置
    params: list[dict[str, Any]] = []
    for i_img, img in enumerate(images):
        # 元画像の余白を除去して外接矩形でトリミング
        im_trim = _trim_bb_image(img)
        im_h, im_w = im_trim.shape[:2]
        im_h = np.ceil(im_h * scl_canv)
        im_w = np.ceil(im_w * scl_canv)

        center = center_images[i_img]  # ボロノイ領域の重心 (描画の中心座標とする) (x, y)
        rgn = rgn_images[i_img]  # ボロノイ領域境界 (x, y)
        # 貼り付ける画像の最大サイズは、ボロノイ領域の最大長に長辺が入るサイズとする
        mx_len_rgn = max(np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in rgn for p2 in rgn)

        init_deg = random.randint(0, 360)  # 初期角度をランダムに決める

        ok = False
        scl = 100  # シンボル画像のスケール率 (%)
        SCL_LIMIT = 5  # 縮小率の下限
        SCL_DECREASE_RATE = 0.95  # シンボル画像のスケール率の減衰率
        l_lngside = max(im_h, im_w)  # 元画像の長辺長
        card_lngside = max(width, height)  # カードの長辺長
        symbol_size_range = (min_image_size_rate * card_lngside, max_image_size_rate * card_lngside)
        scl_l_lngside = l_lngside  # 初期値
        while scl > SCL_LIMIT and symbol_size_range[0] <= scl_l_lngside:
            # 元画像の対角長とボロノイ領域の最大長が一致するスケールを100%としてスケールを計算
            scl_r = mx_len_rgn / l_lngside * (scl / 100)
            scl_l_lngside = int(scl_r * l_lngside)
            if scl_l_lngside > symbol_size_range[1]:
                scl *= SCL_DECREASE_RATE
                continue
            for deg in range(0, 180, 10):  # 画像を少しずつ回転 (矩形で試しているので180度までの確認で良い)
                # 画像と同サイズの矩形をキャンバスに重畳描画
                rot_deg = init_deg + deg
                lt = _rotate_2d((-im_w * scl_r / 2, -im_h * scl_r / 2), rot_deg)
                rt = _rotate_2d((+im_w * scl_r / 2, -im_h * scl_r / 2), rot_deg)
                lb = _rotate_2d((+im_w * scl_r / 2, +im_h * scl_r / 2), rot_deg)
                rb = _rotate_2d((-im_w * scl_r / 2, +im_h * scl_r / 2), rot_deg)

                lt = (lt[0] + center[0], lt[1] + center[1])
                rt = (rt[0] + center[0], rt[1] + center[1])
                lb = (lb[0] + center[0], lb[1] + center[1])
                rb = (rb[0] + center[0], rb[1] + center[1])

                # 重なりなく描画できるか確認
                # 画像貼り付け位置にマスクを描画
                mask_img = np.zeros(canvas.shape[:2], dtype=np.uint8)
                _pts = np.array([lt, rt, lb, rb], dtype=int)
                cv2.fillConvexPoly(mask_img, _pts, OVERLAP_VAL, lineType=cv2.LINE_4)  # type: ignore

                # ボロノイ境界を超えない制約をつけるために境界線を描画
                # polylines (やfillPoly) は[(x0, y0), (x1, y1), ...]を以下の形にreshapeしないと動かない
                # 参考: https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
                #
                # ボロノイ境界はmarginなど他のマスクと重なることがあるので、重畳ではなくOVERLAP_VALの描画にする
                _canvas = canvas.copy()
                _pts = np.array(rgn).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(_canvas, [_pts], True, OVERLAP_VAL, thickness=1, lineType=cv2.LINE_4)  # type: ignore

                # 画像マスクとボロノイ境界マスクを重畳
                _canvas += mask_img

                if show:
                    cv2.imshow("canv_overlap", _canvas)
                    cv2.waitKey(1)

                # 重なりの確認
                if (_canvas > OVERLAP_VAL).sum() == 0:
                    ok = True

                    # 重なりがなければ改めてマスクに画像マスクを重畳
                    canvas += mask_img

                    p = {"scale": scl_r, "center_xy": (np.array(center) / scl_canv).tolist(), "rotation_deg": rot_deg}
                    params.append(p)

                    break
            if ok:
                break

            scl *= SCL_DECREASE_RATE  # 前回の大きさを基準に一定割合だけ縮小

        if not ok:
            break

    if not ok:
        # ここでOKになっていないということは画像を限界まで小さくしてもボロノイ領域に収まらなかったことを意味するので例外とする
        raise LayoutSymbolImageError(
            "ボロノイ領域内へのシンボル画像の描画に失敗 "
            f"(min_image_size_rate ({min_image_size_rate}) を小さくする, "
            "あるいはボロノイ領域がより大きくなるようなパラメータ調整が必要)"
        )

    assert len(params) == len(images)

    return params


def _render_images_with_params(
    images: list[np.ndarray],
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    draw_params_images: list[dict[str, Any]],
    show: bool,
) -> np.ndarray:
    """
    各シンボル画像をカード内に描画

    Args:
        images (list[np.ndarray]): シンボル画像のリスト
        shape (CARD_SHAPE): カード形状
        width (int): カード幅
        height (int): カード高さ
        margin (int): カードの外縁につける余白サイズ
        draw_params_images (list[dict[str, Any]]): 各シンボル画像の描画パラメータ
        show (bool): 計算中の画像を画面表示するならTrue

    Returns:
        np.ndarray: 描画結果
    """
    assert len(images) == len(draw_params_images)

    # 画像を描画するためのキャンバスを作成
    canvas = _make_canvas(shape, width, height, margin, False)

    # 各画像をキャンバスに描画
    for param, img in zip(draw_params_images, images):
        scale = param["scale"]
        center = param["center_xy"]
        rotation_deg = param["rotation_deg"]

        # 画像をトリミング
        img_trim = _trim_bb_image(img)

        # 画像をスケーリング
        img_resized = cv2.resize(img_trim, None, fx=scale, fy=scale, interpolation=_get_interpolation(scale))

        # 画像を回転
        img_rotated = _rotate_fit(img_resized, -rotation_deg, flags=cv2.INTER_CUBIC, borderValue=WHITE)

        # 画像をキャンバスに重畳
        dx = center[0] - img_rotated.shape[1] / 2
        dy = center[1] - img_rotated.shape[0] / 2
        mv_mat = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore
        img_affine = cv2.warpAffine(
            img_rotated,
            mv_mat,  # type: ignore
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderValue=WHITE,
        )  # type: ignore
        mask = np.any(img_affine != WHITE, axis=-1)
        canvas[mask] = img_affine[mask]

        if show:
            cv2.imshow("canvas", canvas)
            cv2.waitKey(1)

    return canvas


def _layout_voronoi(
    shape: CARD_SHAPE,
    width: int,
    height: int,
    margin: int,
    images: list[np.ndarray],
    show: bool,
    *,
    radius_p: float = 0.0,
    n_iters: int = 20,
    min_image_size_rate: float = 0.0,
    max_image_size_rate: float = 1.0,
) -> np.ndarray:
    """画像を重ならないように重心ボロノイ分割で決めた位置にランダムに配置する

    Args:
        shape (CARD_SHAPE): 出力画像の形
        width (int): 出力画像の幅
        height (int): 出力画像の高さ (sahpe == CIRCLEならwidthと同じであること)
        margin (int): 出力画像の外縁につける余白サイズ
        images (list[np.ndarray]): 配置画像ソース
        show (bool): 計算中の画像を画面表示するならTrue
        radius_p: 各母点の半径を決めるパラメータ, 0.0なら通常のボロノイ分割
        n_iters (int): 重心ボロノイ分割の反復回数
        min_image_size_rate (float): 画像長辺サイズの最小サイズ (max(width, height)に対する比, 0.0 以上 max_image_size_rate 未満)
        max_image_size_rate (float): 画像長辺サイズの最大サイズ (max(width, height)に対する比, min_image_size_rate 以上 1.0 以下)

    Raises:
        VoronoiError:
            ボロノイ分割の失敗.
            初期値に依存するため何度か試せば動く可能性がある.
            複数回試しても動かない場合は例外メッセージを参照.
        LayoutSymbolImageError:
            ボロノイ領域へのシンボル画像描画失敗.
            生成されるボロノイ領域に影響を受けるため何度か試せば動く可能性がある.
            複数回試しても動かない場合は例外メッセージを参照.

    Returns:
        描画結果の画像
    """
    if __debug__:
        if shape == CARD_SHAPE.CIRCLE:
            assert width == height
    assert 0.0 <= min_image_size_rate < max_image_size_rate <= 1.0

    # 画像毎の描画位置決め
    draw_params_images = _calc_drawing_params_by_vorooni(
        images,
        shape,
        width,
        height,
        margin,
        radius_p,
        n_iters,
        min_image_size_rate,
        max_image_size_rate,
        show,
    )

    # 描画
    card_image = _render_images_with_params(
        images,
        shape,
        width,
        height,
        margin,
        draw_params_images,
        show,
    )

    return card_image


def layout_images_randomly_wo_overlap(
    images: list[np.ndarray],
    image_indexes: list[int],
    canv_size: int | tuple[int, int],
    margin: int,
    card_shape: CARD_SHAPE,
    *,
    method: Literal["random", "voronoi"] = "random",
    radius_p: float = 0.0,
    n_voronoi_iters: int = 20,
    min_image_size_rate: float = 0.0,
    max_image_size_rate: float = 1.0,
    draw_frame: bool = False,
    show: bool = False,
) -> np.ndarray:
    """画像を重ならないように配置する

    Args:
        images (list[np.ndarray]): 画像リスト
        image_indexes (list[int]): 使用する画像のインデックスのリスト
        canv_size (Union[int, tuple[int, int]]):
            配置先の画像サイズ.
            card_shapeがCIRCLEならintで円の直径、RECTANGLEならtuple[int, int]で矩形の幅, 高さとする
        margin (int): 配置先の画像の外縁につける余白サイズ
        method: レイアウト方法
            "random": ランダム配置
            "voronoi": 重心ボロノイ分割に基づき配置
        radius_p: methodが"voronoi"の場合に各母点の半径を決めるパラメータ, 0.0なら通常のボロノイ分割
        min_image_size: methodが"voronoi"の場合に、各画像の長辺サイズが必ずこのサイズ以上となるようにする
        max_image_size: methodが"voronoi"の場合に、各画像の長辺サイズが必ずこのサイズ未満となるようにする
        n_voronoi_iters: method == "voronoi"の場合の反復回数
        draw_frame (bool): 印刷を想定した枠を描画するならTrue
        show (bool): (optional) 計算中の画像を画面表示するならTrue

    Returns:
        np.ndarray: カード画像
    """
    tar_images = [images[i] for i in image_indexes]

    # 出力先画像を初期化
    if card_shape == CARD_SHAPE.CIRCLE:
        assert isinstance(canv_size, int)
        width = height = canv_size
    elif card_shape == CARD_SHAPE.RECTANGLE:
        assert isinstance(canv_size, tuple) and len(canv_size) == 2
        width, height = canv_size
    else:
        raise NotImplementedError

    if method == "random":
        canvas = _layout_random(card_shape, width, height, margin, tar_images, show)
    elif method == "voronoi":
        n = 0
        n_max = 20  # 最大チャレンジ回数
        while True:
            print(f"** trial {n + 1} (max: {n_max})")
            try:
                canvas = _layout_voronoi(
                    card_shape,
                    width,
                    height,
                    margin,
                    tar_images,
                    show,
                    radius_p=radius_p,
                    n_iters=n_voronoi_iters,
                    min_image_size_rate=min_image_size_rate,
                    max_image_size_rate=max_image_size_rate,
                )
                break  # 例外が起こらなければそのまま終了
            except (VoronoiError, LayoutSymbolImageError):
                n += 1
                if n >= n_max:
                    raise  # ダメならそのまま例外を上げる

    else:
        raise ValueError(f"method ('{method}') の指定ミス")

    if draw_frame:
        gray = (127, 127, 127)
        if card_shape == CARD_SHAPE.CIRCLE:
            center = (height // 2, width // 2)
            radius = width // 2
            cv2.circle(canvas, center, radius, gray, thickness=1)
        elif card_shape == CARD_SHAPE.RECTANGLE:
            cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), gray, thickness=1)
        else:
            raise NotImplementedError

    return canvas


def _rotate_fit(
    img: np.ndarray,
    angle: int,
    *,
    flags: int = cv2.INTER_CUBIC,
    borderValue: cv2.typing.Scalar = 0,  # type: ignore
) -> np.ndarray:
    """
    画像を回転しはみ出ないように画像サイズを修正

    参考: https://qiita.com/ryokomy/items/0d1a879cac59a0bfbdc5

    Args:
        img (np.ndarray): 入力画像
        angle (int): 角度 [degree]
        flags (int): 画素の補間方法
        borderValue: 境界の画素値算出時に画像の外側に持つ値

    Returns:
        np.ndarray: 回転後の画像 (サイズはimgと異なる)

    """
    shp = img.shape
    height, width = shp[0], shp[1]

    # 回転角の指定
    angle_rad = angle / 180.0 * np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(height * np.absolute(np.sin(angle_rad)) + width * np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(height * np.absolute(np.cos(angle_rad)) + width * np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (width / 2, height / 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] - width / 2 + w_rot / 2
    affine_matrix[1][2] = affine_matrix[1][2] - height / 2 + h_rot / 2

    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=flags, borderValue=borderValue)

    return img_rot


def _get_interpolation(scale: float):
    """cv2.resize の interpolation の値を決める"""
    downsize = cv2.INTER_AREA  # 縮小用
    upsize = cv2.INTER_CUBIC  # 拡大用

    return downsize if scale < 1 else upsize


def load_images(
    dir_name: str,
    num: int,
    *,
    ext: list[str] = ["jpg", "png"],
    shuffle: bool = False,
    trim_margin: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """所定数の画像を読み込む

    画像ファイル名がすべて整数表記であれば数値でソートする。
    整数表記でないファイルが1つでもあれば、文字列でソートする。

    Args:
        dir_name (str): 画像フォルダ
        num (int): 読み込む画像数
        ext (list[str], optional): 読み込み対象画像の拡張子. Defaults to ["jpg", "png"].
        shuffle (bool, optional): Trueなら画像ファイル一覧をシャッフルする. Defaults to False.
        trim_margin (bool, optional): Trueなら画像の余白を除去する

    Returns:
        list[np.ndarray]: 読み込んだnum個の画像のリスト
        list[str]: 各画像のファイルパス
    """
    # 画像ファイル一覧を取得
    files_searched: list[str] = [fname for e in ext for fname in glob.glob(f"{dir_name}/*.{e}")]
    if len(files_searched) < num:
        # ファイル数が足りないので例外を返す
        raise ValueError(f"{dir_name}に{num}個以上の画像が存在しない")
    # [0]: ファイルパス (例: "images/1.jpg")
    # [1]: ベースネーム (例: "1" --> 1)
    files_path_base: list[tuple[str, str] | tuple[str, int]] = [
        (x, os.path.splitext(os.path.basename(x))[0]) for x in files_searched
    ]
    if shuffle:
        random.shuffle(files_path_base)
    else:
        # 試しにキャスト
        try:
            files_path_base = [(x[0], int(x[1])) for x in files_path_base]
        except ValueError:
            # キャスト失敗ならstrのままにする
            pass
        # ソート
        files_path_base = sorted(files_path_base, key=lambda x: x[1])
    # 必要数に画像を制限
    files_path_base = files_path_base[:num]

    # 画像読み込み
    images: list[np.ndarray] = []
    image_paths: list[str] = []

    for path, _ in files_path_base:
        img = imread_japanese(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"{path} の読み込み失敗")
        h, w, n_ch = img.shape
        if n_ch == 4:
            # alphaチャネルがある場合、背景を白画素(255, 255, 255)として合成する
            img_bg = np.full((h, w, 3), WHITE, dtype=np.uint8)
            img = (img_bg * (1 - img[:, :, 3:] / 255) + img[:, :, :3] * (img[:, :, 3:] / 255)).astype(np.uint8)
        elif n_ch == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif n_ch == 3:
            # 3チャネルなら何もしない
            pass
        else:
            raise IOError(f"{path} のチャネル数({n_ch})はサポート対象外")

        if trim_margin:
            img = _trim_bb_image(img)
        images.append(img)
        image_paths.append(path)

    assert len(images) == num
    assert len(image_paths) == num

    return images, image_paths


def _merge_pdf(pdf_paths: list[str], output_pdf: str):
    """PDFファイルをマージして新たなPDFファイルを出力

    Args:
        pdf_paths (list[str]): マージ元のPDFファイルパス, 記載の順序で結合される
        output_pdf (str): 出力PDFファイルのパス
    """
    output_dir = os.path.dirname(output_pdf)
    os.makedirs(output_dir, exist_ok=True)

    writer = pypdf.PdfWriter()
    for p in pdf_paths:
        writer.append(p)

    writer.write(output_pdf)
    writer.close()


def _centering_img(src: np.ndarray) -> np.ndarray:
    """入力画像 src の画像領域を中心に配置した画像を返す"""
    assert src.shape[2] == 3  # 3チャネルのカラー画像とする

    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bin_src = cv2.threshold(gray_src, 254, 255, cv2.THRESH_BINARY)
    pos = np.where(bin_src == 0)
    y0 = min(pos[0])
    y1 = max(pos[0])
    x0 = min(pos[1])
    x1 = max(pos[1])
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    page_w = src.shape[1]
    page_h = src.shape[0]

    x = (page_w - w) // 2
    y = (page_h - h) // 2

    dst = np.ones_like(src) * 255
    dst[y : (y + h), x : (x + w), :] = src[y0 : (y0 + h), x0 : (x0 + w), :]

    return dst


def images_to_pdf(
    images: list[np.ndarray],
    pdf_path: str,
    *,
    dpi: int = 300,
    card_long_side_mm: int = 95,
    width_mm: int = 210,
    height_mm: int = 297,
    centering: bool = True,
):
    """画像セットを並べてPDF化し保存

    Args:
        images (list[np.ndarray]): 画像セット, すべて画像サイズは同じとする
        pdf_path (str): 出力するPDFのフルパス
        dpi (int, optional): PDFの解像度. Defaults to 300.
        card_width_mm (int, optional): 画像1枚の長辺サイズ(mm). Defaults to 95.
        width_mm (int, optional): PDFの幅(mm). Defaults to 210 (A4縦).
        height_mm (int, optional): PDFの高さ(mm). Defaults to 297 (A4縦).
        centering (bool, optional): TrueならPDFの中心に画像を配置
    """
    tmpdir = tempfile.TemporaryDirectory()  # 一時フォルダ

    MM_PER_INCH = 25.4
    pix_per_mm = dpi / MM_PER_INCH  # 1mmあたりのピクセル数
    width = int(width_mm * pix_per_mm)  # 紙の幅 [pix]
    height = int(height_mm * pix_per_mm)  # 紙の高さ [pix]
    resize_card_long = int(card_long_side_mm * pix_per_mm)  # カードの長辺サイズ [pix]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)  # キャンバスの初期化
    pos_x = pos_y = 0  # 描画位置
    pos_y_next = 0  # 次の描画位置
    i_pdf = 0  # PDFの番号
    for card in images:
        card_h = card.shape[0]
        card_w = card.shape[1]
        scale = resize_card_long / max(card_h, card_w)
        resize_card = cv2.resize(card, None, fx=scale, fy=scale, interpolation=_get_interpolation(scale))

        resize_h = resize_card.shape[0]
        resize_w = resize_card.shape[1]
        if pos_x + resize_w < width and pos_y + resize_h < height:
            # 収まるならキャンバスに貼り付け
            canvas[pos_y : (pos_y + resize_h), pos_x : (pos_x + resize_w), :] = resize_card
            pos_y_next = pos_y + resize_h
            pos_x += resize_w
        else:
            # 収まらないなら改行してみる
            pos_x = 0
            pos_y = pos_y_next
            if pos_y + resize_h < height:
                # 収まるなら貼り付け
                canvas[pos_y : (pos_y + resize_h), pos_x : (pos_x + resize_w), :] = resize_card
                pos_y_next = pos_y + resize_h
                pos_x += resize_w
            else:
                # 収まらないならPDF出力してから次のキャンバスの先頭に描画
                tmp_name = tmpdir.name + os.sep + f"{i_pdf}.png"
                pdf_name = tmpdir.name + os.sep + f"{i_pdf}.pdf"
                _out_img = _centering_img(canvas) if centering else canvas
                imwrite_japanese(tmp_name, _out_img)
                with open(pdf_name, "wb") as f:
                    bytes = img2pdf.convert(tmp_name, layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi)))
                    f.write(bytes)

                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
                pos_x = pos_y = pos_y_next = 0
                canvas[pos_y : (pos_y + resize_h), pos_x : (pos_x + resize_w), :] = resize_card
                pos_y_next = pos_y + resize_h
                pos_x += resize_w

                i_pdf += 1

    # 最後の1枚が残っていたらPDF出力
    if (canvas != 255).any():
        tmp_name = tmpdir.name + os.sep + f"{i_pdf}.png"
        pdf_name = tmpdir.name + os.sep + f"{i_pdf}.pdf"
        _out_img = _centering_img(canvas) if centering else canvas
        imwrite_japanese(tmp_name, _out_img)
        with open(pdf_name, "wb") as f:
            bytes = img2pdf.convert(tmp_name, layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi)))
            f.write(bytes)

        n_pdf = i_pdf + 1
    else:
        n_pdf = i_pdf

    # すべてのPDFを1つのPDFにまとめて結合
    pdfs = [tmpdir.name + os.sep + f"{i}.pdf" for i in range(n_pdf)]
    _merge_pdf(pdfs, pdf_path)

    return


def _select_font() -> tuple[str, int]:
    """使用するフォントを決定

    Raises:
        NotImplementedError: 使用するフォントが見つからない (Windows標準フォント)

    Returns:
        tuple[str, int]: fontパス, font_index
    """
    font_dir = os.environ["windir"] + os.sep + "fonts"

    # 以下のフォントを順次検索し見つかればそれを使う
    # フォント名は次の関数で確認が可能
    #   ImageFont.truetype(font=ttcファイルパス, index=インデックス).getname()
    candidates = [
        ("uddigikyokashon-b.ttc", 2),  # UD デジタル 教科書体 NK-B (Bold)
        ("meiryob.ttc", 2),  # Meiryo UI (Bold)
        ("msgothic.ttc", 2),  # ＭＳ Ｐゴシック (Regular)
    ]
    for font, font_ind in candidates:
        path = font_dir + os.sep + font
        if os.path.exists(path):
            return path, font_ind

    raise NotImplementedError("Windows標準フォントが見つからない")


def _draw_name_in_thumb(
    img: np.ndarray,
    thumb_w: int,
    thumb_h: int,
    name: str,
    *,
    text_h_rate: float = 1.0,
) -> np.ndarray:
    """画像＋テキストをサムネイル化

    Args:
        img (np.ndarray):
            入力画像, 3チャネルのBGR画像であること
        thumb_w (int):
            サムネイル幅 (テキストを含む)
        thumb_h (int):
            サムネイル高さ (テキストを含む)
        name (str):
            描画テキスト
        text_h_rate:
            thumb_h に対する描画テキストの最大高さの割合

    Returns:
        np.ndarray: サムネイル + 直下にテキストをレンダリングした画像
    """
    assert img.shape[2] == 3

    # Windows標準のフォントから使用するフォントを選択
    font, font_index = _select_font()

    img_w, img_h = img.shape[1], img.shape[0]
    scale = min(thumb_w, thumb_h) / max(img_w, img_h)  # サムネイルの短辺に画像の長辺が収まるように縮小しないといけない

    canv = np.ones((thumb_h, thumb_w, 3), dtype=img.dtype) * 255
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=_get_interpolation(scale))
    x0 = (canv.shape[1] - resized.shape[1]) // 2
    y0 = (canv.shape[0] - resized.shape[0]) // 2
    canv[y0 : (y0 + resized.shape[0]), x0 : (x0 + resized.shape[1]), :] = resized
    cv2_putText(
        canv,
        name,
        (canv.shape[1] // 2, canv.shape[0]),
        font,
        None,
        (0, 0, 0),
        "mb",
        align="center",
        font_index=font_index,
        stroke_width=3,
        stroke_fill=(255, 255, 255),
        text_w=canv.shape[1],
        text_h=int(np.floor(canv.shape[0] * text_h_rate)),
    )

    return canv


def _judge_to_draw_thumb_in_card(
    canvas_ov: np.ndarray,
    table_x0: int | float,
    table_y0: int | float,
    pos_x: int,
    pos_y: int,
    cell_w: int | float,
    cell_h: int | float,
    margin_thumb_w: int | float,
    margin_thumb_h: int | float,
    thumb_w: int,
    thumb_h: int,
    *,
    show: bool = False,
    show_wait_ms: int = 0,
) -> tuple[tuple[int, int], np.ndarray] | None:
    """カード画像の指定位置にサムネイル画像が描画可能か確認

    描画できるなら描画位置と確認用のマスク画像（描画済み）を返す

    Args:
        canvas_ov (np.ndarray): 確認用のマスク画像
        table_x0 (int | float): 全サムネイルを表形式で描画する表の左上座標
        table_y0 (int | float): 全サムネイルを表形式で描画する表の左上座標
        pos_x (int): 表内のx方向インデックス
        pos_y (int): 表内のy方向インデックス
        cell_w (int | float): 表のセル幅
        cell_h (int | float): 表のセル高さ
        margin_thumb_w (int | float): セル内の縁の余白サイズ
        margin_thumb_h (int | float): セル内の縁の余白サイズ
        thumb_w (int): サムネイル画像サイズ, thumb_w + margin_thumb_w <= cell_w であること
        thumb_h (int): サムネイル画像サイズ, thumb_h + margin_thumb_h <= cell_h であること
        show (bool, optional): _description_. Defaults to False.
        show_wait_ms (int, optional): _description_. Defaults to 0.

    Returns:
        tuple[tuple[int, int], np.ndarray] | None:
            - 指定位置にサムネイルを描画できないならNone
            - 描画できるなら (描画位置座標 x, y), 更新後のマスク画像
    """
    assert thumb_w + margin_thumb_w <= cell_w
    assert thumb_h + margin_thumb_h <= cell_h

    # 試しに描画
    try_canv_ov = np.copy(canvas_ov)
    x0 = table_x0 + pos_x * cell_w
    y0 = table_y0 + pos_y * cell_h
    x0_m = int(x0 + margin_thumb_w)
    y0_m = int(y0 + margin_thumb_h)
    try_canv_ov[
        y0_m : (y0_m + thumb_h),
        x0_m : (x0_m + thumb_w),
    ] += OVERLAP_VAL

    if show:
        cv2.imshow("canvas_ov", canvas_ov)
        cv2.imshow("try", try_canv_ov)
        cv2.waitKey(show_wait_ms)

    # 端がギリギリで入りにくいので、適当に重なりを許容
    n_ov_pixs = (try_canv_ov > OVERLAP_VAL).sum() - (canvas_ov > OVERLAP_VAL).sum()

    # マスク画像内で重なりが無ければ描画成功
    if n_ov_pixs <= max(thumb_w, thumb_h):
        return ((x0_m, y0_m), try_canv_ov)
    else:
        return None


def _draw_frame_of_card(width: int, height: int, card_shape: CARD_SHAPE, card: np.ndarray):
    """カードの枠線を描画 (cardsが更新される)

    Args:
        width (int): カードサイズ
        height (int): カードサイズ
        card_shape (CARD_SHAPE): カードの形状
        cards (np.ndarray): カード画像 (更新される)

    Raises:
        NotImplementedError: _description_
    """
    gray = (127, 127, 127)
    if card_shape == CARD_SHAPE.CIRCLE:
        center = (height // 2, width // 2)
        radius = width // 2
        cv2.circle(card, center, radius, gray, thickness=1)
    elif card_shape == CARD_SHAPE.RECTANGLE:
        cv2.rectangle(card, (0, 0), (width - 1, height - 1), gray, thickness=1)
    else:
        raise NotImplementedError


def make_image_of_thumbnails_with_names(
    card_shape: CARD_SHAPE,
    card_size: int | tuple[int, int],
    margin: int,
    table_size: tuple[int, int],
    images: list[np.ndarray],
    names: list[str],
    *,
    thumb_margin: float = 0.05,
    text_h_rate: float = 0.25,
    draw_frame: bool = False,
    show: bool = False,
    show_wait_ms: int = 0,
) -> list[np.ndarray]:
    """シンボル画像のサムネイルと名前を一覧するカード画像を作成

    Args:
        card_shape (CARD_SHAPE): 出力画像の形
        card_size: (int | tuple[int, int]):
            出力画像のサイズ, card_shapeがCIRCLEならintで円の直径、RECTANGLEならtuple[int, int]で矩形の幅, 高さとする
        margin (int): 出力画像の外縁につける余白サイズ
        table_size (tuple[int, int]): サムネイルを表形式で並べる際の表の(行数, 列数)
        images (list[np.ndarray]): 配置画像ソース
        names (list[str]): images毎の表示名
        thumb_margin (float):
            出力画像の描画領域を表サイズで分割した領域(セル)において、
            さらにサムネイルの周囲につける余白のセルサイズに対する割合 (0以上0.5未満)
        text_h_rate (float):
            サムネイルの高さに対する描画テキストの最大高さの割合
        draw_frame (bool): 印刷を想定した枠を描画するならTrue
        show (bool): 計算中の画像を画面表示するならTrue
        show_wait_ms (int): show is True 時に表示するウィンドウの待ち時間, 0ならキー入力

    Returns:
        list[np.ndarray]:
            サムネイルを並べたカード画像のリスト
    """
    if card_shape == CARD_SHAPE.CIRCLE:
        assert isinstance(card_size, int)
        width = height = card_size
    elif card_shape == CARD_SHAPE.RECTANGLE:
        assert isinstance(card_size, tuple) and len(card_size) == 2
        width, height = card_size
    else:
        raise NotImplementedError

    assert len(images) == len(names)
    assert 0.0 <= thumb_margin < 0.5

    table_scale = 1.0
    if card_shape == CARD_SHAPE.CIRCLE:
        table_scale = 0.98  # 円の場合、端にはシンボルを描画できないので少し小さめのサイズを描画範囲とする
    table_x0 = margin + width * (1 - table_scale) / 2
    table_y0 = margin + height * (1 - table_scale) / 2

    # テーブルのセルサイズを計算
    cell_w = table_scale * (width - 2 * margin) / table_size[1]
    cell_h = table_scale * (height - 2 * margin) / table_size[0]

    # 描画可能なサムネイル領域のサイズを計算
    _thumb_w_f = cell_w - 2 * thumb_margin * cell_w
    _thumb_h_f = cell_h - 2 * thumb_margin * cell_h
    thumb_w = int(_thumb_w_f)
    thumb_h = int(_thumb_h_f)

    # セルサイズとサムネイル領域サイズの差 (＝余白) サイズを計算
    margin_thumb_w = (cell_w - _thumb_w_f) / 2
    margin_thumb_h = (cell_h - _thumb_h_f) / 2

    # 各画像 + テキストをレンダリングした画像を生成
    thumbs: list[np.ndarray] = []
    for symbl_img, symbl_name in zip(images, names):
        try:
            thumb = _draw_name_in_thumb(symbl_img, thumb_w, thumb_h, symbl_name, text_h_rate=text_h_rate)
        except Exception as e:
            raise DrawTextError(
                "サムネイル(シンボル一覧)にテキストを描画できない (最大文字高さ比が小さすぎる可能性が高い)"
            ) from e

        thumbs.append(thumb)

    # 各画像をカードにテーブル形式で描画
    cards: list[np.ndarray] = []

    canvas = None
    canvas_ov = None
    pos_x = pos_y = 0  # 描画位置
    cnt_img_in_card = 0  # カードに描画された画像数 (エラーチェック用に使う)
    for thumb_img in thumbs:
        ok = False  # thumb_imgがカードに描画できたか否かを管理するフラグ
        # 対象のサムネイル画像を描画できる場所をカード内のセル位置を更新しながら探索
        while not ok:
            if canvas is None:
                canvas = _make_canvas(card_shape, width, height, margin, False, draw_black_in_rendering_area=True)
                canvas_ov = _make_canvas(card_shape, width, height, margin, True)

            # 描画位置に入るか確認
            assert canvas_ov is not None
            ret = _judge_to_draw_thumb_in_card(
                canvas_ov,
                table_x0,
                table_y0,
                pos_x,
                pos_y,
                cell_w,
                cell_h,
                margin_thumb_w,
                margin_thumb_h,
                thumb_w,
                thumb_h,
                show=show,
                show_wait_ms=show_wait_ms,
            )
            if ret is not None:
                # 描画可能なので描画
                (x0_m, y0_m), canvas_ov = ret  # 描画先座標, 更新後のマスク画像
                canvas[
                    y0_m : (y0_m + thumb_h),
                    x0_m : (x0_m + thumb_w),
                    :,
                ] = thumb_img
                ok = True  # 描画成功のフラグ更新
                cnt_img_in_card += 1  # カード内のサムネイル数を更新

            if show:
                cv2.imshow("canvas", canvas)
                cv2.waitKey(show_wait_ms)

            # 描画先のセル位置を更新
            pos_x += 1
            if pos_x >= table_size[1]:
                pos_x = 0
                pos_y += 1
            if pos_y >= table_size[0]:
                # このカードがいっぱいだったら、カードを出力してから次の画像へ
                if cnt_img_in_card == 0:
                    # 1個も描画できないまま次の画像へ行ってしまったら、テーブルサイズの設定ミスなので例外送出
                    raise ValueError("カードにサムネイル(シンボル一覧)が描画できないのでテーブルサイズの調整が必要")
                canvas = np.where(canvas_ov[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)
                cards.append(canvas)
                canvas = None
                cnt_img_in_card = 0
                pos_x = 0
                pos_y = 0

    if cnt_img_in_card > 0:
        # 未出力のカード画像を処理
        assert canvas is not None
        assert canvas_ov is not None
        canvas = np.where(canvas_ov[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)
        cards.append(canvas)

    # 枠線を描画
    if draw_frame:
        for card in cards:
            _draw_frame_of_card(width, height, card_shape, card)

    if show:
        win_names = []
        for i, card in enumerate(cards):
            win_name = f"#{i}"
            cv2.imshow(win_name, card)
            win_names.append(win_name)
        cv2.waitKey(show_wait_ms)

        for name in win_names:
            cv2.destroyWindow(name)

    return cards
