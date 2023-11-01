import csv
import glob
import math
import os
import random
import tempfile
from typing import Literal

import chardet
import cv2
import galois
import img2pdf
import numpy as np
import openpyxl
import pypdf
import tqdm
from PIL import Image, ImageDraw, ImageFont

from voronoi import cvt

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OVERLAP_VAL = 127


def is_prime(n: int) -> bool:
    """素数判定

    Args:
        n (int): 入力

    Returns:
        bool: 素数ならTrue, 非素数ならFalse
    """
    # 1以下の値は素数ではない
    if n <= 1:
        return False

    # 2と3は素数
    if n <= 3:
        return True

    # 2と3の倍数で割り切れる場合は素数ではない
    if n % 2 == 0 or n % 3 == 0:
        return False

    # 5から始めて、6k ± 1 (kは整数) の形を持つ数を調べる
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            # i か i + 2 で割り切れる場合、素数ではない
            return False
        i += 6

    # 上記の条件に該当しない場合、素数である
    return True


def is_prime_power(n: int) -> tuple[bool, tuple[int, int] | None]:
    """素数の累乗判定

    Args:
        n (int): 入力

    Returns:
        bool: 素数の累乗ならTrue
        Optional[tuple[int, int]]: n = a ** bを満たす(a, b) または None

    """
    if is_prime(n):
        return True, (n, 1)

    # nが任意の素数の累乗かをチェック
    n_s = int(np.floor(np.sqrt(n)))  # 2以上sqrt(n)以下の素数についてチェック
    for s in range(2, n_s + 1):
        if not is_prime(s):
            continue
        # 底をsとするlogを取り、結果が整数ならsの累乗
        v = float(np.log(n) / np.log(s))
        if v.is_integer():
            return True, (s, int(v))

    return False, None


def is_valid_n_symbols_per_card(n: int) -> bool:
    """カード1枚当たりのシンボル数が条件を満たすか

    条件: n が 2, あるいは「任意の素数の累乗 + 1」であること
    参考: カードゲーム ドブル(Dobble)の数理, https://amori.hatenablog.com/entry/2016/10/06/015906

    Args:
        n (int): 入力

    Returns:
        bool: 条件を満たせばTrue
    """
    if n < 2:
        return False
    elif n == 2:
        return True

    # 素数の累乗 + 1 か？
    flg, _ = is_prime_power(n - 1)

    return flg


def make_dobble_deck(n_symbols_per_card: int) -> tuple[list[list[int]], int]:
    """各カードに記載するシンボルの一覧を生成

    参考:
        * カードゲーム ドブル(Dobble)の数理, https://amori.hatenablog.com/entry/2016/10/06/015906
            * 日本語で一番わかりやすいサイト
                * カードの最大数の式などはここから流用
            * 素数の場合の組み合わせの求め方が載っているが、素数の累乗に非対応
        * The Dobble Algorithm, https://mickydore.medium.com/the-dobble-algorithm-b9c9018afc52
            * JavaScriptでの実装と図説があるが、素数の累乗に非対応
        * https://math.stackexchange.com/questions/1303497
            * "What is the algorithm to generate the cards in the game "Dobble" (known as "Spot it" in the USA)?"
            * 上記の Karinka 氏の実装を流用
            * 素数の累乗に対応するためにはガロア体上での計算が必要、との記載がある

    Args:
        n_symbols_per_card (int): カード1枚あたりに記載するシンボル数

    Returns:
        list[list[int]]: 各カードに記載するシンボル番号
        int: 全シンボル数
    """
    assert is_valid_n_symbols_per_card(n_symbols_per_card)
    n = n_symbols_per_card - 1

    # 位数を n とするガロア体
    # 0 以上 n 未満の正の整数で構成される世界のこと。
    # n が素数なら単純な加算、乗算が成り立つが、素数の累乗の場合は複雑な計算が必要。
    # n が素数の累乗ではないガロア体は存在しない。
    if n == 1:
        # 位数が1の場合は常に0を返すだけ
        gf = lambda _: 0  # noqa: E731
    else:
        # ガロア体(厳密には素数の累乗をベースとするためガロア拡大体)の計算をするためにgaloisパッケージを使う
        gf = galois.GF(n)

    pairs: list[list[int]] = []

    # 最初の N * N 枚のカード
    for i in range(n):
        for j in range(n):
            pairs.append([int(gf(i) * gf(k) + gf(j)) * n + k for k in range(n)] + [n * n + i])

    # 次の N 枚のカード
    for i in range(n):
        pairs.append([j * n + i for j in range(n)] + [n * n + n])

    # 最後の1枚
    pairs.append([n * n + i for i in range(n + 1)])

    n_cards = n_symbols_per_card * (n_symbols_per_card - 1) + 1
    assert len(pairs) == n_cards

    if __debug__:
        _n = len(pairs)
        for i in range(_n):
            # 各カードのシンボル数は指定数そろっている
            assert len(pairs[i]) == n_symbols_per_card
        for i in range(_n - 1):
            for j in range(i + 1, _n):
                # 任意のカードを2枚選んだ時、必ず共通するシンボルが1つだけ見つかる
                assert len(set(pairs[i]) & set(pairs[j])) == 1

    n_all_symbols = len(set(np.reshape(pairs, -1)))
    assert n_all_symbols == n_cards  # 全シンボル数は必要カード数と同じになる

    return pairs, n_all_symbols


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
    files: list[str] = [fname for e in ext for fname in glob.glob(f"{dir_name}/*.{e}")]
    if len(files) < num:
        # ファイル数が足りないので例外を返す
        raise ValueError(f"{dir_name}に{num}個以上の画像が存在しない")
    files: list[tuple[str, str]] = [(x, os.path.splitext(os.path.basename(x))[0]) for x in files]
    if shuffle:
        random.shuffle(files)
    else:
        # 試しにキャスト
        try:
            files: list[tuple[str, int]] = [(x[0], int(x[1])) for x in files]
        except ValueError:
            # キャスト失敗ならstrのままにする
            pass
        # ソート
        files = sorted(files, key=lambda x: x[1])
    # 必要数に画像を制限
    files = files[:num]

    # 画像読み込み
    images: list[np.ndarray] = []
    image_paths: list[str] = []

    for path, _ in files:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"{path} の読み込み失敗")
        h, w, n_ch = img.shape
        if n_ch == 4:
            # alphaチャネルがある場合、背景を白画素(255, 255, 255)として合成する
            img_bg = np.full((h, w, 3), WHITE, dtype=np.uint8)
            img = (img_bg * (1 - img[:, :, 3:] / 255) + img[:, :, :3] * (img[:, :, 3:] / 255)).astype(np.uint8)
        elif n_ch == 1:
            img = cv2.cvtColor(img, cv2.GRAY2BGR)
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


def rotate_fit(
    img: np.ndarray, angle: int, *, flags: int = cv2.INTER_CUBIC, borderValue: cv2.typing.Scalar = 0
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


def _make_canvas(is_circle: bool, width: int, height: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    """空のキャンバスを作成"""
    if is_circle:
        # 描画キャンバス
        canvas = np.full((height, width, 3), WHITE, dtype=np.uint8)
        # 重複確認用キャンバス
        canv_ol = np.full((height, width), OVERLAP_VAL, dtype=np.uint8)

        center = (int(height / 2), int(width / 2))
        radius = int(width / 2) - margin
        cv2.circle(canvas, center, radius, BLACK, thickness=-1)
        cv2.circle(canv_ol, center, radius, 0, thickness=-1)
    else:
        # 描画キャンバス
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # 重複確認用キャンバス
        canv_ol = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), WHITE, thickness=margin, lineType=cv2.LINE_4)
        cv2.rectangle(canv_ol, (0, 0), (width - 1, height - 1), OVERLAP_VAL, thickness=margin, lineType=cv2.LINE_4)

    return canvas, canv_ol


def _get_interpolation(scale: float):
    """cv2.resize の interpolation の値を決める"""
    downsize = cv2.INTER_AREA  # 縮小用
    upsize = cv2.INTER_CUBIC  # 拡大用

    return downsize if scale < 1 else upsize


def _layout_random(
    is_circle: bool, width: int, height: int, margin: int, images: list[np.ndarray], show: bool
) -> np.ndarray:
    """画像を重ならないようにランダムに配置する

    Args:
        is_circle (bool): 出力画像が円形か否か
        width (int): 出力画像の幅
        height (int): 出力画像の高さ
        margin (int): 出力画像の外縁につける余白サイズ
        images (list[np.ndarray]): 配置画像ソース
        show (bool): 計算中の画像を画面表示するならTrue
    """
    # 画像1枚当たりの基本サイズを指定
    n_img_in_card = len(images)
    img_base_size = int(np.ceil(max(height, width) / np.ceil(np.sqrt(n_img_in_card))))

    while True:
        # マスク画像に描画するパラメータ
        # キャンバスの作成
        canvas, canv_ol = _make_canvas(is_circle, width, height, margin)
        for img in images:
            # 元画像の余白を除去して外接矩形でトリミング
            im_bin = np.full((img.shape[0], img.shape[1]), OVERLAP_VAL, dtype=np.uint8)

            # 長辺を基本サイズに拡縮
            scale = img_base_size / max(img.shape[0], img.shape[1])
            im_base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=_get_interpolation(scale))
            im_bin_base = cv2.resize(im_bin, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            # ランダムにリサイズ、回転、配置し、重複するならパラメータを変えて最大n_max回やり直し
            # (n_max回試してもダメなら最初からやり直し)
            n_max = 100
            cur_canv = canvas.copy()
            cur_canv_ol = canv_ol.copy()
            ok = False
            for _ in range(n_max):
                _canv = cur_canv.copy()
                _canv_ol = cur_canv_ol.copy()
                # ランダムにリサイズ
                scale = random.uniform(0.5, 0.8)  # NOTE: 小さめの方が敷き詰めるのに時間がかからないが余白が増えるので適宜調整
                _im_scl = cv2.resize(im_base, None, fx=scale, fy=scale, interpolation=_get_interpolation(scale))
                _im_bin_scl = cv2.resize(im_bin_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

                # ランダムに回転
                angle = random.randint(0, 360)
                _im_rot = rotate_fit(_im_scl, angle, flags=cv2.INTER_CUBIC, borderValue=WHITE)  # 境界を白にしないと枠が残る
                _im_bin_rot = rotate_fit(_im_bin_scl, angle, flags=cv2.INTER_NEAREST)

                # ランダムに平行移動
                dy = random.randint(0, height - _im_rot.shape[0])
                dx = random.randint(0, width - _im_rot.shape[1])
                mv_mat = np.float32([[1, 0, dx], [0, 1, dy]])
                im_rnd = cv2.warpAffine(
                    _im_rot, mv_mat, (width, height), flags=cv2.INTER_CUBIC, borderValue=WHITE
                )  # 境界を白にしないと枠が残る
                im_bin_rnd = cv2.warpAffine(_im_bin_rot, mv_mat, (width, height), flags=cv2.INTER_NEAREST)

                # キャンバスに重畳 (マスク処理)
                _canv = np.where(im_bin_rnd[:, :, np.newaxis] != 0, im_rnd, _canv)
                _canv_ol += im_bin_rnd

                if show:
                    cv2.imshow("canv", _canv)
                    cv2.imshow("canv_overlap", _canv_ol)
                    cv2.waitKey(1)

                # 重なりの確認
                if (_canv_ol > OVERLAP_VAL).sum() == 0:
                    ok = True

                    canvas = _canv
                    canv_ol = _canv_ol
                    break

            if not ok:
                break
        if ok:
            # 最終キャンバスに重畳 (マスク処理)
            canvas = np.where(canv_ol[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)
            break

    if show:
        cv2.destroyAllWindows()

    return canvas


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


def _make_drawing_region_in_card(is_circle: bool, width: int, height: int, margin: int) -> list[tuple[float, float]]:
    """カード内の描画範囲を取得

    Args:
        is_circle (bool): True なら 円, False なら 長方形
        width (int): カードの幅
        height (int): カードの高さ (円なら無視)
        margin (int): カード外縁の余白

    Returns:
        list[tulpe[float, float]]: 描画範囲を凸包表現した際の座標リスト
    """
    if is_circle:
        c = width / 2
        r = width / 2 - margin
        bnd_pts = [(c + r * np.cos(x / 180.0 * np.pi), c + r * np.sin(x / 180.0 * np.pi)) for x in range(0, 360, 5)]
    else:
        bnd_pts = [
            (margin, margin),
            (width - margin, margin),
            (width - margin, height - margin),
            (margin, height - margin),
        ]
        bnd_pts = [(float(x), float(y)) for x, y in bnd_pts]  # cast

    return bnd_pts


def _layout_voronoi(
    is_circle: bool,
    width: int,
    height: int,
    margin: int,
    images: list[np.ndarray],
    show: bool,
    *,
    radius_p: float = 0.0,
    n_iters: int = 20,
) -> np.ndarray:
    """画像を重ならないように重心ボロノイ分割で決めた位置にランダムに配置する

    Args:
        is_circle (bool): 出力画像が円形か否か
        width (int): 出力画像の幅
        height (int): 出力画像の高さ (is_circleならwidthと同じであること)
        margin (int): 出力画像の外縁につける余白サイズ
        images (list[np.ndarray]): 配置画像ソース
        show (bool): 計算中の画像を画面表示するならTrue
        radius_p: 各母点の半径を決めるパラメータ, 0.0なら通常のボロノイ分割
        n_iters (int): 重心ボロノイ分割の反復回数
    """
    assert not is_circle or (is_circle and width == height)

    # 各カードの配置位置と範囲を計算 (x, yで計算)
    bnd_pts = _make_drawing_region_in_card(is_circle, width, height, margin)

    # 重心ボロノイ分割で各画像の中心位置と範囲を取得
    # (範囲は目安であり変わっても良い)
    # NOTE: ここのshowは毎回止まってしまうのでデバッグ時に手動でTrueにする
    repeat = True
    n = 0
    while repeat:
        # radius_p を設定した場合に初期値によって例外が生じることがある(0.0指定時は生じたことはない)ので、その場合はやり直す
        repeat = False
        try:
            pos_images, rgn_images = cvt(bnd_pts, len(images), radius_p=radius_p, n_iters=n_iters, show_step=None)
        except Exception:
            repeat = True
        n += 1  # 繰り返し回数を設定
        if n > 5:  # 一定回数以上試してもダメならパラメータ指定が良くないので例外を吐く
            raise ValueError(f"ボロノイ分割が失敗 (radius_p({radius_p:.2f})が大きすぎる可能性が高い)")

    # キャンバスの作成
    # canvas:
    #   最終的に出力する画像 (3ch)
    #   シンボルの画像位置が決まるごとに重畳され、最後に余白(canvas_olで0の箇所)がWHITEで塗りつぶされる
    # canvas_ol:
    #   重なりチェックをするためのマスク画像 (1ch),
    #   描画後に(0またはOVERLAP_VAL)以外の値があったら、その画素は重なりがあったことを意味する
    canvas, canv_ol = _make_canvas(is_circle, width, height, margin)

    # 各画像をpos_imagesに配置
    for i_img, img in enumerate(images):
        # 元画像の余白を除去して外接矩形でトリミング
        im_trim = _trim_bb_image(img)
        im_h = im_trim.shape[0]
        im_w = im_trim.shape[1]

        pos = pos_images[i_img]  # ボロノイ領域の重心 (描画の中心座標とする) (x, y)
        rgn = rgn_images[i_img]  # ボロノイ領域境界 (x, y)
        # 貼り付ける画像の最大サイズは、ボロノイ領域の最大長に長辺が入るサイズとする
        mx_len_rgn = max([np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in rgn for p2 in rgn])

        init_deg = random.randint(0, 360)  # 初期角度をランダムに決める

        ok = False
        scl = 100
        while scl > 5:
            # 元画像の対角長とボロノイ領域の最大長が一致するスケールを100%としてスケールを計算
            l_lngside = max(im_trim.shape[0], im_trim.shape[1])  # 元画像の長辺長
            scl_r = mx_len_rgn / l_lngside * (scl / 100)
            for deg in range(0, 180, 10):  # 画像を少しずつ回転 (矩形で試しているので180度までの確認で良い)
                # 画像と同サイズの矩形をキャンバスに重畳描画
                lt = _rotate_2d((-im_w * scl_r / 2, -im_h * scl_r / 2), deg + init_deg)
                rt = _rotate_2d((+im_w * scl_r / 2, -im_h * scl_r / 2), deg + init_deg)
                lb = _rotate_2d((+im_w * scl_r / 2, +im_h * scl_r / 2), deg + init_deg)
                rb = _rotate_2d((-im_w * scl_r / 2, +im_h * scl_r / 2), deg + init_deg)

                lt = (lt[0] + pos[0], lt[1] + pos[1])
                rt = (rt[0] + pos[0], rt[1] + pos[1])
                lb = (lb[0] + pos[0], lb[1] + pos[1])
                rb = (rb[0] + pos[0], rb[1] + pos[1])

                # 重なりなく描画できるか確認
                # 画像貼り付け位置にマスクを描画
                mask_img = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
                _pts = np.array([lt, rt, lb, rb], dtype=int)
                cv2.fillConvexPoly(mask_img, _pts, OVERLAP_VAL, lineType=cv2.LINE_4)

                # ボロノイ境界を超えない制約をつけるために境界線を描画
                # polylines (やfillPoly) は[(x0, y0), (x1, y1), ...]を以下の形にreshapeしないと動かない
                # 参考: https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
                _canv_ol = canv_ol.copy()  # ボロノイ境界はmarginなど他のマスクと重なることがあるので、重畳ではなくOVERLAP_VALの描画にする
                _pts = np.array(rgn).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(_canv_ol, [_pts], True, OVERLAP_VAL, thickness=1, lineType=cv2.LINE_4)

                # 画像マスクとボロノイ境界マスクを重畳
                _canv_ol += mask_img

                if show:
                    cv2.imshow("canv_overlap", _canv_ol)
                    cv2.waitKey(1)

                # 重なりの確認
                if (_canv_ol > OVERLAP_VAL).sum() == 0:
                    ok = True

                    # 重なりがなければ改めてマスクに画像マスクを重畳
                    canv_ol += mask_img

                    # キャンバスに重畳 (マスク処理)
                    im_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
                    _pts = np.array([lt, rt, lb, rb], dtype=int)
                    cv2.fillConvexPoly(im_mask, np.array(_pts), OVERLAP_VAL, lineType=cv2.LINE_4)

                    _im_scl = cv2.resize(im_trim, None, fx=scl_r, fy=scl_r, interpolation=_get_interpolation(scl_r))
                    _im_rot = rotate_fit(_im_scl, -(deg + init_deg), flags=cv2.INTER_CUBIC, borderValue=WHITE)
                    dx = pos[0] - _im_rot.shape[1] / 2
                    dy = pos[1] - _im_rot.shape[0] / 2
                    mv_mat = np.float32([[1, 0, dx], [0, 1, dy]])
                    im_rnd = cv2.warpAffine(_im_rot, mv_mat, (width, height), flags=cv2.INTER_CUBIC, borderValue=WHITE)
                    canvas = np.where(im_mask[:, :, np.newaxis] != 0, im_rnd, canvas)

                    if show:
                        cv2.imshow("canvas", canvas)
                        cv2.waitKey(1)

                    break
            if ok:
                break

            scl *= 0.95  # 前回の大きさを基準に一定割合だけ縮小

    if not ok:
        # ここでOKになっていないということは画像を限界まで小さくしてもボロノイ領域に収まらなかったことを意味するので例外とする
        raise NotImplementedError("画像がボロノイ領域内に描画できずレイアウトに失敗")

    if show:
        cv2.destroyAllWindows()

    # 最終キャンバスに重畳 (マスク処理)
    canvas = np.where(canv_ol[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)

    return canvas


def layout_images_randomly_wo_overlap(
    images: list[np.ndarray],
    image_indexes: list[int],
    canv_size: int | tuple[int, int],
    margin: int,
    *,
    method: Literal["random", "voronoi"] = "random",
    radius_p: float = 0.0,
    n_voronoi_iters: int = 20,
    draw_frame: bool = False,
    show: bool = False,
) -> np.ndarray:
    """画像を重ならないように配置する

    Args:
        images (list[np.ndarray]): 画像リスト
        image_indexes (list[int]): 使用する画像のインデックスのリスト
        canv_size (Union[int, tuple[int, int]]):
            配置先の画像サイズ. intなら円の直径、tuple[int, int]なら矩形の幅, 高さとする
        margin (int): 配置先の画像の外縁につける余白サイズ
        method: レイアウト方法
            "random": ランダム配置
            "voronoi": 重心ボロノイ分割に基づき配置
        radius_p: methodが"voronoi"の場合に各母点の半径を決めるパラメータ, 0.0なら通常のボロノイ分割
        n_voronoi_iters: method == "voronoi"の場合の反復回数
        draw_frame (bool): 印刷を想定した枠を描画するならTrue
        show (bool): (optional) 計算中の画像を画面表示するならTrue

    Returns:
        np.ndarray: カード画像
    """
    tar_images = [images[i] for i in image_indexes]

    # 出力先画像を初期化
    is_circle: bool = False
    if isinstance(canv_size, int):
        width = height = canv_size
        is_circle = True
    else:
        width, height = canv_size

    if method == "random":
        canvas = _layout_random(is_circle, width, height, margin, tar_images, show)
    elif method == "voronoi":
        canvas = _layout_voronoi(
            is_circle,
            width,
            height,
            margin,
            tar_images,
            show,
            radius_p=radius_p,
            n_iters=n_voronoi_iters,
        )
    else:
        raise ValueError(f"method ('{method}') の指定ミス")

    if draw_frame:
        gray = (127, 127, 127)
        if is_circle:
            center = (int(height / 2), int(width / 2))
            radius = int(width / 2)
            cv2.circle(canvas, center, radius, gray, thickness=1)
        else:
            cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), gray, thickness=1)

    return canvas


def merge_pdf(pdf_paths: list[str], output_pdf: str):
    """PDFファイルをマージして新たなPDFファイルを出力

    Args:
        pdf_paths (list[str]): マージ元のPDFファイルパス, 記載の順序で結合される
        output_pdf (str): 出力PDFファイルのパス
    """
    output_dir = os.path.dirname(output_pdf)
    os.makedirs(output_dir, exist_ok=True)

    merger = pypdf.PdfMerger()
    for p in pdf_paths:
        merger.append(p)

    merger.write(output_pdf)
    merger.close()


def images_to_pdf(
    images: list[np.ndarray],
    pdf_path: str,
    *,
    dpi: int = 300,
    card_long_side_mm: int = 95,
    width_mm: int = 210,
    height_mm: int = 297,
):
    """画像セットを並べてPDF化し保存

    Args:
        images (list[np.ndarray]): 画像セット, すべて画像サイズは同じとする
        pdf_path (str): 出力するPDFのフルパス
        dpi (int, optional): PDFの解像度. Defaults to 300.
        card_width_mm (int, optional): 画像1枚の長辺サイズ(mm). Defaults to 95.
        width_mm (int, optional): PDFの幅(mm). Defaults to 210 (A4縦).
        height_mm (int, optional): PDFの高さ(mm). Defaults to 297 (A4縦).
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
            pos_y_next = max(pos_y_next, pos_y + resize_h)
            pos_x += resize_w
        else:
            # 収まらないなら改行してみる
            pos_x = 0
            pos_y = pos_y_next
            if pos_y + resize_h < height:
                # 収まるなら貼り付け
                canvas[pos_y : (pos_y + resize_h), pos_x : (pos_x + resize_w), :] = resize_card
                pos_x += resize_w
            else:
                # 収まらないならPDF出力してから次のキャンバスの先頭に描画
                tmp_name = tmpdir.name + os.sep + f"{i_pdf}.png"
                pdf_name = tmpdir.name + os.sep + f"{i_pdf}.pdf"
                cv2.imwrite(tmp_name, canvas)
                with open(pdf_name, "wb") as f:
                    f.write(img2pdf.convert(tmp_name, layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))))

                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
                pos_x = pos_y = pos_y_next = 0
                canvas[pos_y : (pos_y + resize_h), pos_x : (pos_x + resize_w), :] = resize_card
                pos_x += resize_w

                i_pdf += 1

    # 最後の1枚が残っていたらPDF出力
    if (canvas != 255).any():
        tmp_name = tmpdir.name + os.sep + f"{i_pdf}.png"
        pdf_name = tmpdir.name + os.sep + f"{i_pdf}.pdf"
        cv2.imwrite(tmp_name, canvas)
        with open(pdf_name, "wb") as f:
            f.write(img2pdf.convert(tmp_name, layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))))

        n_pdf = i_pdf + 1
    else:
        n_pdf = i_pdf

    # すべてのPDFを1つのPDFにまとめて結合
    pdfs = [tmpdir.name + os.sep + f"{i}.pdf" for i in range(n_pdf)]
    merge_pdf(pdfs, pdf_path)

    return


def cv2_putText(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: str | tuple[str, int],
    font_scale: None | int,
    color: str | tuple,
    anchor: str,
    *,
    text_w: None | int = None,
    text_h: None | int = None,
    align: Literal["left", "center", "right"] = "left",
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
        font:
            str: フォントファイルパス (*.ttf | *.ttc)
            int:
                フォントファイルがttcの場合は複数フォントを含む。
                その中で使用するフォントのインデックスを指定する。
                フォント名は以下で確認が可能
                    ImageFont.truetype(font=ttcファイルパス, index=インデックス).getname()
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
        max_text_h:
        align:
            text に複数行(改行文字"\n"を含む文字列)を指定した場合の描画方法
            "left": 左揃え (default)
            "center": 中央揃え
            "right": 右揃え
        stroke_width: 文字枠の太さ
        stroke_fill: 文字枠の色
    """
    assert len(anchor) == 2

    if isinstance(font, str):
        font_face = font
        index = 0
    else:
        font_face, index = font

    # テキスト描画域を取得
    x, y = org
    if font_scale is not None:
        fontPIL = ImageFont.truetype(font=font_face, size=font_scale, index=index)
        dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
        xL, yT, xR, yB = dummy_draw.multiline_textbbox(
            (x, y), text, font=fontPIL, align=align, stroke_width=stroke_width
        )
    else:
        assert type(text_w) is int or type(text_h) is int
        font_scale = 1
        while True:
            fontPIL = ImageFont.truetype(font=font_face, size=font_scale, index=index)
            dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
            xL, yT, xR, yB = dummy_draw.multiline_textbbox(
                (0, 0), text, font=fontPIL, align=align, stroke_width=stroke_width
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
        fontPIL = ImageFont.truetype(font=font_face, size=font_scale, index=index)
        dummy_draw = ImageDraw.Draw(Image.new("L", (0, 0)))
        xL, yT, xR, yB = dummy_draw.multiline_textbbox(
            (x, y), text, font=fontPIL, align=align, stroke_width=stroke_width
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
    draw.text((x0 - x1, y0 - y1), text, color, fontPIL, align=align, stroke_width=stroke_width, stroke_fill=stroke_fill)
    roi = np.array(roiPIL, dtype=np.uint8)
    img[y1:y2, x1:x2] = roi


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

    font = (os.environ["windir"] + r"\fonts\uddigikyokashon-b.ttc", 2)

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
        stroke_width=3,
        stroke_fill=(255, 255, 255),
        text_w=canv.shape[1],
        text_h=int(np.floor(canv.shape[0] * text_h_rate)),
    )

    return canv


def make_image_of_thumbnails_with_names(
    card_size: int | tuple[int, int],
    margin: int,
    table_size: tuple[int, int],
    images: list[np.ndarray],
    names: list[str],
    *,
    thumb_margin: float = 0.05,
    draw_frame: bool = False,
    show: bool = False,
    show_wait_ms: int = 0,
) -> list[np.ndarray]:
    """シンボル画像のサムネイルと名前を一覧するカード画像を作成

    Args:
        is_circle (bool): 出力画像が円形か否か
        width (int): 出力画像の幅
        height (int): 出力画像の高さ (is_circleならwidthと同じであること)
        margin (int): 出力画像の外縁につける余白サイズ
        table_size (tuple[int, int]): サムネイルを表形式で並べる際の表の(行数, 列数)
        images (list[np.ndarray]): 配置画像ソース
        names (list[str]): images毎の表示名
        thumb_margin (float):
            出力画像の描画領域を表サイズで分割した領域(セル)において、
            さらにサムネイルの周囲につける余白のセルサイズに対する割合 (0以上0.5未満)
        draw_frame (bool): 印刷を想定した枠を描画するならTrue
        show (bool): 計算中の画像を画面表示するならTrue
        show_wait_ms (int): show is True 時に表示するウィンドウの待ち時間, 0ならキー入力

    Returns:
        list[np.ndarray]:
            サムネイルを並べたカード画像のリスト
    """
    if isinstance(card_size, int):
        is_circle = True
        width = height = card_size
    else:
        assert isinstance(card_size, tuple) and len(card_size) == 2
        is_circle = False
        width, height = card_size

    assert not is_circle or (is_circle and width == height)
    assert len(images) == len(names)
    assert 0.0 <= thumb_margin < 0.5

    # 円の場合、端にはシンボルを描画できないので少し小さめのサイズを描画範囲とする
    table_scale = 0.98 if is_circle else 1.0
    table_x0 = margin + width * (1 - table_scale) / 2
    table_y0 = margin + height * (1 - table_scale) / 2

    # テーブルのセルサイズを計算
    cell_w = table_scale * (width - 2 * margin) / table_size[1]
    cell_h = table_scale * (height - 2 * margin) / table_size[0]

    # サムネイルのサイズを計算
    _thumb_w = cell_w - 2 * thumb_margin * cell_w
    _thumb_h = cell_h - 2 * thumb_margin * cell_h

    margin_thumb_w = (cell_w - _thumb_w) / 2
    margin_thumb_h = (cell_h - _thumb_h) / 2
    thumb_w = int(_thumb_w)
    thumb_h = int(_thumb_h)

    # 各画像 + テキストをレンダリングした画像を生成
    thumbs: list[np.ndarray] = []
    for symbl_img, symbl_name in zip(images, names):
        thumb = _draw_name_in_thumb(symbl_img, thumb_w, thumb_h, symbl_name, text_h_rate=0.25)
        thumbs.append(thumb)

    # 各画像をカードにテーブル形式で描画
    cards: list[np.ndarray] = []

    canvas = None
    pos_x = pos_y = 0  # 描画位置
    cnt_img_in_card = 0  # カードに描画された画像数 (エラーチェック用に使う)
    for thumb_img in thumbs:
        ok = False
        while not ok:
            if canvas is None:
                canvas, canvas_ov = _make_canvas(is_circle, width, height, margin)
            # 描画位置に入るか確認
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
                cv2.imshow("canvas", canvas)
                cv2.imshow("canvas_ov", canvas_ov)
                cv2.imshow("try", try_canv_ov)
                cv2.waitKey(show_wait_ms)

            # 端がギリギリで入りにくいので、適当に重なりを許容
            n_ov_pixs = (try_canv_ov > OVERLAP_VAL).sum() - (canvas_ov > OVERLAP_VAL).sum()
            if n_ov_pixs <= max(thumb_w, thumb_h):
                # 描画可能なので描画
                canvas_ov = try_canv_ov
                canvas[
                    y0_m : (y0_m + thumb_h),
                    x0_m : (x0_m + thumb_w),
                    :,
                ] = thumb_img
                ok = True
                cnt_img_in_card += 1

                if show:
                    cv2.imshow("canvas", canvas)
                    cv2.waitKey(show_wait_ms)

            # 描画できてもできなくても次の位置へ
            pos_x += 1
            if pos_x >= table_size[1]:
                pos_x = 0
                pos_y += 1
            if pos_y >= table_size[0]:
                # このカードがいっぱいだったら、カードを出力してから次の画像へ
                if cnt_img_in_card == 0:
                    # 1個も描画できないまま次の画像へ行ってしまったら、テーブルサイズの設定ミスなので例外送出
                    raise ValueError("カードにサムネイルが描画できないのでテーブルサイズの調整が必要")
                canvas = np.where(canvas_ov[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)
                cards.append(canvas)
                canvas = None
                cnt_img_in_card = 0
                pos_x = 0
                pos_y = 0

    if cnt_img_in_card > 0:
        # 未出力のカード画像を処理
        assert canvas is not None
        canvas = np.where(canvas_ov[:, :, np.newaxis] == 0, WHITE, canvas).astype(np.uint8)
        cards.append(canvas)

    # 枠線を描画
    if draw_frame:
        gray = (127, 127, 127)
        if is_circle:
            center = (int(height / 2), int(width / 2))
            radius = int(width / 2)
            for card in cards:
                cv2.circle(card, center, radius, gray, thickness=1)
        else:
            for card in cards:
                cv2.rectangle(card, (0, 0), (width - 1, height - 1), gray, thickness=1)

    if show:
        for i, card in enumerate(cards):
            cv2.imshow(f"#{i}", card)
        cv2.waitKey(show_wait_ms)

        cv2.destroyAllWindows()

    return cards


def detect_encoding(file_path: str) -> str:
    """テキストファイルのencodingを推定"""
    with open(file_path, "rb") as f:
        data = f.read()

    return chardet.detect(data)["encoding"]


def load_image_list(image_list_path: str) -> dict[str, str]:
    """画像リストの読み込み

    Args:
        image_list_path (str):
            画像リストファイルパス (.xlsx|.csv)
            * "ファイル名"列, 及び"名前"列を持つこと（列名は先頭行）
            * "ファイル名": 拡張子を除く画像ファイル名
            * "名前": カードに描画する表示名

    Returns:
        dict[str, str]:
            key: ファイル名
            value: 名前
    """
    # 画像リストの読み込み
    # "ファイル名"列: 拡張子を除くファイル名
    # "名前"列: カードに描画する

    data_list: dict[str, str] = dict()  # key: "ファイル名", value: "名前", を持つdict

    ext = os.path.splitext(image_list_path)[1]
    if ext == ".xlsx":
        # xlsx読み込み
        wb = openpyxl.load_workbook(image_list_path, read_only=True)
        sheet = wb.active
        # "ファイル名"と"名前"の列のインデックスを取得
        file_name_index = None
        name_index = None

        for col, header in enumerate(sheet[1], 1):
            if header.value == "ファイル名":
                file_name_index = col
            elif header.value == "名前":
                name_index = col

        # データを取得
        for row in sheet.iter_rows(values_only=True, min_row=2):  # 先頭行は除く
            if file_name_index is not None and name_index is not None:
                data_list[row[file_name_index - 1]] = row[name_index - 1]

        wb.close()
    elif ext == ".csv":
        # 文字コード判定
        encoding = detect_encoding(image_list_path)
        with open(image_list_path, newline="", encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                file_name = row.get("ファイル名")
                name = row.get("名前")

                if file_name and name:
                    data_list[file_name] = name
    else:
        raise NotImplementedError(f"拡張子 {ext} は未対応")

    return data_list


def sort_images_by_image_list(
    images: list[np.ndarray], image_paths: list[str], image_names: dict[str, str]
) -> tuple[list[np.ndarray], list[str]]:
    """画像と画像名を辞書に記載の順序でソートする

    Args:
        images (list[np.ndarray]): 入力画像
        image_paths (list[str]): 各入力画像のファイルパス
        image_names (dict[str, str]):
            key: 拡張子を除く画像ファイル名, image_paths(の拡張子除くファイル名)がすべてここに含まれること
            value: 表示名

    Returns:
        list[np.ndarray]: image_namesに記載の順序でソートされたimages
        list[str]: 対応する表示名
    """
    assert len(images) == len(image_paths)
    image_bases = [os.path.basename(os.path.splitext(x)[0]) for x in image_paths]

    # 使用された画像ファイルがすべて画像名リストに記載されているかチェック
    for img_base in image_bases:
        if img_base not in image_names.keys():
            raise ValueError(f"{img_base} が画像リストの'ファイル名'列に存在しない")

    # images, image_pathsをimage_namesに記載の順序で並び替え
    sorted_images: list[np.ndarray] = list()
    sorted_names: list[str] = list()
    for base, name in image_names.items():
        if base in image_bases:
            p = image_bases.index(base)
            sorted_images.append(images[p].copy())
            sorted_names.append(name)

    assert len(sorted_images) == len(sorted_names) == len(images) == len(image_paths)

    return sorted_images, sorted_names


def main():
    # ============
    # パラメータ設定
    # ============
    # ファイル名
    image_dir = "samples"  # 入力画像ディレクトリ
    output_dir = "output"  # 出力画像ディレクトリ
    pdf_name = "card.pdf"  # 出力するPDF名
    # カードの設定
    n_symbols_per_card: int = 5  # カード1枚当たりのシンボル数
    card_img_size = 1500  # カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
    card_margin = 20  # カード1枚の余白サイズ [pix]
    layout_method: Literal["random", "voronoi"] = "voronoi"  # random: ランダム配置, voronoi: 重心ボロノイ分割に基づき配置
    radius_p: float = 0.5  # "voronoi"における各母点の半径を決めるパラメータ (0.0なら半径なし, つまり通常のボロノイ分割として処理)
    n_voronoi_iters = 10  # "voronoi"における反復回数
    # PDFの設定
    dpi = 300  # 解像度
    card_size_mm = 95  # カードの長辺サイズ[mm]
    page_size_mm = (210, 297)  # PDFの(幅, 高さ)[mm]
    # 画像リストの設定
    image_list_path: None | str = r"samples\画像リスト.xlsx"  # xlsx | csv のパス
    image_table_size: tuple[int, int] = (8, 6)  # 画像リストの表サイズ (行数, 列数)

    # その他
    shuffle: bool = False  # True: 画像読み込みをシャッフルする
    seed: int | None = 0  # 乱数種
    gen_card_images: bool = True  # (主にデバッグ用) もし output_dir にある生成済みの画像群を使うならFalse

    # ========
    # 前処理
    # ========
    # 入力チェック
    if not is_valid_n_symbols_per_card(n_symbols_per_card):
        raise ValueError(f"カード1枚当たりのシンボル数 ({n_symbols_per_card}) が「2 あるいは (任意の素数の累乗 + 1)」ではない")

    # 乱数初期化
    random.seed(seed)
    np.random.seed(seed)

    # ========
    # メイン
    # ========
    # 各カード毎の組み合わせを生成
    pairs, n_symbols = make_dobble_deck(n_symbols_per_card)

    # image_dirからn_symbols数の画像を取得
    images, image_paths = load_images(image_dir, n_symbols, shuffle=shuffle)

    # 出力フォルダ作成
    os.makedirs(output_dir, exist_ok=True)

    # len(pairs)枚のカード画像を作成し、保存
    card_images = []
    for i, image_indexes in enumerate(tqdm.tqdm(pairs, desc="layout images")):
        path = output_dir + os.sep + f"{i}.png"

        if gen_card_images:
            card_img = layout_images_randomly_wo_overlap(
                images,
                image_indexes,
                card_img_size,
                card_margin,
                draw_frame=True,
                method=layout_method,
                radius_p=radius_p,
                n_voronoi_iters=n_voronoi_iters,
            )
            cv2.imwrite(path, card_img)
        else:
            card_img = cv2.imread(path)
            assert card_img is not None  # 必ず読み込める前提

        card_images.append(card_img)

    # 画像リストファイルの指定があれば画像リストカード画像を作成
    image_names = load_image_list(image_list_path)
    sorted_images, sorted_names = sort_images_by_image_list(
        images, image_paths, image_names
    )  # image_namesの順序でimage_pathsをソート
    thumbs_cards = make_image_of_thumbnails_with_names(
        card_img_size,
        card_margin,
        image_table_size,
        sorted_images,
        sorted_names,
        draw_frame=True,
    )  # 画像をサムネイル化したカード画像を作成
    for i, card in enumerate(thumbs_cards):
        path = output_dir + os.sep + f"thumbnail_{i}.png"
        cv2.imwrite(path, card)
        card_images.append(card)

    # 各画像をA4 300 DPIに配置しPDF化
    images_to_pdf(
        card_images,
        output_dir + os.sep + pdf_name,
        dpi=dpi,
        card_long_side_mm=card_size_mm,
        width_mm=page_size_mm[0],
        height_mm=page_size_mm[1],
    )

    return


if __name__ == "__main__":
    main()
