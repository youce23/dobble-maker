import glob
import os
import random
import tempfile

import cv2
import galois
import img2pdf
import numpy as np
import pypdf
import tqdm


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


def make_symbol_combinations(n_symbols_per_card: int) -> tuple[list[list[int]], int]:
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
    if n == 1:
        a = b = 1
    else:
        _, (a, b) = is_prime_power(n)  # n を素数と指数に分解
    assert n == a**b

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

    # 最初の1枚
    pairs.append([n * n + i for i in range(n + 1)])

    n_cards = n_symbols_per_card * (n_symbols_per_card - 1) + 1
    assert len(pairs) == n_cards  # 参考サイトを参照

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
    dir_name: str, num: int, *, ext: list[str] = ["jpg", "png"], shuffle: bool = False
) -> tuple[list[np.ndarray], list[str]]:
    """所定数の画像を読み込む

    画像ファイル名がすべて整数表記であれば数値でソートする。
    整数表記でないファイルが1つでもあれば、文字列でソートする。

    Args:
        dir_name (str): 画像フォルダ
        num (int): 読み込む画像数
        ext (list[str], optional): 読み込み対象画像の拡張子. Defaults to ["jpg", "png"].
        shuffle (bool, optional): Trueなら画像ファイル一覧をシャッフルする. Defaults to False.

    Returns:
        list[np.ndarray]: 読み込んだnum個の画像のリスト
        list[str]]: 各画像のファイルパス
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

    WHITE = (255, 255, 255)
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
        else:
            raise IOError(f"{path} のチャネル数({n_ch})はサポート対象外")
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


def layout_images_randomly_wo_overlap(
    images: list[np.ndarray],
    image_indexes: list[int],
    canv_size: int | tuple[int, int],
    margin: int,
    *,
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

    # 画像1枚当たりの基本サイズを指定
    n_img_in_card = len(image_indexes)
    img_base_size = int(np.ceil(max(height, width) / np.ceil(np.sqrt(n_img_in_card))))

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    while True:
        # マスク画像に描画するパラメータ
        v = 127  # 二値化する値。重複チェックのために中間の値にしておく
        thr = 5  # 二値化の閾値
        # キャンパスの作成
        if is_circle:
            # 描画キャンバス
            canvas = np.full((height, width, 3), WHITE, dtype=np.uint8)
            # 重複確認用キャンバス
            canv_ol = np.full((height, width), v, dtype=np.uint8)

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
            cv2.rectangle(canv_ol, (0, 0), (width - 1, height - 1), v, thickness=margin, lineType=cv2.LINE_4)
        for img in tar_images:
            # 元画像を二値化して外接矩形でトリミング
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, im_bin = cv2.threshold(im_gray, 255 - thr, v, cv2.THRESH_BINARY_INV)
            wh = np.where(im_bin == v)
            y0 = min(wh[0])
            x0 = min(wh[1])
            y1 = max(wh[0])
            x1 = max(wh[1])

            im_trim = img[y0:y1, x0:x1, :]
            im_bin_trim = np.full((im_trim.shape[0], im_trim.shape[1]), v, dtype=np.uint8)

            # 長辺を基本サイズに拡縮
            scale = img_base_size / max(im_bin.shape[0], im_bin.shape[1])
            im_base = cv2.resize(im_trim, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            im_bin_base = cv2.resize(im_bin_trim, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

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
                _im_scl = cv2.resize(im_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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
                if (_canv_ol > v).sum() == 0:
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
        resize_card = cv2.resize(card, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

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


def main():
    # ============
    # パラメータ設定
    # ============
    # ファイル名
    image_dir = "images"  # 入力画像ディレクトリ
    output_dir = "output"  # 出力画像ディレクトリ
    pdf_name = "card.pdf"  # 出力するPDF名
    # カードの設定
    n_symbols_per_card: int = 8  # カード1枚当たりのシンボル数
    card_img_size = 1500  # カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
    card_margin = 20  # カード1枚の余白サイズ [pix]
    # PDFの設定
    dpi = 300  # 解像度
    card_size_mm = 95  # カードの長辺サイズ[mm]
    page_size_mm = (210, 297)  # PDFの(幅, 高さ)[mm]

    # その他
    seed: int | None = None  # 乱数種
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
    pairs, n_symbols = make_symbol_combinations(n_symbols_per_card)

    # image_dirからn_symbols数の画像を取得
    images, _ = load_images(image_dir, n_symbols)

    # len(pairs)枚のカード画像を作成し、保存
    os.makedirs(output_dir, exist_ok=True)

    card_images = []
    for i, image_indexes in enumerate(tqdm.tqdm(pairs, desc="layout images")):
        path = output_dir + os.sep + f"{i}.png"

        if gen_card_images:
            card_img = layout_images_randomly_wo_overlap(
                images, image_indexes, card_img_size, card_margin, draw_frame=True
            )
            cv2.imwrite(path, card_img)
        else:
            card_img = cv2.imread(path)
            assert card_img is not None  # 必ず読み込める前提

        card_images.append(card_img)

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