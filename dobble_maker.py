import csv
import json
import os
import random
from typing import Literal

import chardet
import numpy as np
import openpyxl
import tqdm

from card_drawer.draw_card import (
    CARD_SHAPE,
    images_to_pdf,
    layout_images_randomly_wo_overlap,
    load_images,
    make_image_of_thumbnails_with_names,
)
from cv2_image_utils import imread_japanese, imwrite_japanese
from deck_generator.generator import make_dobble_deck


class FileFormatError(Exception):
    # ファイルフォーマットに関するエラー
    pass


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
            * "名前": カードに描画する表示名, テキストで「\n」と記載がある場合は改行文字に変換する

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
        wb = openpyxl.load_workbook(image_list_path, read_only=True, data_only=True)
        sheet = wb.active
        # "ファイル名"と"名前"の列のインデックスを取得
        file_name_index = None
        name_index = None

        for col, header in enumerate(sheet[1], 1):
            if header.value == "ファイル名":
                file_name_index = col
            elif header.value == "名前":
                name_index = col

        # 両方がそろっていなければファイルのフォーマットエラー
        if file_name_index is None:
            raise FileFormatError(f"'{image_list_path}'に'ファイル名'列が存在しない")
        elif name_index is None:
            raise FileFormatError(f"'{image_list_path}'に'名前'列が存在しない")

        # データを取得
        for row in sheet.iter_rows(values_only=True, min_row=2):  # 先頭行は除く
            if file_name_index is not None and name_index is not None:
                _fname = row[file_name_index - 1]
                _name = row[name_index - 1]
                # 空でもいったん読み込んでおく (csvと挙動を合わせるため)
                file_name = _fname if _fname is not None else ""
                name = _name if _name is not None else ""

                data_list[file_name] = name

        wb.close()
    elif ext == ".csv":
        # 文字コード判定
        encoding = detect_encoding(image_list_path)
        with open(image_list_path, newline="", encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                file_name = row.get("ファイル名")
                name = row.get("名前")
                if file_name is None:
                    raise FileFormatError(f"{image_list_path}に'ファイル名'列が存在しない")
                elif name is None:
                    raise FileFormatError(f"{image_list_path}に'名前'列が存在しない")

                if file_name is not None and name is not None:
                    data_list[file_name] = name
    else:
        raise NotImplementedError(f"拡張子 {ext} は未対応")

    # 空は許容しない
    for file_name, name in data_list.items():
        if file_name == "" and name != "":
            raise FileFormatError(f"'{image_list_path}'の'{name}'のファイル名が空")
        elif file_name != "" and name == "":
            raise FileFormatError(f"'{image_list_path}'の'{file_name}'の名前が空")
        elif file_name == "" and name == "":
            raise FileFormatError(f"'{image_list_path}'のファイル名, 名前が空の行がある")

    # "名前"列の改行文字変換
    data_list = {key: val.replace("\\n", "\n") for key, val in data_list.items()}

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


def save_card_list_to_csv(
    output_dir: str,
    pairs: list[list[int]],
    *,
    image_paths: list[str] | None = None,
    image_names: dict[str, str] | None = None,
):
    """カード毎のID, 画像ファイル一覧, カード毎の画像ファイル, のcsvをそれぞれ出力

    Args:
        output_dir: 出力先ディレクトリ
        pairs: 各カードに記載するシンボル番号
        image_paths:
            シンボル画像ファイルパス.
            pairsのシンボル番号に対応するインデックスで格納されていること.
        image_names:
            シンボル画像名. image_pathsの拡張子を除くファイル名をキー、表示名をバリューとする.
            指定時は必ず image_paths も指定すること.
    """
    if image_names is not None and image_paths is None:
        raise ValueError("image_names 指定時は image_paths を必ず指定する")

    if image_names is not None:
        # image_namesにある"\n"は改行ではなくそのまま文字列として出力できるように修正
        image_names = {k: v.replace("\n", "\\n") for k, v in image_names.items()}

    # 各カードのIDのcsv
    _path = os.path.join(output_dir, "pairs.csv")
    try:
        with open(_path, "w") as f:
            f.write("\n".join([",".join([str(x) for x in row]) for row in pairs]))
    except Exception as e:
        raise type(e)(f"{_path} の保存に失敗") from e

    # 使用された画像ファイル一覧のcsv
    if image_paths is not None:
        _path = os.path.join(output_dir, "images.csv")
        try:
            with open(_path, "w", encoding="utf_8_sig") as f:
                # "画像名"には任意の文字が入る可能性があるためエスケープできるようにcsv.writerを使う
                writer = csv.writer(f, lineterminator="\n")

                header = ["ID", "画像ファイル"]
                if image_names is not None:
                    header.append("画像名")
                writer.writerow(header)

                for i in range(len(image_paths)):
                    img_path = image_paths[i]
                    row = [str(i), img_path]
                    if image_names is not None:
                        img_base = os.path.splitext(os.path.basename(img_path))[0]
                        img_name = image_names.get(img_base, "")
                        row.append(img_name)
                    writer.writerow(row)
        except Exception as e:
            raise type(e)(f"{_path} の保存に失敗") from e

    # 各カードの画像名のcsv
    if image_paths is not None:
        id_to_base: dict[int, str] = {i: os.path.splitext(os.path.basename(x))[0] for i, x in enumerate(image_paths)}
        if image_names is None:
            id_to_name = id_to_base
        else:
            id_to_name: dict[int, str] = {i: image_names.get(name, "") for i, name in id_to_base.items()}

        _path = os.path.join(output_dir, "card_names.csv")
        rows = [[id_to_name[id] for id in row] for row in pairs]
        try:
            with open(_path, "w", encoding="utf_8_sig") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(rows)
        except Exception as e:
            raise type(e)(f"{_path} の保存に失敗") from e

    return


def main() -> None:
    # ============
    # パラメータ設定
    # ============
    # ファイル名
    image_dir = "samples"  # 入力画像ディレクトリ
    output_dir = "output"  # 出力画像ディレクトリ
    pdf_name = "card.pdf"  # 出力するPDF名
    params_name = "parameters.json"  # 実行時のパラメータ値を出力するjson名
    # デッキの設定
    deck_type: Literal["normal", "twin-symbols", "triple-cards"] = "normal"
    # カードの設定
    n_symbols_per_card: int = 5  # カード1枚当たりのシンボル数
    n_cards: int | None = None  # 全カード数 ※ deck_type == "triple-cards" の場合のみ参照される
    card_shape: CARD_SHAPE = CARD_SHAPE.CIRCLE
    card_img_size = 1500  # カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
    card_margin = 20  # カード1枚の余白サイズ [pix]
    layout_method: Literal["random", "voronoi"] = (
        "voronoi"  # random: ランダム配置, voronoi: 重心ボロノイ分割に基づき配置
    )
    radius_p: float = (
        0.5  # "voronoi"における各母点の半径を決めるパラメータ (0.0なら半径なし, つまり通常のボロノイ分割として処理)
    )
    n_voronoi_iters = 10  # "voronoi"における反復回数
    min_image_size_rate: float = 0.1  # "voronoi"における最小画像サイズ (カードサイズ長辺に対する画像サイズ長辺の比)
    max_image_size_rate: float = 0.5  # "voronoi"における最大画像サイズ (カードサイズ長辺に対する画像サイズ長辺の比)
    # PDFの設定
    dpi = 300  # 解像度
    card_size_mm = 95  # カードの長辺サイズ[mm]
    page_size_mm = (210, 297)  # PDFの(幅, 高さ)[mm]
    # 画像リストの設定
    image_list_path: None | str = r"samples\画像リスト.xlsx"  # xlsx | csv のパス
    image_table_size: tuple[int, int] = (8, 6)  # 画像リストの表サイズ (行数, 列数)
    thumb_margin: float = 0.055  # サムネイル周囲の余白調整 (0.0-0.5, 主にカード端に画像が入らない場合の微調整用)
    text_h_rate: float = 0.3  # サムネイル高さに対する上限文字高さの割合

    # その他
    shuffle: bool = False  # True: 画像読み込みをシャッフルする
    seed: int | None = 0  # 乱数種
    gen_card_images: bool = True  # (主にデバッグ用) もし output_dir にある生成済みの画像群を使うならFalse

    # ======================
    # 出力フォルダ作成
    # ======================
    os.makedirs(output_dir, exist_ok=True)

    # ======================
    # パラメータをjsonで出力
    # ======================
    params = {
        "deck_type": deck_type,
        "n_symbols_per_card": n_symbols_per_card,
        "card_shape": card_shape.name,
        "card_img_size": card_img_size,
        "card_margin": card_margin,
        "layout_method": layout_method,
        "radius_p": radius_p,
        "n_voronoi_iters": n_voronoi_iters,
        "min_image_size_rate": min_image_size_rate,
        "max_image_size_rate": max_image_size_rate,
        "dpi": dpi,
        "card_size_mm": card_size_mm,
        "page_size_mm": page_size_mm,
        "image_table_size": image_table_size,
        "thumb_margin": thumb_margin,
        "text_h_rate": text_h_rate,
        "shuffle": shuffle,
        "seed": seed,
    }
    if deck_type == "triple-cards":
        params["n_cards"] = n_cards
    with open(output_dir + os.sep + params_name, mode="w", encoding="utf_8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # ========
    # 前処理
    # ========
    # 乱数初期化
    random.seed(seed)
    np.random.seed(seed)

    # ========
    # メイン
    # ========
    # 各カード毎の組み合わせを生成
    match deck_type:
        case "normal":
            pairs, n_symbols = make_dobble_deck(n_symbols_per_card, n_shared_symbols=1, n_shared_cards=2)
        case "twin-symbols":
            pairs, n_symbols = make_dobble_deck(n_symbols_per_card, n_shared_symbols=2, n_shared_cards=2)
        case "triple-cards":
            pairs, n_symbols = make_dobble_deck(
                n_symbols_per_card, n_shared_symbols=1, n_shared_cards=3, n_cards=n_cards
            )
        case _:
            raise ValueError("deck_type: '{deck_type}' が不適切")

    save_card_list_to_csv(output_dir, pairs)  # 組み合わせのcsvを出力

    # image_dirからn_symbols数の画像を取得
    images, image_paths = load_images(image_dir, n_symbols, shuffle=shuffle)

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
                card_shape,
                draw_frame=True,
                method=layout_method,
                radius_p=radius_p,
                n_voronoi_iters=n_voronoi_iters,
                min_image_size_rate=min_image_size_rate,
                max_image_size_rate=max_image_size_rate,
                show=False,
            )
            imwrite_japanese(path, card_img)
        else:
            card_img = imread_japanese(path)
            assert card_img is not None  # 必ず読み込める前提

        card_images.append(card_img)

    # 画像リストファイルの指定があれば画像リストカード画像を作成
    image_names: None | dict[str, str] = None
    if image_list_path is not None:
        image_names = load_image_list(image_list_path)
        sorted_images, sorted_names = sort_images_by_image_list(
            images, image_paths, image_names
        )  # image_namesの順序でimage_pathsをソート
        thumbs_cards = make_image_of_thumbnails_with_names(
            card_shape,
            card_img_size,
            card_margin,
            image_table_size,
            sorted_images,
            sorted_names,
            thumb_margin=thumb_margin,
            text_h_rate=text_h_rate,
            draw_frame=True,
        )  # 画像をサムネイル化したカード画像を作成
        for i, card in enumerate(thumbs_cards):
            path = output_dir + os.sep + f"thumbnail_{i}.png"
            imwrite_japanese(path, card)
            card_images.append(card)

    # カード、シンボル画像に関する情報をcsv出力
    save_card_list_to_csv(output_dir, pairs, image_paths=image_paths, image_names=image_names)

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
