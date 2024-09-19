"""t = 2, lambda = 2 のブロックデザインを解く

t = 2, λ = 2において、例えば k = 3の場合、b = v = 4, r = k = 3である。
これをデッキの表現として以下のようにする。

1 1 1 0
1 1 0 1
1 0 1 1
0 1 1 1

行列: デッキ構成
行数: カード数 b
列数: シンボル数 v
行番号: カードID
列番号: シンボルID
値: 0 or 1, 1の場合、カードIDがシンボルIDを有するということ
行方向の合計値: カード当たりのシンボル数 k
列方向の合計値: シンボルの出現回数 r

上記行列を D (Deck) として、任意の2つのカード(行)で共通するシンボル(列)の総数が2である、ということは
    D * D^T の対角成分が k (行方向の合計値) , 非対角成分が 2
である、ことが条件となる

全シンボル数をv, 1カードが有するシンボル数をkとするため、カードの種類数は Combination(v, k) となる
さらにその中から任意にb枚のカードを選択してデッキとするため、構成可能なデッキ数は Combination(Combination(v, k), b) となる
これらから、前記の条件を満たすデッキを選択すればよい

t=2つのシンボルがデッキ内でλ=2回同時に出現するデッキを構築する。
t=2であるためBIBDとして成立する。なお、t=2, λ=2は特にbiplaneと呼ばれる。
BIBDであれば対称性(b=v)の制約を付与できるため、カード総数b = シンボル数vとする
ブロックデザインの定義より
    v * r = b * k  (1)
    λ * C(v, t) = b * C(k, t)  (2)
BIBD (t=2) であるため、(2)を展開して
    λ * (v-1) = r * (k - 1)  (3)
対称性の制約を付与して
    b = v  (4)
(1)に(4)を代入して
    r = k  (5)
ここではλ = 2について考えるため、(3)に代入して
    v = k * (k -1) / 2 + 1  (6)

"""

import math
import sys
from itertools import combinations
from typing import Final, Literal, Sequence

import numpy as np
from tqdm import tqdm

# cx_Freezeでbase=Win32GUIでビルドされた場合、標準出力が空になりtqdmが使えないため判定
disable_tqdm: Final[bool] = (not sys.stdout) or (not sys.stderr)


def _check_symmetric_bibd(deck_m: np.ndarray, k: int, lambda_: Literal[1, 2]) -> None:
    r = k

    assert all(deck_m.sum(axis=0) == r)  # デッキ全体での各シンボルの出現回数がrであることを確認
    assert all(deck_m.sum(axis=1) == k)  # 各カードが有するシンボル数は必ずkになる
    # 対角がk, 非対角がλになっているか確認
    product = deck_m.dot(deck_m.T)
    assert np.all(np.diag(product) == k) and np.all(product[~np.eye(product.shape[0], dtype=bool)] == lambda_)


def _deck_to_deck_matrix(deck: Sequence[Sequence[int]], *, n_cards: int = 0, n_symbols: int = 0) -> np.ndarray:
    if n_cards < 1 or n_symbols < 1:
        n_cards = n_symbols = len(set([s for card in deck for s in card]))

    deck_m = np.zeros((n_cards, n_symbols), dtype=int)
    for card_i, card in enumerate(deck):
        for symbol_i in card:
            deck_m[card_i, symbol_i] = 1

    return deck_m


def _deck_matrix_to_deck(deck_m: np.ndarray) -> list[tuple[int, ...]]:
    deck = []
    for i in range(deck_m.shape[0]):
        card = tuple(np.where(deck_m[i] == 1)[0])
        deck.append(card)

    return deck


def SymmetricBIBD_L2_K2() -> tuple[np.ndarray, list[tuple[int, int]]]:
    # 算出済みの行列を定義しておく
    lambda_ = 2
    k = 2
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [(0, 1), (0, 1)]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K3() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    # 算出済みの行列を定義しておく
    # https://math.stackexchange.com/questions/3990502/algorithm-to-create-spot-it-dobble-cards-but-with-two-common-images-from-an
    lambda_ = 2
    k = 3
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K4() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    # 算出済みの行列を定義しておく
    # https://math.stackexchange.com/questions/3990502/algorithm-to-create-spot-it-dobble-cards-but-with-two-common-images-from-an
    lambda_ = 2
    k = 4
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [(0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (0, 3, 5, 6), (1, 2, 5, 6), (1, 3, 4, 6), (2, 3, 4, 5)]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K5() -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    # 算出済みの行列を定義しておく
    # https://math.stackexchange.com/questions/3990502/algorithm-to-create-spot-it-dobble-cards-but-with-two-common-images-from-an
    lambda_ = 2
    k = 5
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [
        (0, 1, 2, 4, 7),
        (0, 1, 3, 6, 10),
        (0, 2, 5, 9, 10),
        (0, 3, 7, 8, 9),
        (0, 4, 5, 6, 8),
        (1, 2, 3, 5, 8),
        (1, 4, 8, 9, 10),
        (1, 5, 6, 7, 9),
        (2, 3, 4, 6, 9),
        (2, 6, 7, 8, 10),
        (3, 4, 5, 7, 10),
    ]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K6() -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]]]:
    # 算出済みの行列を定義しておく
    # SageMathのbalanced_incomplete_block_designクラスを使用
    # - https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/designs/bibd.html
    # - https://sagecell.sagemath.org/
    lambda_ = 2
    k = 6
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [
        (0, 1, 2, 10, 11, 15),
        (0, 1, 4, 5, 6, 14),
        (0, 2, 3, 8, 9, 14),
        (0, 3, 6, 7, 10, 13),
        (0, 4, 9, 11, 12, 13),
        (0, 5, 7, 8, 12, 15),
        (1, 2, 7, 12, 13, 14),
        (1, 3, 4, 7, 8, 11),
        (1, 3, 5, 9, 13, 15),
        (1, 6, 8, 9, 10, 12),
        (2, 3, 4, 5, 10, 12),
        (2, 4, 6, 8, 13, 15),
        (2, 5, 6, 7, 9, 11),
        (3, 6, 11, 12, 14, 15),
        (4, 7, 9, 10, 14, 15),
        (5, 8, 10, 11, 13, 14),
    ]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K9() -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int, int, int]]]:
    # 算出済みの行列を定義しておく
    # SageMathのbalanced_incomplete_block_designクラスを使用
    lambda_ = 2
    k = 9
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [
        (0, 1, 3, 7, 17, 24, 25, 29, 35),
        (0, 1, 5, 11, 13, 14, 16, 20, 30),
        (0, 2, 3, 5, 9, 19, 26, 27, 31),
        (0, 2, 6, 16, 23, 24, 28, 34, 36),
        (0, 4, 10, 12, 13, 15, 19, 29, 36),
        (0, 4, 14, 21, 22, 26, 32, 34, 35),
        (0, 6, 8, 9, 11, 15, 25, 32, 33),
        (0, 7, 8, 12, 18, 20, 21, 23, 27),
        (0, 10, 17, 18, 22, 28, 30, 31, 33),
        (1, 2, 4, 8, 18, 25, 26, 30, 36),
        (1, 2, 6, 12, 14, 15, 17, 21, 31),
        (1, 3, 4, 6, 10, 20, 27, 28, 32),
        (1, 5, 15, 22, 23, 27, 33, 35, 36),
        (1, 7, 9, 10, 12, 16, 26, 33, 34),
        (1, 8, 9, 13, 19, 21, 22, 24, 28),
        (1, 11, 18, 19, 23, 29, 31, 32, 34),
        (2, 3, 7, 13, 15, 16, 18, 22, 32),
        (2, 4, 5, 7, 11, 21, 28, 29, 33),
        (2, 8, 10, 11, 13, 17, 27, 34, 35),
        (2, 9, 10, 14, 20, 22, 23, 25, 29),
        (2, 12, 19, 20, 24, 30, 32, 33, 35),
        (3, 4, 8, 14, 16, 17, 19, 23, 33),
        (3, 5, 6, 8, 12, 22, 29, 30, 34),
        (3, 9, 11, 12, 14, 18, 28, 35, 36),
        (3, 10, 11, 15, 21, 23, 24, 26, 30),
        (3, 13, 20, 21, 25, 31, 33, 34, 36),
        (4, 5, 9, 15, 17, 18, 20, 24, 34),
        (4, 6, 7, 9, 13, 23, 30, 31, 35),
        (4, 11, 12, 16, 22, 24, 25, 27, 31),
        (5, 6, 10, 16, 18, 19, 21, 25, 35),
        (5, 7, 8, 10, 14, 24, 31, 32, 36),
        (5, 12, 13, 17, 23, 25, 26, 28, 32),
        (6, 7, 11, 17, 19, 20, 22, 26, 36),
        (6, 13, 14, 18, 24, 26, 27, 29, 33),
        (7, 14, 15, 19, 25, 27, 28, 30, 34),
        (8, 15, 16, 20, 26, 28, 29, 31, 35),
        (9, 16, 17, 21, 27, 29, 30, 32, 36),
    ]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K11() -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int, int, int, int, int]]]:
    # 算出済みの行列を定義しておく
    # SageMathのbalanced_incomplete_block_designクラスを使用
    lambda_ = 2
    k = 11
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [
        (0, 1, 2, 5, 10, 13, 36, 44, 46, 47, 52),
        (0, 1, 3, 4, 11, 15, 17, 19, 28, 49, 53),
        (0, 2, 7, 12, 20, 25, 27, 35, 37, 38, 53),
        (0, 3, 10, 12, 23, 26, 39, 41, 48, 50, 51),
        (0, 4, 9, 13, 16, 30, 32, 33, 35, 51, 54),
        (0, 5, 14, 17, 24, 25, 31, 32, 40, 42, 50),
        (0, 6, 7, 8, 21, 23, 33, 34, 42, 47, 49),
        (0, 6, 18, 20, 24, 26, 28, 29, 52, 54, 55),
        (0, 8, 11, 16, 18, 22, 31, 38, 39, 43, 46),
        (0, 9, 19, 21, 29, 36, 37, 40, 41, 43, 45),
        (0, 14, 15, 22, 27, 30, 34, 44, 45, 48, 55),
        (1, 2, 6, 11, 32, 38, 41, 42, 45, 48, 54),
        (1, 3, 7, 8, 10, 14, 16, 24, 29, 35, 45),
        (1, 4, 14, 18, 20, 33, 37, 43, 47, 48, 50),
        (1, 5, 6, 12, 18, 19, 22, 34, 35, 40, 51),
        (1, 7, 9, 12, 15, 23, 31, 32, 43, 52, 55),
        (1, 8, 17, 20, 23, 27, 30, 36, 39, 40, 54),
        (1, 9, 21, 24, 26, 30, 34, 38, 46, 50, 53),
        (1, 13, 22, 25, 26, 27, 29, 31, 33, 41, 49),
        (1, 16, 21, 25, 28, 37, 39, 42, 44, 51, 55),
        (2, 3, 6, 8, 9, 13, 17, 22, 37, 50, 55),
        (2, 3, 16, 26, 27, 28, 32, 34, 40, 43, 47),
        (2, 4, 5, 15, 16, 20, 21, 22, 23, 24, 41),
        (2, 4, 7, 19, 29, 31, 34, 39, 44, 50, 54),
        (2, 8, 19, 24, 25, 30, 43, 48, 49, 51, 52),
        (2, 9, 14, 15, 18, 26, 35, 36, 39, 42, 49),
        (2, 10, 12, 17, 18, 21, 28, 30, 31, 33, 45),
        (2, 11, 14, 23, 29, 33, 40, 46, 51, 53, 55),
        (3, 4, 6, 14, 21, 27, 31, 36, 38, 51, 52),
        (3, 5, 9, 11, 20, 25, 33, 34, 39, 45, 52),
        (3, 5, 18, 23, 29, 30, 32, 37, 38, 44, 49),
        (3, 7, 13, 15, 18, 21, 25, 40, 46, 48, 54),
        (3, 12, 22, 24, 33, 36, 42, 43, 44, 53, 54),
        (3, 19, 20, 30, 31, 35, 41, 42, 46, 47, 55),
        (4, 5, 8, 9, 12, 27, 28, 29, 42, 46, 48),
        (4, 6, 17, 23, 25, 26, 35, 43, 44, 45, 46),
        (4, 7, 10, 11, 22, 26, 30, 37, 40, 42, 52),
        (4, 8, 10, 18, 25, 32, 34, 36, 41, 53, 55),
        (4, 12, 13, 24, 38, 39, 40, 45, 47, 49, 55),
        (5, 6, 7, 13, 14, 28, 30, 39, 41, 43, 53),
        (5, 7, 16, 17, 19, 26, 33, 36, 38, 48, 55),
        (5, 8, 15, 26, 31, 37, 45, 47, 51, 53, 54),
        (5, 10, 11, 21, 27, 35, 43, 49, 50, 54, 55),
        (6, 9, 10, 16, 20, 31, 40, 44, 48, 49, 53),
        (6, 10, 15, 19, 24, 27, 32, 33, 37, 39, 46),
        (6, 11, 12, 15, 16, 25, 29, 30, 36, 47, 50),
        (7, 9, 11, 17, 18, 24, 27, 41, 44, 47, 51),
        (7, 20, 22, 28, 32, 36, 45, 46, 49, 50, 51),
        (8, 11, 12, 13, 14, 19, 20, 21, 26, 32, 44),
        (8, 15, 28, 33, 35, 38, 40, 41, 44, 50, 52),
        (9, 10, 14, 19, 22, 23, 25, 28, 38, 47, 54),
        (10, 13, 15, 17, 20, 29, 34, 38, 42, 43, 51),
        (11, 13, 23, 24, 28, 31, 34, 35, 36, 37, 48),
        (12, 14, 16, 17, 34, 37, 41, 46, 49, 52, 54),
        (13, 16, 18, 19, 23, 27, 42, 45, 50, 52, 53),
        (17, 21, 22, 29, 32, 35, 39, 47, 48, 52, 53),
    ]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def SymmetricBIBD_L2_K13() -> (
    tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]]]
):
    # 算出済みの行列を定義しておく
    # SageMathのbalanced_incomplete_block_designクラスを使用
    lambda_ = 2
    k = 13
    # t = 2
    # v = k * (k - 1) // lambda_ + 1
    # b = v
    # r = k

    deck = [
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (0, 1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
        (0, 2, 13, 25, 28, 35, 36, 43, 49, 51, 63, 66, 72),
        (0, 3, 14, 27, 35, 46, 48, 50, 56, 59, 65, 71, 77),
        (0, 4, 15, 31, 32, 37, 42, 48, 53, 54, 66, 70, 73),
        (0, 5, 16, 33, 44, 45, 49, 55, 59, 62, 64, 70, 76),
        (0, 6, 17, 24, 28, 32, 39, 47, 52, 58, 64, 69, 77),
        (0, 7, 18, 33, 37, 41, 43, 52, 57, 60, 67, 71, 74),
        (0, 8, 19, 25, 30, 34, 38, 42, 50, 55, 58, 67, 68),
        (0, 9, 20, 27, 29, 38, 47, 54, 57, 61, 62, 72, 78),
        (0, 10, 21, 34, 40, 45, 46, 61, 63, 69, 73, 74, 75),
        (0, 11, 22, 24, 26, 30, 36, 41, 44, 53, 65, 75, 78),
        (0, 12, 23, 26, 29, 31, 39, 40, 51, 56, 60, 68, 76),
        (1, 2, 13, 29, 37, 48, 50, 55, 60, 64, 69, 75, 78),
        (1, 3, 14, 24, 33, 37, 44, 51, 54, 58, 61, 63, 68),
        (1, 4, 15, 25, 33, 34, 39, 45, 52, 56, 65, 72, 78),
        (1, 5, 16, 24, 26, 32, 40, 46, 50, 57, 66, 67, 72),
        (1, 6, 17, 34, 35, 36, 41, 48, 57, 62, 68, 73, 76),
        (1, 7, 18, 27, 29, 36, 40, 42, 45, 49, 53, 58, 77),
        (1, 8, 19, 28, 43, 46, 47, 53, 60, 61, 65, 70, 76),
        (1, 9, 20, 28, 31, 35, 39, 42, 44, 59, 67, 74, 75),
        (1, 10, 21, 25, 26, 30, 31, 43, 54, 62, 64, 71, 77),
        (1, 11, 22, 32, 38, 47, 49, 51, 55, 56, 71, 73, 74),
        (1, 12, 23, 27, 30, 38, 41, 52, 59, 63, 66, 69, 70),
        (2, 3, 15, 17, 24, 25, 38, 40, 53, 59, 60, 62, 74),
        (2, 4, 16, 20, 40, 41, 42, 47, 63, 64, 65, 68, 71),
        (2, 5, 19, 22, 26, 27, 28, 45, 48, 52, 54, 68, 74),
        (2, 6, 14, 20, 24, 29, 30, 43, 45, 56, 67, 70, 73),
        (2, 7, 19, 23, 39, 44, 46, 58, 62, 66, 71, 73, 78),
        (2, 8, 18, 21, 32, 41, 54, 56, 58, 59, 72, 75, 76),
        (2, 9, 15, 22, 30, 33, 42, 46, 51, 57, 69, 76, 77),
        (2, 10, 17, 23, 27, 31, 32, 33, 36, 55, 61, 65, 67),
        (2, 11, 16, 18, 26, 34, 35, 37, 38, 39, 61, 70, 77),
        (2, 12, 14, 21, 31, 34, 44, 47, 49, 50, 52, 53, 57),
        (3, 4, 13, 18, 25, 26, 27, 44, 47, 67, 69, 73, 76),
        (3, 5, 20, 22, 31, 34, 36, 58, 60, 69, 70, 71, 72),
        (3, 6, 18, 19, 31, 38, 45, 51, 57, 64, 65, 66, 75),
        (3, 7, 17, 21, 26, 28, 42, 55, 56, 57, 63, 70, 78),
        (3, 8, 16, 21, 29, 30, 33, 36, 39, 47, 48, 66, 74),
        (3, 9, 16, 23, 43, 52, 53, 55, 68, 72, 73, 75, 77),
        (3, 10, 19, 20, 30, 32, 35, 37, 40, 49, 52, 76, 78),
        (3, 11, 15, 23, 28, 29, 34, 41, 46, 49, 54, 64, 67),
        (3, 12, 13, 22, 32, 39, 41, 42, 43, 45, 50, 61, 62),
        (4, 5, 13, 23, 30, 35, 53, 56, 57, 58, 61, 64, 74),
        (4, 6, 21, 22, 49, 59, 60, 61, 66, 67, 68, 77, 78),
        (4, 7, 14, 16, 27, 28, 30, 32, 34, 51, 60, 62, 75),
        (4, 8, 18, 22, 24, 29, 31, 35, 46, 52, 55, 62, 63),
        (4, 9, 21, 23, 24, 28, 36, 37, 38, 45, 50, 71, 76),
        (4, 10, 17, 19, 29, 41, 44, 50, 51, 70, 72, 74, 77),
        (4, 11, 14, 19, 36, 39, 40, 43, 54, 55, 57, 59, 69),
        (4, 12, 17, 20, 26, 33, 38, 43, 46, 48, 49, 58, 75),
        (5, 6, 20, 21, 25, 27, 37, 39, 41, 46, 51, 53, 55),
        (5, 7, 15, 21, 29, 32, 35, 38, 43, 44, 65, 68, 69),
        (5, 8, 13, 14, 28, 31, 33, 38, 40, 41, 73, 77, 78),
        (5, 9, 17, 18, 30, 39, 49, 50, 54, 60, 63, 65, 73),
        (5, 10, 18, 23, 24, 34, 42, 43, 47, 48, 51, 59, 78),
        (5, 11, 14, 17, 25, 29, 42, 52, 61, 66, 71, 75, 76),
        (5, 12, 15, 19, 36, 37, 47, 56, 62, 63, 67, 75, 77),
        (6, 7, 22, 23, 25, 33, 35, 40, 47, 50, 54, 70, 75),
        (6, 8, 14, 23, 26, 37, 42, 49, 62, 65, 69, 72, 74),
        (6, 9, 13, 19, 26, 29, 32, 33, 34, 53, 59, 63, 71),
        (6, 10, 13, 16, 36, 38, 42, 44, 46, 52, 54, 56, 60),
        (6, 11, 15, 16, 27, 31, 43, 50, 58, 63, 74, 76, 78),
        (6, 12, 15, 18, 28, 30, 40, 44, 48, 55, 61, 71, 72),
        (7, 8, 15, 20, 26, 36, 50, 51, 52, 59, 61, 64, 73),
        (7, 9, 16, 19, 24, 25, 31, 41, 48, 49, 56, 61, 69),
        (7, 10, 14, 22, 38, 39, 48, 53, 63, 64, 67, 72, 76),
        (7, 11, 13, 17, 30, 31, 37, 45, 46, 47, 59, 68, 72),
        (7, 12, 13, 20, 24, 34, 54, 55, 65, 66, 74, 76, 77),
        (8, 9, 17, 22, 27, 34, 37, 40, 43, 44, 56, 64, 66),
        (8, 10, 13, 15, 24, 27, 39, 49, 57, 68, 70, 71, 75),
        (8, 11, 20, 23, 25, 32, 44, 45, 48, 57, 60, 63, 77),
        (8, 12, 16, 17, 35, 45, 51, 53, 54, 67, 69, 71, 78),
        (9, 10, 14, 15, 26, 35, 41, 45, 47, 55, 58, 60, 66),
        (9, 11, 13, 21, 40, 48, 51, 52, 58, 62, 65, 67, 70),
        (9, 12, 14, 18, 25, 32, 36, 46, 64, 68, 70, 74, 78),
        (10, 11, 18, 20, 28, 33, 50, 53, 56, 62, 66, 68, 69),
        (10, 12, 16, 22, 25, 28, 29, 37, 57, 58, 59, 65, 73),
        (11, 12, 19, 21, 24, 27, 33, 35, 42, 60, 64, 72, 73),
    ]
    deck_m = _deck_to_deck_matrix(deck)

    # 検算
    _check_symmetric_bibd(deck_m, k, lambda_)

    return deck_m, deck


def _calc_symmetric_bibd_brute_force(lambda_: Literal[1, 2], k: int) -> tuple[np.ndarray, list[tuple[int, ...]]] | None:
    """対称BIBDを総当たりで解く

    NOTE: 非常に時間がかかるため未使用

    Args:
        lambda_ (Literal[1, 2]): λ
        k (int): k
    """
    # t: Final[int] = 2
    v: int = k * (k - 1) // lambda_ + 1  # 必ず割り切れる
    r: Final = k
    b: Final = v

    card_patterns_n: int = math.comb(v, k)  # 全カードパターン数
    deck_patterns_n: int = math.comb(card_patterns_n, b)  # 全デッキパターン数
    card_patterns_yield = combinations(range(v), k)
    deck_patterns_yield = combinations(card_patterns_yield, b)
    result_deck_m = None
    for deck in tqdm(deck_patterns_yield, desc="デッキ構築", total=deck_patterns_n, disable=disable_tqdm):
        deck_m = _deck_to_deck_matrix(list(deck), n_cards=v, n_symbols=v)

        if not all(deck_m.sum(axis=0) == r):
            # デッキ全体での各シンボルの出現回数がrであることを確認
            continue
        assert all(deck_m.sum(axis=1) == k)  # 各カードが有するシンボル数は必ずkになる
        # 対角がk, 非対角がλになっているか確認
        product = deck_m.dot(deck_m.T)
        if np.all(np.diag(product) == k) and np.all(product[~np.eye(product.shape[0], dtype=bool)] == lambda_):
            result_deck_m = deck_m
            break

    if result_deck_m is None:
        return None

    result_deck = _deck_matrix_to_deck(result_deck_m)

    return result_deck_m, result_deck


def _calc_symmetric_bibd_lambda2(k: int) -> tuple[np.ndarray, Sequence[Sequence[int]]]:
    match k:
        case 2:
            return SymmetricBIBD_L2_K2()
        case 3:
            return SymmetricBIBD_L2_K3()
        case 4:
            return SymmetricBIBD_L2_K4()
        case 5:
            return SymmetricBIBD_L2_K5()
        case 6:
            return SymmetricBIBD_L2_K6()
        case 9:
            return SymmetricBIBD_L2_K9()
        case 11:
            return SymmetricBIBD_L2_K11()
        case 13:
            return SymmetricBIBD_L2_K13()
        case _:
            raise ValueError(
                f"カード1枚当たりのシンボル数 ({k}) は「2, 3, 4, 5, 6, 9, 11, 13」のいずれかでなければならない"
            )


def get_valid_params() -> list[int]:
    """設定可能な n_symbols_per_card の一覧を取得"""
    return [2, 3, 4, 5, 6, 9, 11, 13]


def make_dobble_deck(n_symbols_per_card: int) -> tuple[list[list[int]], int]:
    """各カードに記載するシンボルの一覧を生成

    2枚のカードに2つの共通するシンボルが出現する拡張ドブル版

    Args:
        n_symbols_per_card (int): カード1枚あたりに記載するシンボル数

    Returns:
        list[list[int]]: 各カードに記載するシンボル番号
        int: 全シンボル数
    """
    _, deck = _calc_symmetric_bibd_lambda2(n_symbols_per_card)
    deck = [list(card) for card in deck]

    n_cards = len(deck)
    n_symbols = len(set(symbol for card in deck for symbol in card))
    assert n_cards == n_symbols  # カード枚数とシンボル数は同じ

    return deck, n_symbols


if __name__ == "__main__":
    for k in get_valid_params():
        deck, n_symbols = make_dobble_deck(k)
        print(f"k={k}, v={n_symbols}, deck={deck}")
