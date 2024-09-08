import math
from itertools import combinations
from typing import Final, Literal, Sequence

import numpy as np
from tqdm import tqdm

from dobble_maker import make_dobble_deck

# t = 2, λ = 2において、例えば k = 3の場合、b = v = 4, r = k = 3である。
# これをデッキの表現として以下のようにする。
#
# 1 1 1 0
# 1 1 0 1
# 1 0 1 1
# 0 1 1 1
#
# 行列: デッキ構成
# 行数: カード数 b
# 列数: シンボル数 v
# 行番号: カードID
# 列番号: シンボルID
# 値: 0 or 1, 1の場合、カードIDがシンボルIDを有するということ
# 行方向の合計値: カード当たりのシンボル数 k
# 列方向の合計値: シンボルの出現回数 r
#
# 上記行列を D (Deck) として、任意の2つのカード(行)で共通するシンボル(列)の総数が2である、ということは
#   D * D^T の対角成分が k (行方向の合計値) , 非対角成分が 2
# である、ことが条件となる
#
# 全シンボル数をv, 1カードが有するシンボル数をkとするため、カードの種類数は Combination(v, k) となる
# さらにその中から任意にb枚のカードを選択してデッキとするため、構成可能なデッキ数は Combination(Combination(v, k), b) となる
# これらから、前記の条件を満たすデッキを選択すればよい

# t=2つのシンボルがデッキ内でλ=2回同時に出現するデッキを構築する
# t=2であるためBIBDとして成立する。
# BIBDであれば対称性(b=v)の制約を付与できるため、カード総数b = シンボル数vとする
# ブロックデザインの定義より
#   v * r = b * k  (1)
#   λ * C(v, t) = b * C(k, t)  (2)
# BIBD (t=2) であるため、(2)を展開して
#   λ * (v-1) = r * (k - 1)  (3)
# 対称性の制約を付与して
#   b = v  (4)
# (1)に(4)を代入して
#   r = k  (5)
# ここではλ = 2について考えるため、(3)に代入して
#   v = k * (k -1) / 2 + 1  (6)


def _check_symmetric_bibd(deck_m: np.ndarray, k: int, lambda_: Literal[1, 2]):
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


def calc_symmetric_bibd_brute_force(lambda_: Literal[1, 2], k: int) -> tuple[np.ndarray, list[tuple[int, ...]]] | None:
    # t: Final[int] = 2
    v: int = k * (k - 1) // lambda_ + 1  # 必ず割り切れる
    r: Final = k
    b: Final = v

    card_patterns_n: int = math.comb(v, k)  # 全カードパターン数
    deck_patterns_n: int = math.comb(card_patterns_n, b)  # 全デッキパターン数
    card_patterns_yield = combinations(range(v), k)
    deck_patterns_yield = combinations(card_patterns_yield, b)
    result_deck_m = None
    for deck in tqdm(deck_patterns_yield, desc="デッキ構築", total=deck_patterns_n):
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


def calc_symmetric_bibd(lambda_: Literal[1, 2], k: int) -> tuple[np.ndarray, Sequence[Sequence[int]]] | None:
    if lambda_ == 1:
        deck, _ = make_dobble_deck(k)
        deck_m = _deck_to_deck_matrix(deck)
        _check_symmetric_bibd(deck_m, k, lambda_)
        return deck_m, deck
    elif lambda_ == 2 and k == 3:
        return SymmetricBIBD_L2_K3()
    elif lambda_ == 2 and k == 4:
        return SymmetricBIBD_L2_K4()
    elif lambda_ == 2 and k == 5:
        return SymmetricBIBD_L2_K5()
    else:
        return calc_symmetric_bibd_brute_force(2, k)


if __name__ == "__main__":
    lambda_: Literal[1, 2] = 2
    k: int = 5

    ret = calc_symmetric_bibd(lambda_, k)
    if ret is not None:
        deck_m, deck = ret
        print(deck_m)
        print(deck)
