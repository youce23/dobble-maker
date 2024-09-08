import math
from itertools import combinations
from typing import Final, Literal

import numpy as np
from tqdm import tqdm

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

t: Final[int] = 2
lambda_: Literal[1, 2] = 2
k: int = 4
v: int = k * (k - 1) // lambda_ + 1  # 必ず割り切れる
r: Final = k
b: Final = v

card_patterns_n: int = math.comb(v, k)  # 全カードパターン数
deck_patterns_n: int = math.comb(card_patterns_n, b)  # 全デッキパターン数
card_patterns_yield = combinations(range(v), k)
deck_patterns_yield = combinations(card_patterns_yield, b)
result_deck = None
for deck in tqdm(deck_patterns_yield, desc="デッキ構築", total=deck_patterns_n):
    deck_m = np.zeros((v, v), dtype=int)
    for card_i, card in enumerate(deck):
        for symbol_i in card:
            deck_m[card_i, symbol_i] = 1
    if not all(deck_m.sum(axis=0) == r):
        # デッキ全体での各シンボルの出現回数がrであることを確認
        continue
    assert all(deck_m.sum(axis=1) == k)  # 各カードが有するシンボル数は必ずkになる
    # 対角がk, 非対角がλになっているか確認
    product = deck_m.dot(deck_m.T)
    if np.all(np.diag(product) == k) and np.all(product[~np.eye(product.shape[0], dtype=bool)] == lambda_):
        result_deck = deck_m
        break

print(result_deck)
