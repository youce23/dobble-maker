from itertools import combinations

import galois
import numpy as np
from tqdm import tqdm


def generate_constacyclic_code(
    n: int, k: int, q: int, generator_poly_coeffs: list[int], alpha: int
) -> galois.FieldArray:
    """数値計算ソフトの MAGMA で定義される ConstaCyclicCode 関数を再現

    https://magma.maths.usyd.edu.au/magma/handbook/text/1913

    - 位数qの有限体における生成多項式の係数を先頭行に並べる
    - 次の行は直前の行を1個右にシフトし、先頭要素 (つまり生成多項式における x^0 の係数) にalphaを掛ける

    Args:
        n: 生成行列の列数
        k: 生成行列の行数
        q: 有限体の位数
        generator_poly_coeffs:
            生成多項式を a_0 + a_1 * x + a_2 * x^2 + ... + a_(n-1) * x^(n-1) とする時、
            係数を [a_0, a_1, ..., a_(n-1)] として並べたリスト
        alpha: 先頭に掛ける定数

    Returns: 生成行列 (GF(q), shape: (k, n))
    """
    # 有限体の定義
    gf = galois.GF(q)

    # 生成行列 G を作成
    g_coeffs = [gf(c) for c in generator_poly_coeffs]  # 生成多項式の係数 (有限体として定義)
    G = gf(np.zeros((k, n), dtype=int))
    G[0, : len(g_coeffs)] = generator_poly_coeffs  # 先頭行を生成多項式の係数で埋める
    for i in range(1, k):  # 次の行をi個分シフトして生成
        G[i] = np.roll(G[0], i)
        G[i, 0] *= alpha  # 先頭要素にalphaを掛ける

    return G


def generate_dobble_deck_from_parity_check_matrix(H: galois.FieldArray) -> tuple[list[list[int]], int, int]:
    """パリティ検査行列に従ってドブルデッキ構築

    以下の制約がある
    - デッキ内の各カードに記載されるシンボル数は一定ではない
    - 各シンボルの出現回数は一定ではない

    Args:
        H: パリティ検査行列

    Returns:
        list[list[int]: デッキ
        int: 全シンボル数
        int: 全カード数

    """
    q = H._order
    d, k = H.shape

    # Hの列ベクトルのうち、非ゼロの要素数が偶数のものは除外する
    # (それと単位ベクトルの組み合わせで線形独立にならなくなってしまうため)

    # Hから(d-1)本の列ベクトルを全組み合わせで選択
    deck: list[set[int]] = [set() for _ in range(k)]  # decks[カードID] = {保有するシンボルID}とする
    all_combi = list(combinations(range(k), d - 1))
    for combi in tqdm(all_combi):
        # 選択した列ベクトルが線形独立か確認
        selected = H[:, combi]
        assert isinstance(selected, galois.FieldArray)
        rank = np.linalg.matrix_rank(selected)
        if rank != d - 1:
            continue
        # 線形独立である組について、その補空間のベクトルを算出(必ず次元は1となる)
        orth_vecs = selected.left_null_space()
        assert len(orth_vecs) == 1
        assert isinstance(orth_vecs, galois.FieldArray)
        orth_vec = orth_vecs[0]
        assert isinstance(orth_vec, galois.FieldArray)

        # 選択した(d-1)本のベクトル = 選択した(d-1)枚のカード, 補空間 = 共通シンボル, と解釈して
        # 各カードが保有するシンボルを記憶
        symbol_id = sum(int(a) * q**i for i, a in enumerate(orth_vec))
        for card_id in combi:
            deck[card_id].add(symbol_id)

    # シンボルIDを整理
    all_symbol_ids = set().union(*deck)  # decksの全ての要素をOR結合
    all_symbol_ids = sorted(all_symbol_ids)
    symbol_ids_map = {symbol_id: i for i, symbol_id in enumerate(all_symbol_ids)}
    sorted_decks = [sorted([symbol_ids_map[id] for id in card]) for card in deck]

    n_symbols = len(all_symbol_ids)
    n_cards = len(sorted_decks)

    return sorted_decks, n_symbols, n_cards


# Parameters
# - https://math.stackexchange.com/questions/4717536/dobble-algorithm-for-three-or-more-matches
#   - answer: https://math.stackexchange.com/a/471923#
#   - 上記のExample 3 [26, 22, 4]_5 のcode の生成多項式
#     - つまり https://www.codetables.de/BKLC/BKLC.php?q=5&n=26&k=22
n = 26  # Code length, 上記のanswerにおける m
k = 22  # Dimension, 上記のanswerにおける m-(n+1)
d = 4  # Minimum distance, 上記のanswerにおける n+1
q = 5  # 位数
generator_poly_coeffs = [1] + [0] * 21 + [3, 2, 3, 4]  # 1 + 3*x^22 + 2*x^23 + 3*x^24 + 4*x^25
alpha = 3

# Generate the code
G = generate_constacyclic_code(n, k, q, generator_poly_coeffs, alpha)
H = G.null_space()  # パリティチェック行列は生成行列GとGH^T=0を満たせば良く、つまりGの行空間に対する補空間を求める
assert G.shape == (k, n)
assert H.shape == (d, n)
assert np.linalg.matrix_rank(G) == k
assert np.linalg.matrix_rank(H) == d
assert np.all(G.dot(H.T) == 0)

# Hの任意の(d-1)本の列ベクトルが線形独立になるはずだが
# G or Hの作り方を誤っている(※)のかならないので、
# 全体で何個の組み合わせが線形独立になるかを確認
#
# NOTE:
# 「巡回符号のパリティ検査行列の作り方」に関して調べると
# 生成行列の作り方と同様に、生成多項式を(x^n-1)で割って得る多項式の係数を右シフトで作る、とあったが、
# それでも線形独立にならない組み合わせがあったため不採用とした
columns = list(combinations(range(H.shape[1]), d - 1))
n_independent: int = 0
for _columns in columns:
    selected = H[:, _columns]
    rank = np.linalg.matrix_rank(selected)
    n_independent += rank == min(selected.shape)
print(f"Combination({n}, {d-1}) = {len(columns)}個のうち {n_independent} 個の組み合わせが線形独立")

generate_dobble_deck_from_parity_check_matrix(H)
