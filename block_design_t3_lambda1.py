"""3枚に1つのシンボルが共通するデッキを符号理論に基づき生成

以下の手順で構築したものを関数でまとめた。

1. 条件に合う符号(code)を Code Tables で検索
    - Code Tables: Bounds on the parameters of various types of codes
        - https://www.codetables.de/
    - 「d枚に1つのシンボルが共通」するデッキにする場合、[n, k, d]_GF(q)は[n, n-(d+1), d]_GF(q)を満たす必要がある
    - ここではd = 3なので、上記サイトより[n, n-4, 3]_GF(q)の符号を検索する
        - "Complete tables with bounds on linear codes [n,k,d] over GF(q)"というコンテンツで、
        GF(q)毎に全てのn, k, dのセットがテーブル形式で一覧できるので、ここから探すのが良い
2. 1で見つけた算出方法は数値計算ソフト MAGMA を使うことを前提とているため、MAGMA calculatorで実装する
    - MAGMA calculator
        - http://magma.maths.usyd.edu.au/calc/
    - 以下の手順で実装するとラク
        - ChatGPT あるいは Claude などで「以下をMAGMAで実装してください。{1のConstructionを貼り付け}」としてプログラムを生成
        - MAGMA calculator に張り付けて実行し、エラーが出たら再度 ChatGPT / Claude に質問
        - 適宜 MAGMA のドキュメントも参照しながら修正する
            - http://magma.maths.usyd.edu.au/magma/handbook/
3. 上記で符号の 生成行列 G 及び 検査行列 H を生成
4. 検査行列 H からデッキを構築する
"""

from itertools import combinations
from typing import Literal

import galois
import numpy as np
from tqdm import tqdm


def generate_code_26_22_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [26, 22, 4] 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=5&n=26&k=22>

    ```
    Construction of a linear code [26,22,4] over GF(5):
    [1]:
        [26, 22, 4] Constacyclic by 3 Linear Code over GF(5)
        ConstaCyclicCode generated by 4*x^25 + 3*x^24 + 2*x^23 + 3*x^22 + 1 with shift constant 3
    ```

    実装

    ```MAGMA
    // 変数定義
    q := 5;  // 位数
    F<x> := PolynomialRing(GF(q));  // GF(q)上の多項式環を定義
    g := 4*x^25 + 3*x^24 + 2*x^23 + 3*x^22 + 1;  // 生成多項式
    shift := GF(q)!3;  // シフト定数
    n := 26;  // 符号長

    // 定数巡回符号を構築
    C := ConstaCyclicCode(n, g, shift);

    // 符号の情報を表示
    print "Code C:";
    print C;

    // 符号のパラメータを表示 (Parameters(C)でエラーが生じたため個別に表示)
    print "Parameters of C:";
    print "Length:", Length(C);
    print "Dimension:", Dimension(C);
    print "Minimum Distance:", MinimumDistance(C);

    // 生成行列, 検査行列を生成し表示
    G := GeneratorMatrix(C);
    H := ParityCheckMatrix(C);
    print "Generator matrix of C:";
    print G;
    print "Parity check matrix of C:";
    print H;

    // 情報表示
    print "Dimension of C:", Dimension(C);  // 次元
    print "Minimum distance of C:", MinimumDistance(C);  // 最小距離

    // 生成行列と検査行列の積が0になることを確認
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```

    実行結果

    ```
    Code C:
    [26, 22] Constacyclic by 3 Linear Code over GF(5)
    Generator matrix:
    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 2 3 4]
    [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 1 0]
    [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 1]
    [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 0 2]
    [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 1 1]
    [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 4]
    [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 2 0 2]
    [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 4 1]
    [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 3 2]
    [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 2 4 3 4]
    [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 4 3 3 0]
    [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 4 3 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 3 2 2 2]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 1 4 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 3 4 4 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 3 0 2 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 3 0 3 1]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 2 1 1]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 3 4]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 4 2 4 0]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 4 2 4]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 1 3 4]
    Parameters of C:
    Length: 26
    Dimension: 22
    Minimum Distance: 4
    Generator matrix of C:
    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 2 3 4]
    [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 1 0]
    [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 1]
    [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 0 2]
    [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 1 1]
    [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 4]
    [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 2 0 2]
    [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 4 1]
    [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 3 2]
    [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 2 4 3 4]
    [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 4 3 3 0]
    [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 4 3 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 3 2 2 2]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 1 4 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 3 4 4 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 3 0 2 3]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 3 0 3 1]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 2 1 1]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 3 4]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 4 2 4 0]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 4 2 4]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 1 3 4]
    Parity check matrix of C:
    [1 0 0 0 1 3 0 2 4 2 3 4 2 4 3 0 1 4 1 1 1 2 2 3 0 3]
    [0 1 0 0 4 3 3 3 3 2 4 4 2 3 1 3 4 2 3 0 0 4 0 4 3 2]
    [0 0 1 0 1 2 3 0 2 0 0 3 1 1 1 1 4 3 3 4 1 2 1 3 4 1]
    [0 0 0 1 3 0 2 4 2 3 4 2 4 3 0 1 4 1 1 1 2 2 3 0 3 3]
    Dimension of C: 22
    Minimum distance of C: 4
    G * Transpose(H) = 0: true
    ```
    """
    gf = galois.GF(5)
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 4],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 2],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 4, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 2, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 3, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 2, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 2, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 3, 4],
    ]
    H = [
        [1, 0, 0, 0, 1, 3, 0, 2, 4, 2, 3, 4, 2, 4, 3, 0, 1, 4, 1, 1, 1, 2, 2, 3, 0, 3],
        [0, 1, 0, 0, 4, 3, 3, 3, 3, 2, 4, 4, 2, 3, 1, 3, 4, 2, 3, 0, 0, 4, 0, 4, 3, 2],
        [0, 0, 1, 0, 1, 2, 3, 0, 2, 0, 0, 3, 1, 1, 1, 1, 4, 3, 3, 4, 1, 2, 1, 3, 4, 1],
        [0, 0, 0, 1, 3, 0, 2, 4, 2, 3, 4, 2, 4, 3, 0, 1, 4, 1, 1, 1, 2, 2, 3, 0, 3, 3],
    ]

    return gf(G), gf(H)


def generate_deck_from_parity_check_matrix(H: galois.FieldArray) -> tuple[list[list[int]], int, int, int]:
    """パリティ検査行列に従ってドブルデッキ構築

    Args:
        H: パリティ検査行列

    Returns:
        list[list[int]: デッキ
        int: 全シンボル数
        int: 全カード数
        int: カード1枚当たりのシンボル数

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
    sorted_deck = [sorted([symbol_ids_map[id] for id in card]) for card in deck]
    sorted_deck = sorted(sorted_deck)

    n_symbols_per_card = len(sorted_deck[0])
    n_symbols = len(all_symbol_ids)
    n_cards = len(sorted_deck)

    assert all([len(card) == n_symbols_per_card for card in deck])

    return sorted_deck, n_symbols, n_cards, n_symbols_per_card


def _check_columns_independent_of_parity_check_matrix(H: galois.FieldArray) -> bool:
    """検査行列 H の(d-1)本の列ベクトルが線形独立であるかを確認

    全組み合わせが線形独立ならTrue
    """
    d, n = H.shape

    columns = list(combinations(range(H.shape[1]), d - 1))
    n_independent: int = 0
    for _columns in columns:
        selected = H[:, _columns]
        rank = np.linalg.matrix_rank(selected)
        n_independent += rank == min(selected.shape)
    print(f"Combination({n}, {d-1}) = {len(columns)}個のうち {n_independent} 個の組み合わせが線形独立")

    return n_independent == len(columns)


def make_dobble31_deck(param: tuple[Literal[30], Literal[26]]) -> tuple[list[list[int]], int]:
    """各カードに記載するシンボルの一覧を生成

    3枚のカードに1つの共通するシンボルが出現する拡張ドブル版

    Args:
        param[0]: カード1枚あたりに記載するシンボル数
        param[1]: 全カード数

    Returns:
        list[list[int]]: デッキ (各カードに記載するシンボル番号)
        int: 全シンボル数
    """
    n_symbols_per_card, n_cards = param
    match (n_symbols_per_card, n_cards):
        # n: Code length, 上記のanswerにおける m
        # k: Dimension, 上記のanswerにおける m-(n+1)
        # d: Minimum distance, 上記のanswerにおける n+1
        # q: 位数
        case (30, 26):
            n, k, d, q = (26, 22, 4, 5)  # noqa qは未使用だが記録目的で記載
            G, H = generate_code_26_22_4_GF5()
        case _:
            raise ValueError

    # 条件を満たした行列か確認
    assert G.shape == (k, n)
    assert H.shape == (d, n)
    assert np.linalg.matrix_rank(G) == k
    assert np.linalg.matrix_rank(H) == d
    assert np.all(G.dot(H.T) == 0)
    assert _check_columns_independent_of_parity_check_matrix(H)

    deck, n_symbols, _, _ = generate_deck_from_parity_check_matrix(H)

    return deck, n_symbols


def main() -> None:
    n_symbols_per_card, n_card = (30, 26)

    deck, n_symbols = make_dobble31_deck((n_symbols_per_card, n_card))
    print(f"n_symbols_per_card: {n_symbols_per_card}")
    print(f"n_card: {n_card}")
    print(f"n_symbols: {n_symbols}")
    print("deck:")
    print(deck)

    return


if __name__ == "__main__":
    main()
