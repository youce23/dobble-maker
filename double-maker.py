from typing import List

import cv2
import numpy as np
import pandas as pd


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


def is_prime_power(value: int) -> bool:
    """素数の累乗であるかを判定

    Args:
        value (int): 入力

    Returns:
        bool: 素数の累乗ならTrue, そうでないならFalse
    """
    # 0以下の値は素数の累乗ではない
    if value <= 0:
        return False
    # 1は素数の0乗
    if value == 1:
        return True
    # 素数は素数の1乗
    if is_prime(value):
        return True

    # valueを2から平方根までの指数で割り切れるか調べる
    for exponent in range(2, int(value**0.5) + 1):
        # valueをexponentで割り切れ、かつexponentが素数の場合
        if value % exponent == 0 and is_prime(exponent):
            quotient = value
            # exponentで割り切れる限りquotientを割る
            while quotient % exponent == 0:
                quotient //= exponent
            # quotientが1になれば素数の累乗である
            if quotient == 1:
                return True

    # どの条件にも該当しない
    return False


def is_valid_n_symbols(n: int) -> bool:
    """カード1枚当たりのシンボル数が条件を満たすか

    条件: nが「素数の累乗+1」かつ「偶数」であること

    「nが偶数であること」は参考サイトに記載されている組み合わせの作り方がそれに限定しているため
    参考: ドブルの数理(3) ドブル構成8の実現 その1 位数7の有限体, https://amori.hatenablog.com/entry/2016/10/21/002032

    Args:
        n (int): 入力

    Returns:
        bool: 条件を満たせばTrue
    """
    if n <= 2:
        return False

    return n % 2 == 0 and is_prime_power(n - 1)


def make_symbol_combinations(n_symbols: int) -> List[List[int]]:
    assert is_valid_n_symbols(n_symbols)
    n = n_symbols - 1

    # 以下、n_symbolsが4の場合 (n = 3)で解説

    # a. 素のままの組み合わせ (各行がカード1枚分のシンボル(正確にはもう1要素追加される))
    # 0 1 2
    # 3 4 5
    # 6 7 8
    pairs_a = np.reshape(np.arange(n * n), [n, n])

    # b. aの転置
    # 0 3 6
    # 1 4 7
    # 2 5 8
    pairs_b = pairs_a.T

    # c. bの2列目以降をn = 1, ... n-1までローテーション
    # 0 5 7
    # 1 3 8
    # 2 4 6
    #
    # 0 4 8
    # 1 5 6
    # 2 3 7
    pairs_c: List[np.ndarray] = []
    for rot in range(1, n):
        new_pairs = pairs_b.copy()
        for col in range(1, n):
            new_pairs[:, col] = np.roll(new_pairs[:, col], rot * col)
        pairs_c.append(new_pairs)

    # あと1要素追加
    # 9 10 11 12
    pairs_ext = n * n + np.arange(n_symbols)
    # a, b, cにpairs_extの各要素を追加
    # a
    # 0 1 2 --> 0 1 2 9
    # 3 4 5 --> 3 4 5 9
    # 6 7 8 --> 6 7 8 9
    pairs_a = np.concatenate([pairs_a, np.resize(pairs_ext[0], [n, 1])], axis=1)
    # b
    # 0 3 6 --> 0 3 6 10
    # 1 4 7 --> 1 4 7 10
    # 2 5 8 --> 2 5 8 10
    pairs_b = np.concatenate([pairs_b, np.resize(pairs_ext[1], [n, 1])], axis=1)
    # 同様にcに11, 12を追加
    for i in range(len(pairs_c)):
        pairs_c[i] = np.concatenate([pairs_c[i], np.resize(pairs_ext[2 + i], [n, 1])], axis=1)

    # ext及びa, b, cの各行をカード1枚当たりのシンボルとして出力
    pairs: List[List[int]] = []
    pairs.append(pairs_ext.tolist())  # pairs_extは1次元配列で、それ以外は2次元配列のため、appendとextendを使い分ける
    pairs.extend(pairs_a.tolist())
    pairs.extend(pairs_b.tolist())
    for pr in pairs_c:
        pairs.extend(pr.tolist())

    assert len(pairs) == n_symbols * (n_symbols - 1) + 1  # 参考サイトを参照

    if __debug__:
        _n = len(pairs)
        for i in range(_n - 1):
            for j in range(i + 1, _n):
                # 任意のカードを2枚選んだ時、必ず共通するシンボルが1つだけ見つかる
                assert len(set(pairs[i]) & set(pairs[j])) == 1

    return pairs


def main():
    n_symbols: int = 10  # カード1枚当たりのシンボル数
    # 入力チェック
    if not is_valid_n_symbols(n_symbols):
        raise ValueError(f"カード1枚当たりのシンボル数 ({n_symbols}) が「任意の素数の累乗+1」かつ「偶数」ではない")

    # 各カード毎の組み合わせを生成
    pairs = make_symbol_combinations(n_symbols)

    return


if __name__ == "__main__":
    main()
