import galois
import numpy as np


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
    if not is_valid_n_symbols_per_card(n_symbols_per_card):
        raise ValueError(
            f"カード1枚当たりのシンボル数 ({n_symbols_per_card}) が「2 あるいは (任意の素数の累乗 + 1)」ではない"
        )

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
