from deck_generator.variations.shared_1_with_2_cards_named_original import (
    get_valid_params as _get_params_12_original,
)
from deck_generator.variations.shared_1_with_2_cards_named_original import (
    make_dobble_deck as _make_dobble_deck_12_original,
)
from deck_generator.variations.shared_1_with_3_cards_by_code_theory import (
    get_valid_params as _get_params_13_code,
)
from deck_generator.variations.shared_1_with_3_cards_by_code_theory import (
    make_dobble_deck as _make_dobble_deck_13_code,
)
from deck_generator.variations.shared_2_with_2_cards_by_bibd import (
    get_valid_params as _get_params_22_bibd,
)
from deck_generator.variations.shared_2_with_2_cards_by_bibd import (
    make_dobble_deck as _make_dobble_deck_22_bibd,
)


def _make_dobble_deck_22(n_symbols_per_card: int) -> tuple[list[list[int]], int]:
    """BIBDで作れるものはそれで作り、作れないものはオリジナル方式で2つ作ったデッキをガッチャンコ"""
    try:
        return _make_dobble_deck_22_bibd(n_symbols_per_card)
    except ValueError:
        if n_symbols_per_card % 2 != 0:
            raise ValueError(f"カード1枚あたりのシンボル数 {n_symbols_per_card} がデッキ構築の条件を満たしていない")
        # オリジナル版 (2枚に1つのシンボルが共通) のデッキを構築
        _n_symbols_per_card = n_symbols_per_card // 2
        try:
            deck_1, n_symbols_1 = _make_dobble_deck_12_original(_n_symbols_per_card)
        except ValueError:
            raise ValueError(f"カード1枚あたりのシンボル数 {n_symbols_per_card} がデッキ構築の条件を満たしていない")
        # 2つ目のデッキを構築 (構成は1つ目と全く同じだがシンボルが被らないように調整)
        deck_2 = [[symbol + n_symbols_1 for symbol in card] for card in deck_1]
        # 2つのデッキを結合 (両者のN枚目のカードをマージしたカードを新たなデッキのN枚目とする)
        deck = sorted([sorted(deck_1[i] + deck_2[i]) for i in range(len(deck_1))])
        n_symbols = 2 * n_symbols_1

        return deck, n_symbols


def get_valid_params(
    *,
    n_shared_symbols: int = 1,
    n_shared_cards: int = 2,
    max_n_symbols_per_card: int = 30,
    n_symbols_per_card: int | None = None,
) -> list[int] | list[tuple[int, int]]:
    """作成可能なパラメータのリストを返す

    Args:
        n_shared_symbols (int, optional):
            カード間に共通するシンボル数
        n_shared_cards (int, optional):
            シンボルを共通させたいカード数
        max_n_symbols_per_card (int, optional):
            出力する n_symbols_per_card の最大値
        n_symbols_per_card (int | None, optional):
            (n_shared_symbols, n_shared_cards) = (1, 3)の場合, 戻り値は有効な (n_symbols_per_card, n_cards) のセットを返す。
            この時 n_symbols_per_card が指定されていると、対応する n_cards のみが返る。

    Returns:
        (n_shared_smbols, n_shared_cards) != (1, 3):
            有効な n_symbols_per_card (<= max_n_symbols_per_card) のリスト (list[int])
        (n_shared_smbols, n_shared_cards) == (1, 3):
            n_symbols_per_card is None:
                有効な (n_symbols_per_card, n_cards) のリスト (list[tuple[int, int]]),
                max_n_symbols_per_cardは無効
            n_symbols_per_card is not None:
                有効な n_cards のリスト (list[int]),
                max_n_symbols_per_cardは無効
    """
    match (n_shared_symbols, n_shared_cards):
        case (1, 2):
            return _get_params_12_original(max_symbols_per_card=max_n_symbols_per_card)
        case (2, 2):
            _orig = _get_params_12_original(max_symbols_per_card=max_n_symbols_per_card)
            _bibd = _get_params_22_bibd()
            merged = {2 * x for x in _orig} | set(_bibd)
            filtered = [x for x in sorted(merged) if x <= max_n_symbols_per_card]
            return filtered
        case (1, 3):
            params = _get_params_13_code()
            if n_symbols_per_card is None:
                return params
            else:
                n_cards_list = [n_cards for n_syms, n_cards in params if n_syms == n_symbols_per_card]
                return sorted(set(n_cards_list))

        case _:
            raise ValueError(
                f"(n_shared_symbols, n_shared_cards) = ({n_shared_symbols}, {n_shared_cards}) は "
                "(1, 2), (2, 2), (1, 3) のいずれかでなければならない"
            )


def make_dobble_deck(
    n_symbols_per_card: int, *, n_shared_symbols: int = 1, n_shared_cards: int = 2, n_cards: None | int = None
) -> tuple[list[list[int]], int]:
    """ドブルデッキ構築

    Args:
        n_symbols_per_card (int):
            カード1枚あたりのシンボル数
        n_shared_symbols (int, optional):
            カード間に共通するシンボル数. Defaults to 1.
        n_shared_cards (int, optional):
            シンボルを共通させたいカード数. Defaults to 2.
        n_cards (None | int, optional):
            デッキ全体のカード数. (n_shared_symbols, n_shared_cards) = (1, 3)の場合で必須. Defaults to None.

    Raises:
        ValueError: (n_shared_symbols, n_shared_cards) = (1, 3) で n_cards がNone
        ValueError: (n_shared_symbols, n_shared_cards) が条件を満たさない

    Returns:
        list[list[int]]: 各カードに記載するシンボル番号
        int: 全シンボル数
    """
    match (n_shared_symbols, n_shared_cards):
        case (1, 2):
            pairs, n_symbols = _make_dobble_deck_12_original(n_symbols_per_card)
        case (2, 2):
            pairs, n_symbols = _make_dobble_deck_22(n_symbols_per_card)
        case (1, 3):
            if n_cards is None:
                raise ValueError("n_cardsが指定されていません")
            pairs, n_symbols = _make_dobble_deck_13_code(n_symbols_per_card, n_cards)
        case _:
            raise ValueError(
                f"(n_shared_symbols, n_shared_cards) = ({n_shared_symbols}, {n_shared_cards}) は "
                "(1, 2), (2, 2), (1, 3) のいずれかでなければならない"
            )

    return pairs, n_symbols
