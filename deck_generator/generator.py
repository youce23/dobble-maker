from deck_generator.variations.shared_1_with_2_cards_named_original import (
    make_dobble_deck as _make_dobble_deck_12_original,
)
from deck_generator.variations.shared_1_with_3_cards_by_code_theory import (
    make_dobble_deck as _make_dobble_deck_13_code,
)
from deck_generator.variations.shared_2_with_2_cards_by_bibd import (
    make_dobble_deck as _make_dobble_deck_22_bibd,
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
            pairs, n_symbols = _make_dobble_deck_22_bibd(n_symbols_per_card)
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
