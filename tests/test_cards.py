import pytest

from hokm.cards import Card, create_deck, shuffle_deck, deal_cards


def test_card_creation():
    card = Card("hearts", "A")

    assert card.suit == "hearts"
    assert card.rank == "A"
    assert card.value == 14
    assert str(card) == "A of hearts"


def test_invalid_suit_raises_error():
    with pytest.raises(ValueError):
        Card("stars", "A")


def test_invalid_rank_raises_error():
    with pytest.raises(ValueError):
        Card("hearts", "1")


def test_create_deck_has_52_cards():
    deck = create_deck()

    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_shuffle_deck_keeps_same_cards():
    deck = create_deck()
    shuffled = shuffle_deck(deck, seed=42)

    assert len(shuffled) == 52
    assert set(shuffled) == set(deck)
    assert shuffled != deck


def test_shuffle_with_same_seed_is_reproducible():
    deck = create_deck()

    shuffled_1 = shuffle_deck(deck, seed=42)
    shuffled_2 = shuffle_deck(deck, seed=42)

    assert shuffled_1 == shuffled_2


def test_deal_cards_to_4_players():
    deck = shuffle_deck(create_deck(), seed=42)
    hands = deal_cards(deck)

    assert len(hands) == 4
    assert all(len(hand) == 13 for hand in hands)

    all_cards = [card for hand in hands for card in hand]

    assert len(all_cards) == 52
    assert len(set(all_cards)) == 52