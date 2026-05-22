from hokm.cards import Card
from hokm.card_memory import (
    build_memory_features,
    highest_remaining_card_value_by_suit_vector,
    infer_void_suits,
    remaining_cards_by_suit_vector,
    void_suits_to_vector,
)


def test_infer_void_suits_from_completed_trick():
    completed_tricks = [
        [
            (0, Card("spades", "A")),
            (1, Card("spades", "2")),
            (2, Card("hearts", "3")),  # player 2 is void in spades
            (3, Card("spades", "K")),
        ]
    ]

    void_suits = infer_void_suits(completed_tricks, current_trick=[])

    assert "spades" in void_suits[2]
    assert "spades" not in void_suits[1]
    assert "spades" not in void_suits[3]


def test_infer_void_suits_from_current_trick():
    completed_tricks = []

    current_trick = [
        (0, Card("clubs", "A")),
        (1, Card("diamonds", "2")),  # player 1 is void in clubs
    ]

    void_suits = infer_void_suits(completed_tricks, current_trick)

    assert "clubs" in void_suits[1]


def test_void_suits_to_vector_length():
    void_suits = {
        0: {"hearts"},
        1: {"spades"},
        2: set(),
        3: {"clubs", "diamonds"},
    }

    vector = void_suits_to_vector(void_suits)

    assert len(vector) == 16
    assert sum(vector) == 4


def test_remaining_cards_by_suit_vector():
    completed_tricks = [
        [
            (0, Card("hearts", "A")),
            (1, Card("hearts", "K")),
            (2, Card("clubs", "2")),
            (3, Card("spades", "3")),
        ]
    ]

    vector = remaining_cards_by_suit_vector(completed_tricks, current_trick=[])

    assert len(vector) == 4

    # hearts has 2 played, so 11 remain
    assert vector[0] == 11 / 13


def test_highest_remaining_card_value_by_suit_vector():
    completed_tricks = [
        [
            (0, Card("hearts", "A")),
            (1, Card("hearts", "K")),
            (2, Card("hearts", "Q")),
            (3, Card("hearts", "J")),
        ]
    ]

    vector = highest_remaining_card_value_by_suit_vector(
        completed_tricks,
        current_trick=[],
    )

    assert len(vector) == 4

    # hearts highest remaining should be 10
    assert vector[0] == 10 / 14


def test_build_memory_features_length():
    completed_tricks = [
        [
            (0, Card("spades", "A")),
            (1, Card("spades", "2")),
            (2, Card("hearts", "3")),
            (3, Card("spades", "K")),
        ]
    ]

    features = build_memory_features(completed_tricks, current_trick=[])

    assert len(features) == 24