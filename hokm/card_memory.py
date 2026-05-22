from typing import Dict, List, Sequence, Set

from hokm.cards import Card, SUITS, RANKS, RANK_VALUE
from hokm.rules import PlayedCard


def flatten_played_cards(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> List[Card]:
    cards: List[Card] = []

    for trick in completed_tricks:
        for _, card in trick:
            cards.append(card)

    for _, card in current_trick:
        cards.append(card)

    return cards


def infer_void_suits(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> Dict[int, Set[str]]:
    """
    Infer which suits each player is likely void in.

    Hokm rule:
    If a suit is led and a player does not follow that suit,
    then that player has no cards left in that suit.
    """
    void_suits: Dict[int, Set[str]] = {
        0: set(),
        1: set(),
        2: set(),
        3: set(),
    }

    all_tricks = list(completed_tricks)

    if current_trick:
        all_tricks.append(current_trick)

    for trick in all_tricks:
        if not trick:
            continue

        led_suit = trick[0][1].suit

        for player_id, card in trick[1:]:
            if card.suit != led_suit:
                void_suits[player_id].add(led_suit)

    return void_suits


def void_suits_to_vector(void_suits: Dict[int, Set[str]]) -> List[int]:
    """
    Convert void suit info to 16 values.

    Order:
    Player 0: hearts, diamonds, clubs, spades
    Player 1: hearts, diamonds, clubs, spades
    Player 2: hearts, diamonds, clubs, spades
    Player 3: hearts, diamonds, clubs, spades
    """
    vector: List[int] = []

    for player_id in range(4):
        for suit in SUITS:
            vector.append(1 if suit in void_suits.get(player_id, set()) else 0)

    return vector


def remaining_cards_by_suit_vector(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> List[float]:
    """
    Return 4 normalized counts for unseen cards in each suit.

    Each suit starts with 13 cards.
    """
    played_cards = flatten_played_cards(completed_tricks, current_trick)

    played_count_by_suit = {suit: 0 for suit in SUITS}

    for card in played_cards:
        played_count_by_suit[card.suit] += 1

    return [
        (13 - played_count_by_suit[suit]) / 13
        for suit in SUITS
    ]


def highest_remaining_card_value_by_suit_vector(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> List[float]:
    """
    Return 4 normalized values showing the highest unplayed card in each suit.

    Example:
    If A, K, Q, J of hearts are gone, highest remaining hearts is 10.
    Normalized value = 10 / 14.
    """
    played_cards = flatten_played_cards(completed_tricks, current_trick)

    played_values_by_suit = {
        suit: set()
        for suit in SUITS
    }

    for card in played_cards:
        played_values_by_suit[card.suit].add(card.value)

    result: List[float] = []

    for suit in SUITS:
        remaining_values = [
            RANK_VALUE[rank]
            for rank in RANKS
            if RANK_VALUE[rank] not in played_values_by_suit[suit]
        ]

        if remaining_values:
            result.append(max(remaining_values) / 14)
        else:
            result.append(0.0)

    return result


def build_memory_features(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> List[float]:
    """
    Memory features for RL observation.

    Feature size:
    - 16 void suit values
    - 4 remaining-card-count-by-suit values
    - 4 highest-remaining-card-by-suit values

    Total: 24
    """
    void_suits = infer_void_suits(completed_tricks, current_trick)

    return (
        [float(value) for value in void_suits_to_vector(void_suits)]
        + remaining_cards_by_suit_vector(completed_tricks, current_trick)
        + highest_remaining_card_value_by_suit_vector(completed_tricks, current_trick)
    )