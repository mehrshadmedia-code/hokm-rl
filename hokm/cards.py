from dataclasses import dataclass
from random import Random
from typing import List, Optional


SUITS = ("hearts", "diamonds", "clubs", "spades")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

    def __post_init__(self):
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")

    @property
    def value(self) -> int:
        return RANK_VALUE[self.rank]

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


def create_deck() -> List[Card]:
    return [Card(suit=suit, rank=rank) for suit in SUITS for rank in RANKS]


def shuffle_deck(deck: List[Card], seed: Optional[int] = None) -> List[Card]:
    shuffled = deck.copy()
    rng = Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def deal_cards(deck: List[Card], num_players: int = 4) -> List[List[Card]]:
    if len(deck) != 52:
        raise ValueError("Deck must contain exactly 52 cards.")

    if num_players != 4:
        raise ValueError("Hokm requires exactly 4 players.")

    hands = [[] for _ in range(num_players)]

    for index, card in enumerate(deck):
        player_id = index % num_players
        hands[player_id].append(card)

    return hands

def deal_hokm_style(deck: List[Card], hakem: int = 0) -> List[List[Card]]:
    """
    Deal cards in Hokm style.

    For now:
    - Hakem receives the first 5 cards.
    - Then the remaining cards are dealt round-robin starting from the player after Hakem.
    - Final result: each player has 13 cards.

    This supports the real flow where Hakem chooses trump after seeing 5 cards.
    """
    if len(deck) != 52:
        raise ValueError("Deck must contain exactly 52 cards.")

    if hakem not in range(4):
        raise ValueError("Hakem must be a player id from 0 to 3.")

    hands = [[] for _ in range(4)]

    # First 5 cards go to Hakem.
    hands[hakem].extend(deck[:5])

    remaining_deck = deck[5:]

    # Continue dealing until everyone has 13 cards.
    current_player = (hakem + 1) % 4

    for card in remaining_deck:
        while len(hands[current_player]) >= 13:
            current_player = (current_player + 1) % 4

        hands[current_player].append(card)
        current_player = (current_player + 1) % 4

    return hands

def card_to_id(card: Card) -> int:
    """
    Convert a card to a unique id from 0 to 51.
    """
    suit_index = SUITS.index(card.suit)
    rank_index = RANKS.index(card.rank)

    return suit_index * len(RANKS) + rank_index


def id_to_card(card_id: int) -> Card:
    """
    Convert a card id from 0 to 51 back to a Card.
    """
    if card_id < 0 or card_id >= 52:
        raise ValueError(f"Card id must be between 0 and 51. Got {card_id}.")

    suit_index = card_id // len(RANKS)
    rank_index = card_id % len(RANKS)

    return Card(
        suit=SUITS[suit_index],
        rank=RANKS[rank_index],
    )


def cards_to_vector(cards: List[Card]) -> List[int]:
    """
    Convert a list of cards to a 52-length binary vector.

    Example:
    - vector[i] = 1 means card with id i is present.
    - vector[i] = 0 means card with id i is absent.
    """
    vector = [0] * 52

    for card in cards:
        vector[card_to_id(card)] = 1

    return vector


def vector_to_cards(vector: List[int]) -> List[Card]:
    """
    Convert a 52-length binary vector back to a list of cards.
    """
    if len(vector) != 52:
        raise ValueError("Card vector must have length 52.")

    return [id_to_card(index) for index, value in enumerate(vector) if value == 1]