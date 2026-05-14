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