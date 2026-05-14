from random import Random
from typing import Optional, Sequence

from hokm.cards import Card
from hokm.rules import PlayedCard, get_legal_cards


class RandomAgent:
    """
    Simple random Hokm agent.

    This agent does not use strategy.
    It only chooses randomly from legal cards.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = Random(seed)

    def choose_trump(self, hand: Sequence[Card]) -> str:
        """
        Choose trump randomly from suits currently in the player's hand.
        """
        if not hand:
            raise ValueError("Cannot choose trump from an empty hand.")

        suits_in_hand = sorted({card.suit for card in hand})
        return self.rng.choice(suits_in_hand)

    def choose_card(
        self,
        hand: Sequence[Card],
        current_trick: Sequence[PlayedCard],
        trump_suit: str,
    ) -> Card:
        """
        Choose a random legal card.
        """
        legal_cards = get_legal_cards(hand, current_trick)

        if not legal_cards:
            raise ValueError("No legal cards available.")

        return self.rng.choice(legal_cards)