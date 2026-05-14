from random import Random
from typing import List, Optional, Sequence

from hokm.cards import Card, card_to_id, id_to_card
from hokm.game import TEAM_BY_PLAYER
from hokm.rules import PlayedCard, determine_trick_winner, get_legal_cards


def sort_cards_low_to_high(cards: Sequence[Card]) -> List[Card]:
    """
    Sort cards from low to high by rank value.
    """
    return sorted(cards, key=lambda card: card.value)


def sort_cards_high_to_low(cards: Sequence[Card]) -> List[Card]:
    """
    Sort cards from high to low by rank value.
    """
    return sorted(cards, key=lambda card: card.value, reverse=True)


def would_card_win_trick(
    player_id: int,
    card: Card,
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> bool:
    """
    Check whether this card would currently win the trick.

    This assumes the trick may not be complete yet.
    We compare the candidate card against the cards already played.
    """
    simulated_trick = list(current_trick) + [(player_id, card)]

    if len(simulated_trick) == 1:
        return True

    led_suit = simulated_trick[0][1].suit

    trump_cards = [
        played
        for played in simulated_trick
        if played[1].suit == trump_suit
    ]

    if trump_cards:
        winner, _ = max(trump_cards, key=lambda played: played[1].value)
        return winner == player_id

    led_suit_cards = [
        played
        for played in simulated_trick
        if played[1].suit == led_suit
    ]

    winner, _ = max(led_suit_cards, key=lambda played: played[1].value)
    return winner == player_id


def current_winning_player(
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> Optional[int]:
    """
    Return the player currently winning the trick.

    If the trick is empty, return None.
    """
    if not current_trick:
        return None

    if len(current_trick) == 4:
        return determine_trick_winner(current_trick, trump_suit)

    led_suit = current_trick[0][1].suit

    trump_cards = [
        played
        for played in current_trick
        if played[1].suit == trump_suit
    ]

    if trump_cards:
        winner, _ = max(trump_cards, key=lambda played: played[1].value)
        return winner

    led_suit_cards = [
        played
        for played in current_trick
        if played[1].suit == led_suit
    ]

    winner, _ = max(led_suit_cards, key=lambda played: played[1].value)
    return winner


class RandomLegalPolicy:
    """
    Choose randomly from legal actions.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = Random(seed)

    def choose_action(
        self,
        player_id: int,
        hand: Sequence[Card],
        current_trick: Sequence[PlayedCard],
        trump_suit: str,
    ) -> int:
        legal_cards = get_legal_cards(hand, current_trick)
        chosen_card = self.rng.choice(legal_cards)
        return card_to_id(chosen_card)


class SimpleRulePolicy:
    """
    A basic Hokm policy.

    Rules:
    - If leading, play the lowest non-trump card if possible.
      Otherwise play the lowest card.
    - If partner is currently winning, play the lowest legal card.
    - If this player can win the trick, play the lowest legal card that wins.
    - Otherwise play the lowest legal card.
    """

    def choose_action(
        self,
        player_id: int,
        hand: Sequence[Card],
        current_trick: Sequence[PlayedCard],
        trump_suit: str,
    ) -> int:
        legal_cards = get_legal_cards(hand, current_trick)

        if not legal_cards:
            raise ValueError("No legal cards available.")

        # Leading the trick: avoid spending trump if possible.
        if not current_trick:
            non_trump_cards = [
                card for card in legal_cards if card.suit != trump_suit
            ]

            if non_trump_cards:
                chosen_card = sort_cards_low_to_high(non_trump_cards)[0]
            else:
                chosen_card = sort_cards_low_to_high(legal_cards)[0]

            return card_to_id(chosen_card)

        winning_player = current_winning_player(current_trick, trump_suit)

        # If partner is currently winning, do not waste a high card.
        if winning_player is not None:
            my_team = TEAM_BY_PLAYER[player_id]
            winning_team = TEAM_BY_PLAYER[winning_player]

            if my_team == winning_team:
                chosen_card = sort_cards_low_to_high(legal_cards)[0]
                return card_to_id(chosen_card)

        # Try to win using the lowest winning card.
        winning_cards = [
            card
            for card in legal_cards
            if would_card_win_trick(
                player_id=player_id,
                card=card,
                current_trick=current_trick,
                trump_suit=trump_suit,
            )
        ]

        if winning_cards:
            chosen_card = sort_cards_low_to_high(winning_cards)[0]
            return card_to_id(chosen_card)

        # Otherwise throw the lowest legal card.
        chosen_card = sort_cards_low_to_high(legal_cards)[0]
        return card_to_id(chosen_card)


def action_to_card(action: int) -> Card:
    """
    Convert policy action to a card.
    """
    return id_to_card(action)