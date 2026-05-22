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

def card_sort_key(card: Card) -> int:
    return card.value


def get_cards_of_suit(cards: Sequence[Card], suit: str) -> List[Card]:
    return [card for card in cards if card.suit == suit]


def get_trump_cards(cards: Sequence[Card], trump_suit: str) -> List[Card]:
    return get_cards_of_suit(cards, trump_suit)


def is_card_highest_remaining_known(
    card: Card,
    played_cards: Sequence[Card],
) -> bool:
    """
    Check whether all higher cards of this suit are already known to be gone.

    This is not perfect because we do not know opponents' hands,
    but it is useful for rule-based strategy.
    """
    higher_ranks_exist = [
        rank_value
        for rank_value in range(card.value + 1, 15)
    ]

    played_same_suit_values = {
        played_card.value
        for played_card in played_cards
        if played_card.suit == card.suit
    }

    return all(value in played_same_suit_values for value in higher_ranks_exist)


def lowest_card(cards: Sequence[Card]) -> Card:
    if not cards:
        raise ValueError("Cannot choose from empty cards.")

    return sort_cards_low_to_high(cards)[0]


def highest_card(cards: Sequence[Card]) -> Card:
    if not cards:
        raise ValueError("Cannot choose from empty cards.")

    return sort_cards_high_to_low(cards)[0]


def lowest_winning_card(
    player_id: int,
    legal_cards: Sequence[Card],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> Optional[Card]:
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

    if not winning_cards:
        return None

    return lowest_card(winning_cards)


def lowest_trump_that_wins(
    player_id: int,
    legal_cards: Sequence[Card],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> Optional[Card]:
    trump_cards = get_trump_cards(legal_cards, trump_suit)

    winning_trumps = [
        card
        for card in trump_cards
        if would_card_win_trick(
            player_id=player_id,
            card=card,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )
    ]

    if not winning_trumps:
        return None

    return lowest_card(winning_trumps)


def flatten_played_cards(
    completed_tricks: Optional[Sequence[Sequence[PlayedCard]]] = None,
    current_trick: Optional[Sequence[PlayedCard]] = None,
) -> List[Card]:
    cards: List[Card] = []

    if completed_tricks:
        for trick in completed_tricks:
            for _, card in trick:
                cards.append(card)

    if current_trick:
        for _, card in current_trick:
            cards.append(card)

    return cards

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

class AdvancedRulePolicy:
    """
    A stronger extendable Hokm rule-based policy.

    Current strategy:
    - If leading and holding many trump cards, lead trump.
    - If partner is currently winning, play the lowest legal card.
    - If partner is losing and we can trump to win, use the lowest winning trump.
    - If third to play, only spend a high/winning card if it can win.
    - If we can win normally, play the lowest winning card.
    - Otherwise play the lowest legal card.

    This class is designed so we can keep adding more Hokm rules.
    """

    def __init__(self, trump_pressure_threshold: int = 5):
        self.trump_pressure_threshold = trump_pressure_threshold

    def choose_action(
        self,
        player_id: int,
        hand: Sequence[Card],
        current_trick: Sequence[PlayedCard],
        trump_suit: str,
        completed_tricks: Optional[Sequence[Sequence[PlayedCard]]] = None,
    ) -> int:
        legal_cards = get_legal_cards(hand, current_trick)

        if not legal_cards:
            raise ValueError("No legal cards available.")

        played_cards = flatten_played_cards(
            completed_tricks=completed_tricks,
            current_trick=current_trick,
        )

        # Rule 1: If leading and holding many trump cards, apply trump pressure.
        chosen_card = self._choose_leading_card_with_trump_pressure(
            hand=hand,
            legal_cards=legal_cards,
            trump_suit=trump_suit,
            played_cards=played_cards,
        )

        if chosen_card is not None:
            return card_to_id(chosen_card)

        winning_player = current_winning_player(current_trick, trump_suit)

        # Rule 2: If partner is currently winning, do not waste a good card.
        if winning_player is not None:
            my_team = TEAM_BY_PLAYER[player_id]
            winning_team = TEAM_BY_PLAYER[winning_player]

            if my_team == winning_team:
                return card_to_id(lowest_card(legal_cards))

        # Rule 3: If our team is losing and we can trump to win, use trump.
        trump_winner = lowest_trump_that_wins(
            player_id=player_id,
            legal_cards=legal_cards,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )

        if trump_winner is not None:
            return card_to_id(trump_winner)

        # Rule 4: If third to play, play high only if it wins.
        if len(current_trick) == 2:
            winning_card = lowest_winning_card(
                player_id=player_id,
                legal_cards=legal_cards,
                current_trick=current_trick,
                trump_suit=trump_suit,
            )

            if winning_card is not None:
                return card_to_id(winning_card)

            return card_to_id(lowest_card(legal_cards))

        # Rule 5: If we can win without special trump logic, win cheaply.
        winning_card = lowest_winning_card(
            player_id=player_id,
            legal_cards=legal_cards,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )

        if winning_card is not None:
            return card_to_id(winning_card)

        # Rule 6: Otherwise throw lowest legal card.
        return card_to_id(lowest_card(legal_cards))

    def _choose_leading_card_with_trump_pressure(
        self,
        hand: Sequence[Card],
        legal_cards: Sequence[Card],
        trump_suit: str,
        played_cards: Sequence[Card],
    ) -> Optional[Card]:
        """
        If we are leading and have many trump cards, lead trump.

        If our highest trump is the highest remaining known trump, play it.
        Otherwise, play the lowest trump.
        """
        # This rule only applies when leading.
        # If all legal cards are available, current_trick is empty.
        # Since we only receive legal_cards here, check whether legal == hand.
        if set(legal_cards) != set(hand):
            return None

        trump_cards_in_hand = get_trump_cards(hand, trump_suit)

        if len(trump_cards_in_hand) < self.trump_pressure_threshold:
            return None

        legal_trumps = get_trump_cards(legal_cards, trump_suit)

        if not legal_trumps:
            return None

        highest_trump = highest_card(legal_trumps)

        if is_card_highest_remaining_known(highest_trump, played_cards):
            return highest_trump

        return lowest_card(legal_trumps)
    
    
def action_to_card(action: int) -> Card:
    """
    Convert policy action to a card.
    """
    return id_to_card(action)