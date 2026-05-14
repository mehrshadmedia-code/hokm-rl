from typing import List, Sequence, Tuple

from hokm.cards import Card


PlayedCard = Tuple[int, Card]


def get_legal_cards(hand: Sequence[Card], current_trick: Sequence[PlayedCard]) -> List[Card]:
    """
    Return the cards a player is legally allowed to play.

    Hokm rule:
    - If the player leads the trick, they can play any card.
    - If a suit has already been led, the player must follow that suit if possible.
    - If they do not have the led suit, they can play any card.
    """
    if not hand:
        raise ValueError("Hand cannot be empty.")

    if not current_trick:
        return list(hand)

    led_suit = current_trick[0][1].suit
    same_suit_cards = [card for card in hand if card.suit == led_suit]

    if same_suit_cards:
        return same_suit_cards

    return list(hand)


def determine_trick_winner(
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    """
    Determine which player wins a completed trick.

    Args:
        current_trick:
            A sequence of (player_id, card) pairs in the order cards were played.
        trump_suit:
            The selected Hokm/trump suit.

    Returns:
        The player_id of the trick winner.
    """
    if len(current_trick) != 4:
        raise ValueError("A completed Hokm trick must contain exactly 4 played cards.")

    led_suit = current_trick[0][1].suit

    trump_cards = [
        (player_id, card)
        for player_id, card in current_trick
        if card.suit == trump_suit
    ]

    if trump_cards:
        winner_player_id, _ = max(trump_cards, key=lambda played: played[1].value)
        return winner_player_id

    led_suit_cards = [
        (player_id, card)
        for player_id, card in current_trick
        if card.suit == led_suit
    ]

    winner_player_id, _ = max(led_suit_cards, key=lambda played: played[1].value)
    return winner_player_id


def is_legal_play(
    card: Card,
    hand: Sequence[Card],
    current_trick: Sequence[PlayedCard],
) -> bool:
    """
    Check whether a selected card is legal.
    """
    if card not in hand:
        return False

    legal_cards = get_legal_cards(hand, current_trick)
    return card in legal_cards