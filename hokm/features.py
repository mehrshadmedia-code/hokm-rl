from typing import Dict, List, Sequence

from hokm.cards import Card
from hokm.game import TEAM_BY_PLAYER
from hokm.rules import PlayedCard, get_legal_cards
from hokm.policies import current_winning_player, would_card_win_trick


def count_legal_trump_cards(
    hand: Sequence[Card],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    legal_cards = get_legal_cards(hand, current_trick)
    return sum(1 for card in legal_cards if card.suit == trump_suit)


def count_legal_winning_cards(
    player_id: int,
    hand: Sequence[Card],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    legal_cards = get_legal_cards(hand, current_trick)

    return sum(
        1
        for card in legal_cards
        if would_card_win_trick(
            player_id=player_id,
            card=card,
            current_trick=current_trick,
            trump_suit=trump_suit,
        )
    )


def is_partner_currently_winning(
    player_id: int,
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    winner = current_winning_player(current_trick, trump_suit)

    if winner is None:
        return 0

    return int(
        TEAM_BY_PLAYER[winner] == TEAM_BY_PLAYER[player_id]
        and winner != player_id
    )


def is_player_currently_winning(
    player_id: int,
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    winner = current_winning_player(current_trick, trump_suit)

    if winner is None:
        return 0

    return int(winner == player_id)


def is_opponent_currently_winning(
    player_id: int,
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
) -> int:
    winner = current_winning_player(current_trick, trump_suit)

    if winner is None:
        return 0

    return int(TEAM_BY_PLAYER[winner] != TEAM_BY_PLAYER[player_id])


def is_last_to_play_in_trick(current_trick: Sequence[PlayedCard]) -> int:
    return int(len(current_trick) == 3)


def cards_left_in_current_trick(current_trick: Sequence[PlayedCard]) -> int:
    return 4 - len(current_trick)


def normalize_count(value: int, max_value: int) -> float:
    if max_value <= 0:
        raise ValueError("max_value must be positive.")

    return value / max_value


def build_tactical_features(
    player_id: int,
    hand: Sequence[Card],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
    team_tricks: Dict[int, int],
) -> List[float]:
    """
    Build extra model-friendly tactical features.

    Feature order:
    0. is_partner_currently_winning
    1. is_player_currently_winning
    2. is_opponent_currently_winning
    3. is_last_to_play_in_trick
    4. cards_left_in_current_trick normalized by 4
    5. legal_cards_count normalized by 13
    6. legal_trump_cards_count normalized by 13
    7. legal_winning_cards_count normalized by 13
    8. my_team_tricks normalized by 7
    9. opponent_team_tricks normalized by 7
    """
    if player_id not in range(4):
        raise ValueError("player_id must be from 0 to 3.")

    legal_cards = get_legal_cards(hand, current_trick)

    my_team = TEAM_BY_PLAYER[player_id]
    opponent_team = 1 - my_team

    legal_trump_count = count_legal_trump_cards(
        hand=hand,
        current_trick=current_trick,
        trump_suit=trump_suit,
    )

    legal_winning_count = count_legal_winning_cards(
        player_id=player_id,
        hand=hand,
        current_trick=current_trick,
        trump_suit=trump_suit,
    )

    return [
        float(is_partner_currently_winning(player_id, current_trick, trump_suit)),
        float(is_player_currently_winning(player_id, current_trick, trump_suit)),
        float(is_opponent_currently_winning(player_id, current_trick, trump_suit)),
        float(is_last_to_play_in_trick(current_trick)),
        normalize_count(cards_left_in_current_trick(current_trick), 4),
        normalize_count(len(legal_cards), 13),
        normalize_count(legal_trump_count, 13),
        normalize_count(legal_winning_count, 13),
        normalize_count(team_tricks.get(my_team, 0), 7),
        normalize_count(team_tricks.get(opponent_team, 0), 7),
    ]