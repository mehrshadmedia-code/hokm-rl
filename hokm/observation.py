from dataclasses import dataclass
from typing import Dict, List, Sequence

from hokm.cards import Card, SUITS, card_to_id, cards_to_vector
from hokm.rules import PlayedCard, get_legal_cards
from hokm.features import build_tactical_features


@dataclass(frozen=True)
class PlayerObservation:
    """
    RL-friendly observation for one Hokm player.

    This structure only includes information the player is allowed to know:
    - their own hand
    - cards already played
    - current trick cards
    - trump suit
    - current team trick score
    - player identity/position
    - hakem identity
    - legal action mask
    """

    player_id: int
    hand_vector: List[int]
    played_cards_vector: List[int]
    current_trick_vector: List[int]
    legal_action_mask: List[int]
    trump_vector: List[int]
    team_tricks: List[int]
    is_hakem: int
    tactical_features: List[float]


def trump_to_vector(trump_suit: str) -> List[int]:
    """
    Convert trump suit to a 4-length one-hot vector.

    Suit order comes from hokm.cards.SUITS.
    """
    if trump_suit not in SUITS:
        raise ValueError(f"Invalid trump suit: {trump_suit}")

    vector = [0] * len(SUITS)
    vector[SUITS.index(trump_suit)] = 1
    return vector


def played_cards_from_tricks(
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
) -> List[Card]:
    """
    Return all cards that have already appeared in the hand.

    This includes:
    - cards in completed tricks
    - cards currently on the table in the current trick
    """
    cards: List[Card] = []

    for trick in completed_tricks:
        for _, card in trick:
            cards.append(card)

    for _, card in current_trick:
        cards.append(card)

    return cards


def legal_action_mask(
    hand: Sequence[Card],
    current_trick: Sequence[PlayedCard],
) -> List[int]:
    """
    Return a 52-length mask.

    mask[i] = 1 means card id i is legal to play.
    mask[i] = 0 means card id i is not legal to play.
    """
    mask = [0] * 52

    legal_cards = get_legal_cards(hand, current_trick)

    for card in legal_cards:
        mask[card_to_id(card)] = 1

    return mask


def build_player_observation(
    player_id: int,
    hand: Sequence[Card],
    completed_tricks: Sequence[Sequence[PlayedCard]],
    current_trick: Sequence[PlayedCard],
    trump_suit: str,
    team_tricks: Dict[int, int],
    hakem: int,
) -> PlayerObservation:
    """
    Build one player's RL-friendly observation.
    """
    if player_id not in range(4):
        raise ValueError("player_id must be from 0 to 3.")

    if hakem not in range(4):
        raise ValueError("hakem must be from 0 to 3.")

    played_cards = played_cards_from_tricks(
        completed_tricks=completed_tricks,
        current_trick=current_trick,
    )

    return PlayerObservation(
        player_id=player_id,
        hand_vector=cards_to_vector(list(hand)),
        played_cards_vector=cards_to_vector(played_cards),
        current_trick_vector=cards_to_vector([card for _, card in current_trick]),
        legal_action_mask=legal_action_mask(hand, current_trick),
        trump_vector=trump_to_vector(trump_suit),
        team_tricks=[
            team_tricks.get(0, 0),
            team_tricks.get(1, 0),
        ],
        is_hakem=1 if player_id == hakem else 0,
        tactical_features=build_tactical_features(
            player_id=player_id,
            hand=hand,
            current_trick=current_trick,
            trump_suit=trump_suit,
            team_tricks=team_tricks,
        ),
    )


def observation_to_flat_vector(observation: PlayerObservation) -> List[int]:
    """
    Convert PlayerObservation into one flat numeric vector.

    This is useful later for simple neural network inputs.
    """
    return (
        [observation.player_id]
        + observation.hand_vector
        + observation.played_cards_vector
        + observation.current_trick_vector
        + observation.legal_action_mask
        + observation.trump_vector
        + observation.team_tricks
        + [observation.is_hakem]
        + observation.tactical_features
    )