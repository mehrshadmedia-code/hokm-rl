import pytest

from hokm.cards import Card, card_to_id
from hokm.observation import (
    build_player_observation,
    legal_action_mask,
    observation_to_flat_vector,
    played_cards_from_tricks,
    trump_to_vector,
)


def test_trump_to_vector():
    vector = trump_to_vector("spades")

    assert len(vector) == 4
    assert sum(vector) == 1


def test_invalid_trump_to_vector_raises_error():
    with pytest.raises(ValueError):
        trump_to_vector("stars")


def test_legal_action_mask_when_player_leads():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "2"),
        Card("spades", "K"),
    ]

    mask = legal_action_mask(hand, current_trick=[])

    assert len(mask) == 52
    assert sum(mask) == 3

    for card in hand:
        assert mask[card_to_id(card)] == 1


def test_legal_action_mask_when_player_must_follow_suit():
    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    mask = legal_action_mask(hand, current_trick)

    assert sum(mask) == 2
    assert mask[card_to_id(Card("hearts", "A"))] == 1
    assert mask[card_to_id(Card("hearts", "3"))] == 1
    assert mask[card_to_id(Card("clubs", "K"))] == 0


def test_played_cards_from_tricks_includes_completed_and_current_trick():
    completed_tricks = [
        [
            (0, Card("hearts", "A")),
            (1, Card("hearts", "K")),
            (2, Card("hearts", "Q")),
            (3, Card("hearts", "J")),
        ]
    ]

    current_trick = [
        (1, Card("clubs", "2")),
        (2, Card("clubs", "3")),
    ]

    played_cards = played_cards_from_tricks(completed_tricks, current_trick)

    assert len(played_cards) == 6
    assert Card("hearts", "A") in played_cards
    assert Card("clubs", "2") in played_cards
    assert Card("clubs", "3") in played_cards


def test_build_player_observation():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "2"),
        Card("spades", "K"),
    ]

    completed_tricks = [
        [
            (0, Card("diamonds", "2")),
            (1, Card("diamonds", "3")),
            (2, Card("diamonds", "4")),
            (3, Card("diamonds", "5")),
        ]
    ]

    current_trick = [
        (1, Card("clubs", "A")),
    ]

    observation = build_player_observation(
        player_id=2,
        hand=hand,
        completed_tricks=completed_tricks,
        current_trick=current_trick,
        trump_suit="spades",
        team_tricks={0: 1, 1: 0},
        hakem=0,
    )

    assert observation.player_id == 2
    assert len(observation.hand_vector) == 52
    assert len(observation.played_cards_vector) == 52
    assert len(observation.current_trick_vector) == 52
    assert len(observation.legal_action_mask) == 52
    assert len(observation.trump_vector) == 4
    assert observation.team_tricks == [1, 0]
    assert observation.is_hakem == 0
    assert len(observation.tactical_features) == 10


    assert sum(observation.hand_vector) == 3
    assert sum(observation.played_cards_vector) == 5
    assert sum(observation.current_trick_vector) == 1


def test_build_player_observation_marks_hakem():
    observation = build_player_observation(
        player_id=0,
        hand=[Card("hearts", "A")],
        completed_tricks=[],
        current_trick=[],
        trump_suit="hearts",
        team_tricks={0: 0, 1: 0},
        hakem=0,
    )

    assert observation.is_hakem == 1
    assert len(observation.tactical_features) == 10


def test_invalid_player_id_raises_error():
    with pytest.raises(ValueError):
        build_player_observation(
            player_id=4,
            hand=[Card("hearts", "A")],
            completed_tricks=[],
            current_trick=[],
            trump_suit="hearts",
            team_tricks={0: 0, 1: 0},
            hakem=0,
        )


def test_observation_to_flat_vector_length():
    observation = build_player_observation(
        player_id=0,
        hand=[Card("hearts", "A"), Card("clubs", "2")],
        completed_tricks=[],
        current_trick=[],
        trump_suit="hearts",
        team_tricks={0: 0, 1: 0},
        hakem=0,
    )

    flat = observation_to_flat_vector(observation)

    expected_length = 1 + 52 + 52 + 52 + 52 + 4 + 2 + 1 + 10
    assert len(flat) == expected_length