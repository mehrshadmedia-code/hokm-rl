import pytest

from hokm.cards import Card
from hokm.features import (
    build_tactical_features,
    cards_left_in_current_trick,
    count_legal_trump_cards,
    count_legal_winning_cards,
    is_last_to_play_in_trick,
    is_opponent_currently_winning,
    is_partner_currently_winning,
    is_player_currently_winning,
    normalize_count,
)


def test_count_legal_trump_cards_when_player_can_play_anything():
    hand = [
        Card("spades", "2"),
        Card("spades", "A"),
        Card("hearts", "3"),
    ]

    assert count_legal_trump_cards(hand, [], trump_suit="spades") == 2


def test_count_legal_trump_cards_when_player_must_follow_suit():
    hand = [
        Card("spades", "2"),
        Card("hearts", "A"),
        Card("hearts", "3"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    assert count_legal_trump_cards(hand, current_trick, trump_suit="spades") == 0


def test_count_legal_winning_cards():
    hand = [
        Card("hearts", "A"),
        Card("hearts", "Q"),
        Card("clubs", "2"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    count = count_legal_winning_cards(
        player_id=2,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert count == 1


def test_partner_currently_winning():
    # Player 0 and 2 are partners.
    current_trick = [
        (1, Card("hearts", "7")),
        (2, Card("hearts", "K")),
    ]

    assert is_partner_currently_winning(
        player_id=0,
        current_trick=current_trick,
        trump_suit="spades",
    ) == 1


def test_player_currently_winning():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("hearts", "K")),
    ]

    assert is_player_currently_winning(
        player_id=0,
        current_trick=current_trick,
        trump_suit="spades",
    ) == 1


def test_opponent_currently_winning():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "A")),
    ]

    assert is_opponent_currently_winning(
        player_id=0,
        current_trick=current_trick,
        trump_suit="spades",
    ) == 1


def test_empty_trick_has_no_current_winner_flags():
    assert is_partner_currently_winning(0, [], "spades") == 0
    assert is_player_currently_winning(0, [], "spades") == 0
    assert is_opponent_currently_winning(0, [], "spades") == 0


def test_is_last_to_play_in_trick():
    assert is_last_to_play_in_trick([]) == 0
    assert is_last_to_play_in_trick([(0, Card("hearts", "A"))]) == 0
    assert is_last_to_play_in_trick(
        [
            (0, Card("hearts", "A")),
            (1, Card("hearts", "K")),
            (2, Card("hearts", "Q")),
        ]
    ) == 1


def test_cards_left_in_current_trick():
    assert cards_left_in_current_trick([]) == 4
    assert cards_left_in_current_trick([(0, Card("hearts", "A"))]) == 3


def test_normalize_count():
    assert normalize_count(0, 4) == 0
    assert normalize_count(2, 4) == 0.5
    assert normalize_count(4, 4) == 1


def test_normalize_count_invalid_max_value_raises_error():
    with pytest.raises(ValueError):
        normalize_count(1, 0)


def test_build_tactical_features_length_and_values():
    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("clubs", "2"),
    ]

    current_trick = [
        (1, Card("hearts", "7")),
        (2, Card("hearts", "K")),
    ]

    features = build_tactical_features(
        player_id=0,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
        team_tricks={0: 2, 1: 3},
    )

    assert len(features) == 10

    # Partner player 2 is currently winning.
    assert features[0] == 1.0

    # Player 0 is not currently winning.
    assert features[1] == 0.0

    # Opponent is not currently winning.
    assert features[2] == 0.0

    # Two cards have already been played, so player is not last.
    assert features[3] == 0.0

    # 2 cards left in trick, normalized by 4.
    assert features[4] == 0.5

    # Must follow hearts, so 2 legal cards from 3-card hand.
    assert features[5] == 2 / 13

    # No legal trump because player must follow hearts.
    assert features[6] == 0

    # A hearts ace can win.
    assert features[7] == 1 / 13

    # Team score.
    assert features[8] == 2 / 7
    assert features[9] == 3 / 7


def test_invalid_player_id_raises_error():
    with pytest.raises(ValueError):
        build_tactical_features(
            player_id=4,
            hand=[Card("hearts", "A")],
            current_trick=[],
            trump_suit="hearts",
            team_tricks={0: 0, 1: 0},
        )