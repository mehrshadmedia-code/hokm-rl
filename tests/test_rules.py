import pytest

from hokm.cards import Card
from hokm.rules import get_legal_cards, is_legal_play, determine_trick_winner


def test_leading_player_can_play_any_card():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "2"),
        Card("spades", "10"),
    ]

    legal_cards = get_legal_cards(hand, current_trick=[])

    assert set(legal_cards) == set(hand)


def test_player_must_follow_led_suit_if_possible():
    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    legal_cards = get_legal_cards(hand, current_trick)

    assert set(legal_cards) == {
        Card("hearts", "A"),
        Card("hearts", "3"),
    }


def test_player_can_play_any_card_if_no_led_suit_in_hand():
    hand = [
        Card("clubs", "K"),
        Card("spades", "10"),
        Card("diamonds", "2"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    legal_cards = get_legal_cards(hand, current_trick)

    assert set(legal_cards) == set(hand)


def test_is_legal_play_true_for_valid_card():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    assert is_legal_play(Card("hearts", "A"), hand, current_trick)


def test_is_legal_play_false_when_player_must_follow_suit():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    assert not is_legal_play(Card("clubs", "K"), hand, current_trick)


def test_is_legal_play_false_when_card_not_in_hand():
    hand = [
        Card("hearts", "A"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    assert not is_legal_play(Card("spades", "Q"), hand, current_trick)


def test_highest_led_suit_wins_when_no_trump_played():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
        (2, Card("hearts", "10")),
        (3, Card("hearts", "2")),
    ]

    winner = determine_trick_winner(current_trick, trump_suit="spades")

    assert winner == 1


def test_off_suit_high_card_does_not_win_without_trump():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("clubs", "A")),
        (2, Card("hearts", "10")),
        (3, Card("diamonds", "K")),
    ]

    winner = determine_trick_winner(current_trick, trump_suit="spades")

    assert winner == 2


def test_trump_card_wins_over_led_suit():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("hearts", "K")),
        (2, Card("spades", "2")),
        (3, Card("hearts", "Q")),
    ]

    winner = determine_trick_winner(current_trick, trump_suit="spades")

    assert winner == 2


def test_highest_trump_wins_if_multiple_trumps_played():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("spades", "3")),
        (2, Card("spades", "K")),
        (3, Card("spades", "7")),
    ]

    winner = determine_trick_winner(current_trick, trump_suit="spades")

    assert winner == 2


def test_trick_must_have_four_cards():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("hearts", "K")),
    ]

    with pytest.raises(ValueError):
        determine_trick_winner(current_trick, trump_suit="spades")


def test_empty_hand_raises_error_for_legal_cards():
    with pytest.raises(ValueError):
        get_legal_cards([], current_trick=[])