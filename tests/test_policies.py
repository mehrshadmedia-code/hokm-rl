from hokm.cards import Card, card_to_id
from hokm.policies import AdvancedRulePolicy
from hokm.game import TEAM_BY_PLAYER
from hokm.policies import (
    RandomLegalPolicy,
    SimpleRulePolicy,
    current_winning_player,
    sort_cards_high_to_low,
    sort_cards_low_to_high,
    would_card_win_trick,
)


def test_sort_cards_low_to_high():
    cards = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("hearts", "K"),
    ]

    sorted_cards = sort_cards_low_to_high(cards)

    assert sorted_cards == [
        Card("hearts", "3"),
        Card("hearts", "K"),
        Card("hearts", "A"),
    ]


def test_sort_cards_high_to_low():
    cards = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("hearts", "K"),
    ]

    sorted_cards = sort_cards_high_to_low(cards)

    assert sorted_cards == [
        Card("hearts", "A"),
        Card("hearts", "K"),
        Card("hearts", "3"),
    ]


def test_current_winning_player_with_led_suit():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
        (2, Card("clubs", "A")),
    ]

    assert current_winning_player(current_trick, trump_suit="spades") == 1


def test_current_winning_player_with_trump():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("spades", "2")),
        (2, Card("hearts", "K")),
    ]

    assert current_winning_player(current_trick, trump_suit="spades") == 1


def test_current_winning_player_empty_trick():
    assert current_winning_player([], trump_suit="spades") is None


def test_would_card_win_trick_true_with_higher_led_suit():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    assert would_card_win_trick(
        player_id=2,
        card=Card("hearts", "A"),
        current_trick=current_trick,
        trump_suit="spades",
    )


def test_would_card_win_trick_false_with_lower_led_suit():
    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    assert not would_card_win_trick(
        player_id=2,
        card=Card("hearts", "10"),
        current_trick=current_trick,
        trump_suit="spades",
    )


def test_would_card_win_trick_true_with_trump():
    current_trick = [
        (0, Card("hearts", "A")),
        (1, Card("hearts", "K")),
    ]

    assert would_card_win_trick(
        player_id=2,
        card=Card("spades", "2"),
        current_trick=current_trick,
        trump_suit="spades",
    )


def test_simple_policy_leads_lowest_non_trump():
    policy = SimpleRulePolicy()

    hand = [
        Card("spades", "2"),
        Card("hearts", "5"),
        Card("clubs", "3"),
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=[],
        trump_suit="spades",
    )

    assert action == card_to_id(Card("clubs", "3"))


def test_simple_policy_when_partner_winning_plays_lowest_legal():
    policy = SimpleRulePolicy()

    # Player 0 and 2 are partners.
    assert TEAM_BY_PLAYER[0] == TEAM_BY_PLAYER[2]

    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("clubs", "K"),
    ]

    current_trick = [
        (1, Card("hearts", "7")),
        (2, Card("hearts", "K")),  # partner currently winning
        (3, Card("hearts", "9")),
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("hearts", "3"))


def test_simple_policy_plays_lowest_winning_card():
    policy = SimpleRulePolicy()

    hand = [
        Card("hearts", "A"),
        Card("hearts", "Q"),
        Card("clubs", "2"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    action = policy.choose_action(
        player_id=2,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("hearts", "A"))


def test_simple_policy_throws_lowest_when_cannot_win():
    policy = SimpleRulePolicy()

    hand = [
        Card("hearts", "3"),
        Card("hearts", "4"),
        Card("clubs", "A"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    action = policy.choose_action(
        player_id=2,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("hearts", "3"))

def test_advanced_policy_uses_trump_to_win_when_team_losing():
    policy = AdvancedRulePolicy()

    hand = [
        Card("spades", "2"),
        Card("clubs", "3"),
        Card("diamonds", "4"),
    ]

    current_trick = [
        (1, Card("hearts", "A")),
        (2, Card("hearts", "3")),  # partner is not winning
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("spades", "2"))


def test_advanced_policy_does_not_waste_card_when_partner_winning():
    policy = AdvancedRulePolicy()

    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("spades", "2"),
    ]

    current_trick = [
        (1, Card("hearts", "7")),
        (2, Card("hearts", "K")),  # partner winning
        (3, Card("hearts", "9")),
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("hearts", "3"))


def test_advanced_policy_leads_low_trump_when_holding_many_trumps():
    policy = AdvancedRulePolicy(trump_pressure_threshold=5)

    hand = [
        Card("spades", "2"),
        Card("spades", "4"),
        Card("spades", "6"),
        Card("spades", "8"),
        Card("spades", "10"),
        Card("hearts", "A"),
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=[],
        trump_suit="spades",
    )

    assert action == card_to_id(Card("spades", "2"))


def test_advanced_policy_leads_highest_remaining_trump_if_safe():
    policy = AdvancedRulePolicy(trump_pressure_threshold=5)

    hand = [
        Card("spades", "K"),
        Card("spades", "4"),
        Card("spades", "6"),
        Card("spades", "8"),
        Card("spades", "10"),
        Card("hearts", "A"),
    ]

    completed_tricks = [
        [(0, Card("spades", "A"))],
    ]

    action = policy.choose_action(
        player_id=0,
        hand=hand,
        current_trick=[],
        trump_suit="spades",
        completed_tricks=completed_tricks,
    )

    assert action == card_to_id(Card("spades", "K"))


def test_advanced_policy_third_player_wins_only_if_possible():
    policy = AdvancedRulePolicy()

    hand = [
        Card("hearts", "A"),
        Card("hearts", "3"),
        Card("clubs", "2"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
        (1, Card("hearts", "K")),
    ]

    action = policy.choose_action(
        player_id=2,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )

    assert action == card_to_id(Card("hearts", "A"))
    
def test_random_legal_policy_returns_legal_action():
    policy = RandomLegalPolicy(seed=42)

    hand = [
        Card("hearts", "3"),
        Card("clubs", "A"),
    ]

    current_trick = [
        (0, Card("hearts", "7")),
    ]

    action = policy.choose_action(
        player_id=1,
        hand=hand,
        current_trick=current_trick,
        trump_suit="spades",
    )


    assert action == card_to_id(Card("hearts", "3"))