from hokm.agents import RandomAgent
from hokm.game import HokmHand, TEAM_BY_PLAYER


def make_agents():
    return [
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        RandomAgent(seed=3),
        RandomAgent(seed=4),
    ]


def test_play_order_from_player_zero():
    assert HokmHand.get_play_order(0) == [0, 1, 2, 3]


def test_play_order_from_player_two():
    assert HokmHand.get_play_order(2) == [2, 3, 0, 1]


def test_random_hand_finishes_with_winner():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    result = hand.play()

    assert result.winning_team in [0, 1]
    assert result.team_tricks[result.winning_team] == 7
    assert sum(result.team_tricks.values()) <= 13
    assert len(result.tricks) <= 13


def test_each_completed_trick_has_four_cards():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    result = hand.play()

    for trick in result.tricks:
        assert len(trick.cards_played) == 4


def test_cards_are_removed_from_hands_after_play():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    result = hand.play()
    cards_played_count = len(result.tricks) * 4
    remaining_cards_count = sum(len(player_hand) for player_hand in hand.hands)

    assert cards_played_count + remaining_cards_count == 52


def test_winning_team_matches_winner_player():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    result = hand.play()

    for trick in result.tricks:
        assert trick.winning_team == TEAM_BY_PLAYER[trick.winner]


def test_final_score_has_one_team_with_seven_tricks():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    result = hand.play()

    assert 7 in result.team_tricks.values()

def test_hakem_chooses_trump_from_first_five_cards():
    hand = HokmHand(
        agents=make_agents(),
        hakem=0,
        seed=42,
    )

    hand.setup()

    first_five_suits = {card.suit for card in hand.hands[0][:5]}

    assert hand.trump_suit in first_five_suits