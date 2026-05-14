from scripts.simulate_policy_games import (
    play_hand_with_policies,
    simulate_matchup,
)
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy


def test_play_hand_with_simple_policies_finishes():
    policies = {
        0: SimpleRulePolicy(),
        1: SimpleRulePolicy(),
        2: SimpleRulePolicy(),
        3: SimpleRulePolicy(),
    }

    result = play_hand_with_policies(seed=42, policies=policies)

    assert result["winning_team"] in [0, 1]
    assert 7 in result["team_tricks"].values()
    assert result["steps"] <= 52
    assert result["completed_tricks"] <= 13


def test_play_hand_with_mixed_policies_finishes():
    policies = {
        0: SimpleRulePolicy(),
        2: SimpleRulePolicy(),
        1: RandomLegalPolicy(seed=1),
        3: RandomLegalPolicy(seed=3),
    }

    result = play_hand_with_policies(seed=42, policies=policies)

    assert result["winning_team"] in [0, 1]
    assert 7 in result["team_tricks"].values()


def test_simulate_matchup_finishes():
    stats = simulate_matchup(
        num_hands=20,
        seed=42,
        team_0_policy_name="simple",
        team_1_policy_name="random",
    )

    assert stats["hands"] == 20
    assert stats["team_0_wins"] + stats["team_1_wins"] == 20
    assert 0 <= stats["team_0_win_rate"] <= 1
    assert 0 <= stats["team_1_win_rate"] <= 1