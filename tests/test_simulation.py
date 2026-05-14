from scripts.simulate_random_env_games import play_random_hand, simulate_random_hands


def test_play_random_hand_finishes():
    result = play_random_hand(seed=42)

    assert result["winning_team"] in [0, 1]
    assert 7 in result["team_tricks"].values()
    assert result["steps"] <= 52
    assert result["completed_tricks"] <= 13
    assert result["trump_suit"] in ["hearts", "diamonds", "clubs", "spades"]


def test_simulate_random_hands_completes_without_crashes():
    stats = simulate_random_hands(num_hands=20, seed=42)

    assert stats["num_hands_requested"] == 20
    assert stats["num_hands_completed"] == 20
    assert stats["crashes"] == 0
    assert stats["team_0_wins"] + stats["team_1_wins"] == 20
    assert 0 <= stats["team_0_win_rate"] <= 1
    assert 0 <= stats["team_1_win_rate"] <= 1
    assert stats["avg_steps"] <= 52
    assert stats["avg_completed_tricks"] <= 13