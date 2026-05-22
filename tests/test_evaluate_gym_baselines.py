import pytest

from scripts.evaluate_gym_baselines import (
    choose_baseline_action,
    evaluate_baseline,
    run_one_episode,
)


def test_choose_first_legal_action():
    action = choose_baseline_action(
        policy_name="first_legal",
        legal_actions=[5, 10, 20],
        rng=None,
    )

    assert action == 5


def test_choose_random_legal_action():
    import random

    rng = random.Random(42)

    action = choose_baseline_action(
        policy_name="random_legal",
        legal_actions=[5, 10, 20],
        rng=rng,
    )

    assert action in [5, 10, 20]


def test_unknown_baseline_policy_raises_error():
    with pytest.raises(ValueError):
        choose_baseline_action(
            policy_name="bad_policy",
            legal_actions=[1, 2, 3],
            rng=None,
        )


def test_empty_legal_actions_raises_error():
    with pytest.raises(ValueError):
        choose_baseline_action(
            policy_name="first_legal",
            legal_actions=[],
            rng=None,
        )


def test_run_one_episode_finishes():
    result = run_one_episode(
        seed=42,
        learning_policy_name="first_legal",
        opponent_policy_name="simple",
        partner_policy_name="simple",
    )

    assert -1.1 <= result["reward"] <= 1.1
    assert result["reward"] != 0.0
    assert isinstance(result["won"], bool)
    assert result["steps"] <= 13
    assert result["winning_team"] in [0, 1]


def test_evaluate_baseline_runs_multiple_episodes():
    stats = evaluate_baseline(
        num_episodes=20,
        seed=42,
        learning_policy_name="random_legal",
        opponent_policy_name="simple",
        partner_policy_name="simple",
    )

    assert stats["num_episodes"] == 20
    assert stats["wins"] + stats["losses"] == 20
    assert 0 <= stats["win_rate"] <= 1
    assert -1.1 <= stats["avg_reward"] <= 1.1   
    assert stats["avg_learning_steps"] <= 13