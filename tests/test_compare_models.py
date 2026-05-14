from scripts.compare_models import (
    compare_policies,
    evaluate_baseline_entry,
    run_baseline_episode,
    summarize_results,
)


def test_run_baseline_episode_finishes():
    result = run_baseline_episode(
        seed=42,
        baseline_policy_name="first_legal",
        opponent_policy_name="simple",
        partner_policy_name="simple",
    )

    assert result["reward"] != 0
    assert isinstance(result["won"], bool)
    assert result["steps"] <= 13
    assert result["winning_team"] in [0, 1]


def test_summarize_results():
    results = [
        {"reward": 1.05, "won": True, "steps": 10},
        {"reward": -1.05, "won": False, "steps": 12},
    ]

    summary = summarize_results("test_policy", results)

    assert summary["name"] == "test_policy"
    assert summary["episodes"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["avg_learning_steps"] == 11


def test_evaluate_baseline_entry_runs():
    summary = evaluate_baseline_entry(
        name="random_legal",
        num_episodes=5,
        seed=42,
        opponent_policy_name="simple",
        partner_policy_name="simple",
    )

    assert summary["episodes"] == 5
    assert summary["wins"] + summary["losses"] == 5
    assert 0 <= summary["win_rate"] <= 1


def test_compare_policies_runs_with_baselines_and_missing_models():
    rows = compare_policies(
        num_episodes=3,
        seed=42,
        opponent_policy_name="simple",
        partner_policy_name="simple",
    )

    assert len(rows) >= 2

    names = {row["name"] for row in rows}

    assert "first_legal" in names
    assert "random_legal" in names