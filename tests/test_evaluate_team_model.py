from scripts.evaluate_team_model import (
    build_action_mask_from_observation,
    evaluate_team_model,
    run_team_model_episode,
)
from hokm.cards import Card
from hokm.observation import build_player_observation


def test_build_action_mask_from_observation():
    observation = build_player_observation(
        player_id=0,
        hand=[Card("hearts", "A"), Card("clubs", "2")],
        completed_tricks=[],
        current_trick=[],
        trump_suit="hearts",
        team_tricks={0: 0, 1: 0},
        hakem=0,
    )

    mask = build_action_mask_from_observation(observation)

    assert len(mask) == 52
    assert sum(mask) == 2


def test_run_team_model_episode_smoke():
    # This test assumes you already have at least one trained model.
    # If not, skip this test manually or point it to an available model later.
    pass