import pytest

from hokm.cards import card_to_id
from hokm.env import HokmEnv


def test_env_reset_returns_observation():
    env = HokmEnv(seed=42, hakem=0)
    observation = env.reset()

    assert observation.player_id == 0
    assert len(observation.hand_vector) == 52
    assert len(observation.legal_action_mask) == 52
    assert sum(observation.hand_vector) == 13
    assert sum(observation.legal_action_mask) >= 1


def test_legal_actions_match_mask():
    env = HokmEnv(seed=42, hakem=0)
    observation = env.reset()

    legal_actions = env.legal_actions()

    assert len(legal_actions) == sum(observation.legal_action_mask)

    for action in legal_actions:
        assert observation.legal_action_mask[action] == 1


def test_step_plays_one_legal_card():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    starting_player = env.current_player
    starting_hand_size = len(env.hands[starting_player])

    action = env.legal_actions()[0]
    result = env.step(action)

    assert result.done is False
    assert result.reward == 0
    assert len(env.hands[starting_player]) == starting_hand_size - 1
    assert result.info["player"] == starting_player
    assert result.info["played_card"] is not None


def test_illegal_action_raises_error():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    legal_actions = env.legal_actions()
    illegal_actions = [action for action in range(52) if action not in legal_actions]

    if not illegal_actions:
        pytest.skip("No illegal action available in this state.")

    with pytest.raises(ValueError):
        env.step(illegal_actions[0])


def test_four_steps_complete_one_trick():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    result = None

    for _ in range(4):
        action = env.legal_actions()[0]
        result = env.step(action)

    assert result is not None
    assert result.info["trick_completed"] is True
    assert result.info["trick_winner"] in [0, 1, 2, 3]
    assert len(env.completed_tricks) == 1
    assert len(env.current_trick) == 0
    assert sum(env.team_tricks.values()) == 1


def test_env_can_play_full_hand_with_first_legal_action_policy():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    steps = 0

    while not env.done:
        action = env.legal_actions()[0]
        result = env.step(action)
        steps += 1

    assert result.done is True
    assert env.winning_team in [0, 1]
    assert 7 in env.team_tricks.values()
    assert steps <= 52


def test_cannot_step_after_done():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    while not env.done:
        action = env.legal_actions()[0]
        env.step(action)

    with pytest.raises(RuntimeError):
        env.step(0)


def test_cannot_observe_after_done():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    while not env.done:
        action = env.legal_actions()[0]
        env.step(action)

    with pytest.raises(RuntimeError):
        env.get_observation()


def test_terminal_reward_matches_winning_team():
    env = HokmEnv(seed=42, hakem=0)
    env.reset()

    final_result = None

    while not env.done:
        action = env.legal_actions()[0]
        final_result = env.step(action)

    last_player = final_result.info["player"]
    last_player_team = 0 if last_player in [0, 2] else 1

    if last_player_team == env.winning_team:
        assert final_result.reward == 1
    else:
        assert final_result.reward == -1