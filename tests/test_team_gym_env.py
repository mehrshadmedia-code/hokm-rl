import numpy as np

from hokm.team_gym_env import HokmTeamGymEnv


def test_team_gym_env_reset_returns_team_turn():
    env = HokmTeamGymEnv(seed=42, opponent_policy_name="simple")
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert env.env.current_player in [0, 2]
    assert len(info["legal_actions"]) >= 1
    assert len(info["action_mask"]) == 52


def test_team_gym_env_step_with_legal_action():
    env = HokmTeamGymEnv(seed=42, opponent_policy_name="simple")
    obs, info = env.reset()

    action = info["legal_actions"][0]

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert -1.1 <= reward <= 1.1
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "illegal_action" in step_info


def test_team_gym_env_can_finish_episode():
    env = HokmTeamGymEnv(seed=42, opponent_policy_name="simple")
    obs, info = env.reset()

    terminated = False
    steps = 0

    while not terminated:
        legal_actions = info["legal_actions"]
        action = legal_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert terminated is True
    assert truncated is False
    assert reward != 0
    assert env.env.winning_team in [0, 1]
    assert steps <= 26


def test_team_gym_action_mask_matches_legal_actions():
    env = HokmTeamGymEnv(seed=42, opponent_policy_name="simple")
    obs, info = env.reset()

    mask = info["action_mask"]
    legal_actions = info["legal_actions"]

    assert mask.sum() == len(legal_actions)

    for action in legal_actions:
        assert mask[action] == 1


def test_team_gym_env_with_random_opponents():
    env = HokmTeamGymEnv(seed=42, opponent_policy_name="random")
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert env.env.current_player in [0, 2]
    assert len(info["legal_actions"]) >= 1