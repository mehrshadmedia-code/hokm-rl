import numpy as np

from hokm.gym_env import HokmGymEnv


def test_gym_env_reset_returns_observation_and_info():
    env = HokmGymEnv(seed=42)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert "legal_actions" in info
    assert "action_mask" in info
    assert len(info["action_mask"]) == 52
    assert len(info["legal_actions"]) >= 1


def test_gym_env_step_with_legal_action():
    env = HokmGymEnv(seed=42)
    obs, info = env.reset()

    action = info["legal_actions"][0]

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert reward in [-1.0, 0.0, 1.0]
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "illegal_action" in step_info


def test_gym_env_illegal_action_terminates_with_penalty():
    env = HokmGymEnv(seed=42)
    obs, info = env.reset()

    legal_actions = info["legal_actions"]
    illegal_actions = [action for action in range(52) if action not in legal_actions]

    if not illegal_actions:
        return

    next_obs, reward, terminated, truncated, step_info = env.step(illegal_actions[0])

    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert step_info["illegal_action"] is True


def test_gym_env_can_finish_episode_with_first_legal_policy():
    env = HokmGymEnv(seed=42)
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
    assert reward in [-1.0, 1.0]
    assert steps <= 13


def test_action_mask_matches_legal_actions():
    env = HokmGymEnv(seed=42)
    obs, info = env.reset()

    mask = info["action_mask"]
    legal_actions = info["legal_actions"]

    assert mask.sum() == len(legal_actions)

    for action in legal_actions:
        assert mask[action] == 1


def test_gym_env_with_random_opponents():
    env = HokmGymEnv(
        seed=42,
        learning_player=0,
        opponent_policy_name="random",
        partner_policy_name="simple",
    )

    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert len(info["legal_actions"]) >= 1


def test_multiple_resets_work():
    env = HokmGymEnv(seed=42)

    obs1, info1 = env.reset()
    obs2, info2 = env.reset()

    assert obs1.shape == env.observation_space.shape
    assert obs2.shape == env.observation_space.shape
    assert len(info1["legal_actions"]) >= 1
    assert len(info2["legal_actions"]) >= 1