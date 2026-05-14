from scripts.train_maskable_ppo import make_env
from hokm.gym_env import HokmGymEnv


def test_training_env_has_action_masks():
    env = make_env(seed=42)

    obs, info = env.reset()

    unwrapped = env.unwrapped

    assert isinstance(unwrapped, HokmGymEnv)
    assert hasattr(unwrapped, "action_masks")
    assert unwrapped.action_masks().shape == (52,)
    assert unwrapped.action_masks().sum() >= 1