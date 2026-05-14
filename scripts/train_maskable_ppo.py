import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from hokm.gym_env import HokmGymEnv


def make_env(seed: int) -> Monitor:
    env = HokmGymEnv(
        learning_player=0,
        opponent_policy_name="simple",
        partner_policy_name="simple",
        seed=seed,
    )

    return Monitor(env)


def train(
    total_timesteps: int,
    seed: int,
    output_dir: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=seed)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = output_path / "hokm_maskable_ppo"

    model.save(model_path)

    print("=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"Saved model to: {model_path}.zip")

    return model_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000,
        help="Number of PPO training timesteps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()