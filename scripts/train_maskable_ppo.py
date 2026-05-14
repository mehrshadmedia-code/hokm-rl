import argparse
from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from hokm.gym_env import HokmGymEnv


def make_env(
    seed: int,
    opponent_policy_name: str,
    partner_policy_name: str,
) -> Monitor:
    env = HokmGymEnv(
        learning_player=0,
        opponent_policy_name=opponent_policy_name,
        partner_policy_name=partner_policy_name,
        seed=seed,
    )

    return Monitor(env)


def train(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    opponent_policy_name: str,
    partner_policy_name: str,
    model_path: Optional[str] = None,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.02,
    save_name: str = "hokm_maskable_ppo",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        seed=seed,
        opponent_policy_name=opponent_policy_name,
        partner_policy_name=partner_policy_name,
    )

    if model_path:
        print("=" * 60)
        print("Continuing training from existing model")
        print("=" * 60)
        print(f"Loading model: {model_path}")

        model = MaskablePPO.load(
            model_path,
            env=env,
            seed=seed,
        )
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=seed,
            learning_rate=learning_rate,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            ent_coef=ent_coef,
        )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=model_path is None,
    )

    save_path = output_path / save_name
    model.save(save_path)

    print("=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"Opponent policy: {opponent_policy_name}")
    print(f"Partner policy: {partner_policy_name}")
    print(f"Saved model to: {save_path}.zip")

    return save_path


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

    parser.add_argument(
        "--opponent-policy-name",
        type=str,
        default="simple",
        choices=["simple", "random"],
        help="Policy used by opponent players.",
    )

    parser.add_argument(
        "--partner-policy-name",
        type=str,
        default="simple",
        choices=["simple", "random"],
        help="Policy used by the learning player's partner.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional existing model path to continue training.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )

    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="Entropy coefficient for exploration.",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default="hokm_maskable_ppo",
        help="Model filename without .zip extension.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
        opponent_policy_name=args.opponent_policy_name,
        partner_policy_name=args.partner_policy_name,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        save_name=args.save_name,
    )


if __name__ == "__main__":
    main()