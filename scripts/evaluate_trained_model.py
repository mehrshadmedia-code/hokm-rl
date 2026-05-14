import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List

from sb3_contrib import MaskablePPO

from hokm.gym_env import HokmGymEnv


def run_episode(model: MaskablePPO, seed: int, deterministic: bool = True) -> Dict:
    env = HokmGymEnv(
        learning_player=0,
        opponent_policy_name="simple",
        partner_policy_name="simple",
        seed=seed,
    )

    obs, info = env.reset()

    terminated = False
    final_reward = 0.0
    steps = 0

    while not terminated:
        action_masks = env.action_masks()

        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )

        obs, reward, terminated, truncated, info = env.step(int(action))

        if truncated:
            raise RuntimeError("Unexpected truncation.")

        final_reward = reward
        steps += 1

    return {
        "reward": final_reward,
        "won": final_reward == 1.0,
        "steps": steps,
        "winning_team": info["winning_team"],
    }


def evaluate_model(
    model_path: str,
    num_episodes: int,
    seed: int,
    deterministic: bool = True,
) -> Dict:
    model = MaskablePPO.load(model_path)

    results: List[Dict] = []

    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index

        results.append(
            run_episode(
                model=model,
                seed=episode_seed,
                deterministic=deterministic,
            )
        )

    wins = sum(1 for result in results if result["won"])
    losses = num_episodes - wins

    return {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / num_episodes if num_episodes else 0,
        "avg_reward": mean(result["reward"] for result in results) if results else 0,
        "avg_learning_steps": mean(result["steps"] for result in results) if results else 0,
    }


def print_stats(stats: Dict) -> None:
    print("=" * 60)
    print("Trained Model Evaluation")
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hokm_maskable_ppo.zip",
        help="Path to trained MaskablePPO model.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="Evaluation seed.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Train first with: python -m scripts.train_maskable_ppo"
        )

    stats = evaluate_model(
        model_path=str(model_path),
        num_episodes=args.num_episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()