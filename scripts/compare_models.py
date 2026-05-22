import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from sb3_contrib import MaskablePPO

from hokm.gym_env import HokmGymEnv
from scripts.evaluate_gym_baselines import choose_baseline_action
from scripts.evaluate_trained_model import run_episode as run_model_episode


def run_baseline_episode(
    seed: int,
    baseline_policy_name: str,
    opponent_policy_name: str,
    partner_policy_name: str,
) -> Dict:
    from random import Random

    rng = Random(seed)

    env = HokmGymEnv(
        learning_player=0,
        opponent_policy_name=opponent_policy_name,
        partner_policy_name=partner_policy_name,
        seed=seed,
    )

    obs, info = env.reset()

    terminated = False
    steps = 0
    final_reward = 0.0

    while not terminated:
        action = choose_baseline_action(
            policy_name=baseline_policy_name,
            legal_actions=info["legal_actions"],
            rng=rng,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            raise RuntimeError("Unexpected truncation.")

        final_reward = reward
        steps += 1

    return {
        "reward": final_reward,
        "won": final_reward > 0,
        "steps": steps,
        "winning_team": info["winning_team"],
    }


def summarize_results(name: str, results: List[Dict]) -> Dict:
    wins = sum(1 for result in results if result["won"])
    total = len(results)

    return {
        "name": name,
        "episodes": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total if total else 0,
        "avg_reward": mean(result["reward"] for result in results) if results else 0,
        "avg_learning_steps": mean(result["steps"] for result in results) if results else 0,
    }


def evaluate_baseline_entry(
    name: str,
    num_episodes: int,
    seed: int,
    opponent_policy_name: str,
    partner_policy_name: str,
) -> Dict:
    results = []

    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index

        results.append(
            run_baseline_episode(
                seed=episode_seed,
                baseline_policy_name=name,
                opponent_policy_name=opponent_policy_name,
                partner_policy_name=partner_policy_name,
            )
        )

    return summarize_results(name=name, results=results)

def evaluate_model_entry(
    name: str,
    model_path: str,
    num_episodes: int,
    seed: int,
    opponent_policy_name: str,
    partner_policy_name: str,
    deterministic: bool = True,
) -> Optional[Dict]:
    path = Path(model_path)

    if not path.exists():
        print(f"Skipping {name}: model not found at {model_path}")
        return None

    try:
        model = MaskablePPO.load(str(path))

        results = []

        for episode_index in range(num_episodes):
            episode_seed = seed + episode_index

            results.append(
                run_model_episode(
                    model=model,
                    seed=episode_seed,
                    deterministic=deterministic,
                    opponent_policy_name=opponent_policy_name,
                    partner_policy_name=partner_policy_name,
                )
            )

        return summarize_results(name=name, results=results)

    except ValueError as exc:
        print(f"Skipping {name}: incompatible model/environment shape.")
        print(f"  Reason: {exc}")
        return None
    
def print_comparison_table(rows: List[Dict]) -> None:
    print("=" * 90)
    print("Hokm Policy Comparison")
    print("=" * 90)
    print(
        f"{'Name':<32} "
        f"{'Episodes':>8} "
        f"{'Wins':>8} "
        f"{'Losses':>8} "
        f"{'Win Rate':>10} "
        f"{'Avg Reward':>12} "
        f"{'Avg Steps':>10}"
    )
    print("-" * 90)

    for row in rows:
        print(
            f"{row['name']:<32} "
            f"{row['episodes']:>8} "
            f"{row['wins']:>8} "
            f"{row['losses']:>8} "
            f"{row['win_rate']:>10.3f} "
            f"{row['avg_reward']:>12.3f} "
            f"{row['avg_learning_steps']:>10.3f}"
        )

    print("=" * 90)


def compare_policies(
    num_episodes: int,
    seed: int,
    opponent_policy_name: str,
    partner_policy_name: str,
    deterministic: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []

    baseline_names = [
        "first_legal",
        "random_legal",
    ]

    for baseline_name in baseline_names:
        rows.append(
            evaluate_baseline_entry(
                name=baseline_name,
                num_episodes=num_episodes,
                seed=seed,
                opponent_policy_name=opponent_policy_name,
                partner_policy_name=partner_policy_name,
            )
        )

        model_entries = [
        (
            "stage_a_features_random",
            "models/stage_a_features_random.zip",
        ),
        (
            "stage_b_features_simple",
            "models/stage_b_features_simple.zip",
        ),
        (
            "latest_default_model",
            "models/hokm_maskable_ppo.zip",
        ),
    ]

    for name, model_path in model_entries:
        result = evaluate_model_entry(
            name=name,
            model_path=model_path,
            num_episodes=num_episodes,
            seed=seed,
            opponent_policy_name=opponent_policy_name,
            partner_policy_name=partner_policy_name,
            deterministic=deterministic,
        )

        if result is not None:
            rows.append(result)

    rows.sort(key=lambda row: row["win_rate"], reverse=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of episodes per policy.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="Evaluation seed.",
    )

    parser.add_argument(
        "--opponent-policy-name",
        type=str,
        default="simple",
        choices=["simple", "advanced", "random"],
        help="Policy used by opponent players.",
    )

    parser.add_argument(
        "--partner-policy-name",
        type=str,
        default="simple",
        choices=["simple", "advanced", "random"],
        help="Policy used by the learning player's partner.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic model actions instead of deterministic.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rows = compare_policies(
        num_episodes=args.num_episodes,
        seed=args.seed,
        opponent_policy_name=args.opponent_policy_name,
        partner_policy_name=args.partner_policy_name,
        deterministic=not args.stochastic,
    )

    print()
    print(f"Opponent policy: {args.opponent_policy_name}")
    print(f"Partner policy: {args.partner_policy_name}")
    print(f"Model deterministic: {not args.stochastic}")
    print()

    print_comparison_table(rows)


if __name__ == "__main__":
    main()