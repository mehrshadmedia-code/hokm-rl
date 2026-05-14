from random import Random
from statistics import mean
from typing import Dict, List

from hokm.gym_env import HokmGymEnv


def choose_baseline_action(policy_name: str, legal_actions: List[int], rng: Random) -> int:
    if not legal_actions:
        raise ValueError("No legal actions available.")

    if policy_name == "first_legal":
        return legal_actions[0]

    if policy_name == "random_legal":
        return rng.choice(legal_actions)

    raise ValueError(f"Unknown baseline policy: {policy_name}")


def run_one_episode(
    seed: int,
    learning_policy_name: str,
    opponent_policy_name: str = "simple",
    partner_policy_name: str = "simple",
) -> Dict:
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
            policy_name=learning_policy_name,
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


def evaluate_baseline(
    num_episodes: int = 1000,
    seed: int = 42,
    learning_policy_name: str = "random_legal",
    opponent_policy_name: str = "simple",
    partner_policy_name: str = "simple",
) -> Dict:
    results = []

    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index

        result = run_one_episode(
            seed=episode_seed,
            learning_policy_name=learning_policy_name,
            opponent_policy_name=opponent_policy_name,
            partner_policy_name=partner_policy_name,
        )

        results.append(result)

    wins = sum(1 for result in results if result["won"])
    losses = num_episodes - wins

    return {
        "num_episodes": num_episodes,
        "learning_policy": learning_policy_name,
        "opponent_policy": opponent_policy_name,
        "partner_policy": partner_policy_name,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / num_episodes if num_episodes else 0,
        "avg_reward": mean(result["reward"] for result in results) if results else 0,
        "avg_learning_steps": mean(result["steps"] for result in results) if results else 0,
    }


def print_stats(stats: Dict) -> None:
    print("=" * 60)
    print(
        f"Learning: {stats['learning_policy']} | "
        f"Partner: {stats['partner_policy']} | "
        f"Opponents: {stats['opponent_policy']}"
    )
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print()


def main():
    experiments = [
        {
            "learning_policy_name": "first_legal",
            "partner_policy_name": "simple",
            "opponent_policy_name": "simple",
        },
        {
            "learning_policy_name": "random_legal",
            "partner_policy_name": "simple",
            "opponent_policy_name": "simple",
        },
        {
            "learning_policy_name": "random_legal",
            "partner_policy_name": "simple",
            "opponent_policy_name": "random",
        },
    ]

    for experiment in experiments:
        stats = evaluate_baseline(
            num_episodes=1000,
            seed=42,
            **experiment,
        )

        print_stats(stats)


if __name__ == "__main__":
    main()