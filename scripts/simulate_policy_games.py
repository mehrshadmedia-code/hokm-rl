from statistics import mean
from typing import Dict, List

from hokm.env import HokmEnv
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy, AdvancedRulePolicy

def play_hand_with_policies(
    seed: int,
    policies: Dict[int, object],
) -> dict:
    env = HokmEnv(seed=seed, hakem=0)
    env.reset()

    steps = 0

    while not env.done:
        player_id = env.current_player
        policy = policies[player_id]

        action = policy.choose_action(
            player_id=player_id,
            hand=env.hands[player_id],
            current_trick=env.current_trick,
            trump_suit=env.trump_suit,
        )

        env.step(action)
        steps += 1

    return {
        "winning_team": env.winning_team,
        "team_tricks": env.team_tricks.copy(),
        "steps": steps,
        "completed_tricks": len(env.completed_tricks),
        "trump_suit": env.trump_suit,
    }


def summarize_results(results: List[dict]) -> dict:
    team_0_wins = sum(1 for result in results if result["winning_team"] == 0)
    team_1_wins = sum(1 for result in results if result["winning_team"] == 1)
    total = len(results)

    return {
        "hands": total,
        "team_0_wins": team_0_wins,
        "team_1_wins": team_1_wins,
        "team_0_win_rate": team_0_wins / total if total else 0,
        "team_1_win_rate": team_1_wins / total if total else 0,
        "avg_steps": mean(result["steps"] for result in results) if results else 0,
        "avg_completed_tricks": mean(result["completed_tricks"] for result in results)
        if results
        else 0,
    }


def simulate_matchup(
    num_hands: int = 1000,
    seed: int = 42,
    team_0_policy_name: str = "simple",
    team_1_policy_name: str = "random",
) -> dict:
    results = []

    for index in range(num_hands):
        hand_seed = seed + index

        policies = {
            0: make_policy(team_0_policy_name, seed=hand_seed + 0),
            2: make_policy(team_0_policy_name, seed=hand_seed + 2),
            1: make_policy(team_1_policy_name, seed=hand_seed + 1),
            3: make_policy(team_1_policy_name, seed=hand_seed + 3),
        }

        results.append(
            play_hand_with_policies(
                seed=hand_seed,
                policies=policies,
            )
        )

    return summarize_results(results)


def make_policy(policy_name: str, seed: int):
    if policy_name == "simple":
        return SimpleRulePolicy()

    if policy_name == "advanced":
        return AdvancedRulePolicy()

    if policy_name == "random":
        return RandomLegalPolicy(seed=seed)

    raise ValueError(f"Unknown policy name: {policy_name}")

def main():
    matchups = [
        ("random", "random"),
        ("simple", "random"),
        ("advanced", "random"),
        ("advanced", "simple"),
        ("advanced", "advanced"),
    ]

    for team_0_policy, team_1_policy in matchups:
        stats = simulate_matchup(
            num_hands=1000,
            seed=42,
            team_0_policy_name=team_0_policy,
            team_1_policy_name=team_1_policy,
        )

        print("=" * 60)
        print(f"Team 0: {team_0_policy} | Team 1: {team_1_policy}")
        print("=" * 60)

        for key, value in stats.items():
            print(f"{key}: {value}")

        print()


if __name__ == "__main__":
    main()