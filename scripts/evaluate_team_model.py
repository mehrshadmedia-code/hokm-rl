import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List

from sb3_contrib import MaskablePPO

from hokm.env import HokmEnv
from hokm.observation import observation_to_flat_vector
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy


TEAM_MODEL_PLAYERS = {0, 2}


def make_policy(policy_name: str, seed: int):
    if policy_name == "simple":
        return SimpleRulePolicy()

    if policy_name == "random":
        return RandomLegalPolicy(seed=seed)

    raise ValueError(f"Unknown policy name: {policy_name}")


def build_action_mask_from_observation(observation) -> List[bool]:
    return [value == 1 for value in observation.legal_action_mask]


def run_team_model_episode(
    model: MaskablePPO,
    seed: int,
    opponent_policy_name: str = "simple",
    deterministic: bool = True,
) -> Dict:
    env = HokmEnv(seed=seed, hakem=0)
    env.reset()

    opponent_policies = {
        1: make_policy(opponent_policy_name, seed + 1),
        3: make_policy(opponent_policy_name, seed + 3),
    }

    team_steps = 0

    while not env.done:
        player_id = env.current_player

        if player_id in TEAM_MODEL_PLAYERS:
            observation = env.get_observation()
            obs = observation_to_flat_vector(observation)
            action_mask = build_action_mask_from_observation(observation)

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_mask,
            )

            env.step(int(action))
            team_steps += 1

        else:
            policy = opponent_policies[player_id]

            action = policy.choose_action(
                player_id=player_id,
                hand=env.hands[player_id],
                current_trick=env.current_trick,
                trump_suit=env.trump_suit,
            )

            env.step(action)

    team_won = env.winning_team == 0  # players 0 and 2 are Team 0

    return {
        "won": team_won,
        "reward": 1.0 if team_won else -1.0,
        "team_steps": team_steps,
        "winning_team": env.winning_team,
        "team_tricks": env.team_tricks.copy(),
    }


def evaluate_team_model(
    model_path: str,
    num_episodes: int,
    seed: int,
    opponent_policy_name: str = "simple",
    deterministic: bool = True,
) -> Dict:
    model = MaskablePPO.load(model_path)

    results: List[Dict] = []

    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index
        results.append(
            run_team_model_episode(
                model=model,
                seed=episode_seed,
                opponent_policy_name=opponent_policy_name,
                deterministic=deterministic,
            )
        )

    wins = sum(1 for result in results if result["won"])
    losses = num_episodes - wins

    return {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "opponent_policy": opponent_policy_name,
        "deterministic": deterministic,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / num_episodes if num_episodes else 0,
        "avg_reward": mean(result["reward"] for result in results) if results else 0,
        "avg_team_steps": mean(result["team_steps"] for result in results) if results else 0,
    }


def print_stats(stats: Dict) -> None:
    print("=" * 60)
    print("Team Model Evaluation")
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/stage_b_features_simple.zip",
        help="Path to trained model.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="Evaluation seed.",
    )

    parser.add_argument(
        "--opponent-policy-name",
        type=str,
        default="simple",
        choices=["simple", "advanced", "random"],
        help="Opponent policy used by players 1 and 3.",
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
        raise FileNotFoundError(f"Model not found: {model_path}")

    stats = evaluate_team_model(
        model_path=str(model_path),
        num_episodes=args.num_episodes,
        seed=args.seed,
        opponent_policy_name=args.opponent_policy_name,
        deterministic=not args.stochastic,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()