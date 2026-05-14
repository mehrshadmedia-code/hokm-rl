from random import Random
from statistics import mean

from hokm.env import HokmEnv


def play_random_hand(seed: int) -> dict:
    rng = Random(seed)

    env = HokmEnv(seed=seed, hakem=0)
    env.reset()

    steps = 0

    while not env.done:
        legal_actions = env.legal_actions()
        action = rng.choice(legal_actions)
        env.step(action)
        steps += 1

    return {
        "winning_team": env.winning_team,
        "team_tricks": env.team_tricks.copy(),
        "steps": steps,
        "completed_tricks": len(env.completed_tricks),
        "trump_suit": env.trump_suit,
    }


def simulate_random_hands(num_hands: int = 1000, seed: int = 42) -> dict:
    results = []
    crashes = 0

    for game_index in range(num_hands):
        hand_seed = seed + game_index

        try:
            result = play_random_hand(seed=hand_seed)
            results.append(result)
        except Exception as exc:
            crashes += 1
            print(f"Crash in hand {game_index} with seed {hand_seed}: {exc}")

    team_0_wins = sum(1 for result in results if result["winning_team"] == 0)
    team_1_wins = sum(1 for result in results if result["winning_team"] == 1)

    total_completed = len(results)

    if total_completed == 0:
        return {
            "num_hands_requested": num_hands,
            "num_hands_completed": 0,
            "crashes": crashes,
        }

    return {
        "num_hands_requested": num_hands,
        "num_hands_completed": total_completed,
        "crashes": crashes,
        "team_0_wins": team_0_wins,
        "team_1_wins": team_1_wins,
        "team_0_win_rate": team_0_wins / total_completed,
        "team_1_win_rate": team_1_wins / total_completed,
        "avg_steps": mean(result["steps"] for result in results),
        "avg_completed_tricks": mean(result["completed_tricks"] for result in results),
        "trump_counts": {
            "hearts": sum(1 for result in results if result["trump_suit"] == "hearts"),
            "diamonds": sum(1 for result in results if result["trump_suit"] == "diamonds"),
            "clubs": sum(1 for result in results if result["trump_suit"] == "clubs"),
            "spades": sum(1 for result in results if result["trump_suit"] == "spades"),
        },
    }


def main():
    stats = simulate_random_hands(num_hands=1000, seed=42)

    print("=" * 60)
    print("Random HokmEnv Simulation")
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()