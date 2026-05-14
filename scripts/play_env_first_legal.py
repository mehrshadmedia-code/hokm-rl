from hokm.env import HokmEnv


def main():
    env = HokmEnv(seed=42, hakem=0)
    observation = env.reset()

    print("=" * 60)
    print("HokmEnv Demo: First Legal Action Policy")
    print("=" * 60)
    print(f"Trump suit: {env.trump_suit}")
    print(f"Starting player: {observation.player_id}")
    print()

    step_number = 1

    while not env.done:
        player = env.current_player
        legal_actions = env.legal_actions()
        action = legal_actions[0]

        result = env.step(action)

        print(f"Step {step_number}")
        print(f"  Player {player} played: {result.info['played_card']}")

        if result.info["trick_completed"]:
            print(f"  Trick winner: Player {result.info['trick_winner']}")
            print(f"  Team tricks: {result.info['team_tricks']}")
            print("-" * 60)

        step_number += 1

    print()
    print("=" * 60)
    print("Final Result")
    print("=" * 60)
    print(f"Winning team: {env.winning_team}")
    print(f"Final tricks: {env.team_tricks}")


if __name__ == "__main__":
    main()