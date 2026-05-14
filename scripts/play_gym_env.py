from hokm.gym_env import HokmGymEnv


def main():
    env = HokmGymEnv(
        learning_player=0,
        opponent_policy_name="simple",
        partner_policy_name="simple",
        seed=42,
    )

    obs, info = env.reset()

    print("=" * 60)
    print("HokmGymEnv Demo")
    print("=" * 60)
    print(f"Trump suit: {info['trump_suit']}")
    print(f"Learning player: {info['learning_player']}")
    print()

    terminated = False
    step_number = 1

    while not terminated:
        legal_actions = info["legal_actions"]
        action = legal_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Learning step {step_number}")
        print(f"  Action/card id: {action}")
        print(f"  Reward: {reward}")
        print(f"  Team tricks: {info.get('team_tricks')}")
        print(f"  Terminated: {terminated}")
        print("-" * 60)

        step_number += 1

    print()
    print("=" * 60)
    print("Episode finished")
    print("=" * 60)
    print(f"Final reward: {reward}")
    print(f"Winning team: {info.get('winning_team')}")


if __name__ == "__main__":
    main()