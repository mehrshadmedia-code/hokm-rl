from hokm.agents import RandomAgent
from hokm.game import HokmHand


def main():
    agents = [
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        RandomAgent(seed=3),
        RandomAgent(seed=4),
    ]

    hand = HokmHand(
        agents=agents,
        hakem=0,
        seed=42,
    )

    result = hand.play()

    print("=" * 60)
    print("Random Hokm Hand Result")
    print("=" * 60)
    print(f"Hakem: Player {result.hakem}")
    print(f"Trump suit: {result.trump_suit}")
    print(f"Winning team: Team {result.winning_team}")
    print(f"Final tricks: {result.team_tricks}")
    print()

    for trick in result.tricks:
        print(f"Trick {trick.trick_number}")
        for player_id, card in trick.cards_played:
            print(f"  Player {player_id}: {card}")
        print(f"  Winner: Player {trick.winner}")
        print(f"  Score: {trick.team_tricks}")
        print("-" * 60)


if __name__ == "__main__":
    main()