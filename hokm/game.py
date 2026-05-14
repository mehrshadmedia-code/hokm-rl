from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from hokm.agents import RandomAgent
from hokm.cards import Card, create_deck, deal_hokm_style, shuffle_deck
from hokm.rules import PlayedCard, determine_trick_winner, is_legal_play


TEAM_BY_PLAYER = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
}


@dataclass
class TrickResult:
    trick_number: int
    cards_played: List[PlayedCard]
    winner: int
    winning_team: int
    team_tricks: Dict[int, int]


@dataclass
class HandResult:
    trump_suit: str
    hakem: int
    winning_team: int
    team_tricks: Dict[int, int]
    tricks: List[TrickResult]


@dataclass
class HokmHand:
    agents: Sequence[RandomAgent]
    hakem: int = 0
    seed: Optional[int] = None
    hands: List[List[Card]] = field(default_factory=list)
    trump_suit: Optional[str] = None
    team_tricks: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    tricks: List[TrickResult] = field(default_factory=list)

    def __post_init__(self):
        if len(self.agents) != 4:
            raise ValueError("Hokm requires exactly 4 agents.")

        if self.hakem not in range(4):
            raise ValueError("Hakem must be a player id from 0 to 3.")

    def setup(self) -> None:
        deck = shuffle_deck(create_deck(), seed=self.seed)

        # First give Hakem only 5 cards to choose trump.
        first_five_cards = deck[:5]
        self.trump_suit = self.agents[self.hakem].choose_trump(first_five_cards)

        # Then deal the full hand Hokm-style.
        self.hands = deal_hokm_style(deck, hakem=self.hakem)

    def play(self) -> HandResult:
        if not self.hands:
            self.setup()

        if self.trump_suit is None:
            raise RuntimeError("Trump suit was not selected.")

        current_leader = self.hakem
        trick_number = 1

        while max(self.team_tricks.values()) < 7:
            trick = self.play_trick(
                trick_number=trick_number,
                leader=current_leader,
            )

            self.tricks.append(trick)
            current_leader = trick.winner
            trick_number += 1

        winning_team = 0 if self.team_tricks[0] >= 7 else 1

        return HandResult(
            trump_suit=self.trump_suit,
            hakem=self.hakem,
            winning_team=winning_team,
            team_tricks=self.team_tricks.copy(),
            tricks=self.tricks.copy(),
        )

    def play_trick(self, trick_number: int, leader: int) -> TrickResult:
        if self.trump_suit is None:
            raise RuntimeError("Trump suit was not selected.")

        current_trick: List[PlayedCard] = []
        play_order = self.get_play_order(leader)

        for player_id in play_order:
            agent = self.agents[player_id]
            hand = self.hands[player_id]

            chosen_card = agent.choose_card(
                hand=hand,
                current_trick=current_trick,
                trump_suit=self.trump_suit,
            )

            if not is_legal_play(chosen_card, hand, current_trick):
                raise RuntimeError(
                    f"Player {player_id} attempted illegal card: {chosen_card}"
                )

            hand.remove(chosen_card)
            current_trick.append((player_id, chosen_card))

        winner = determine_trick_winner(
            current_trick=current_trick,
            trump_suit=self.trump_suit,
        )

        winning_team = TEAM_BY_PLAYER[winner]
        self.team_tricks[winning_team] += 1

        return TrickResult(
            trick_number=trick_number,
            cards_played=current_trick,
            winner=winner,
            winning_team=winning_team,
            team_tricks=self.team_tricks.copy(),
        )

    @staticmethod
    def get_play_order(leader: int) -> List[int]:
        if leader not in range(4):
            raise ValueError("Leader must be a player id from 0 to 3.")

        return [(leader + offset) % 4 for offset in range(4)]