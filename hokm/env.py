from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from hokm.cards import Card, create_deck, deal_hokm_style, id_to_card, shuffle_deck
from hokm.game import TEAM_BY_PLAYER
from hokm.observation import PlayerObservation, build_player_observation
from hokm.rules import PlayedCard, determine_trick_winner, is_legal_play


@dataclass
class EnvStepResult:
    observation: Optional[PlayerObservation]
    reward: int
    done: bool
    info: Dict


@dataclass
class HokmEnv:
    """
    Turn-based Hokm environment.

    This is our first RL-style environment.

    Action space:
        52 discrete actions.
        Each action is a card id from 0 to 51.

    Observation:
        PlayerObservation for the current player.

    Important:
        This is not yet a Gymnasium/PettingZoo environment.
        It is our clean internal environment first.
    """

    seed: Optional[int] = None
    hakem: int = 0

    hands: List[List[Card]] = field(default_factory=list)
    trump_suit: Optional[str] = None
    current_player: Optional[int] = None
    current_trick: List[PlayedCard] = field(default_factory=list)
    completed_tricks: List[List[PlayedCard]] = field(default_factory=list)
    team_tricks: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    done: bool = False
    winning_team: Optional[int] = None

    def reset(self) -> PlayerObservation:
        """
        Start a new Hokm hand and return the first player's observation.
        """
        if self.hakem not in range(4):
            raise ValueError("Hakem must be a player id from 0 to 3.")

        deck = shuffle_deck(create_deck(), seed=self.seed)

        # Hakem chooses trump after seeing the first 5 cards.
        first_five_cards = deck[:5]
        self.trump_suit = self.choose_trump_for_now(first_five_cards)

        self.hands = deal_hokm_style(deck, hakem=self.hakem)
        self.current_player = self.hakem
        self.current_trick = []
        self.completed_tricks = []
        self.team_tricks = {0: 0, 1: 0}
        self.done = False
        self.winning_team = None

        return self.get_observation()

    def choose_trump_for_now(self, first_five_cards: Sequence[Card]) -> str:
        """
        Temporary trump policy.

        Later, trump selection can also become an RL action.
        For now, we choose the suit that appears most often in Hakem's first 5 cards.
        Tie-breaker follows the order in hokm.cards.SUITS.
        """
        if not first_five_cards:
            raise ValueError("Cannot choose trump from empty cards.")

        suit_counts: Dict[str, int] = {}

        for card in first_five_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

        return max(suit_counts, key=suit_counts.get)

    def get_observation(self) -> PlayerObservation:
        """
        Return observation for the current player.
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() to start a new hand.")

        if self.current_player is None:
            raise RuntimeError("Environment has not been reset.")

        if self.trump_suit is None:
            raise RuntimeError("Trump suit has not been selected.")

        return build_player_observation(
            player_id=self.current_player,
            hand=self.hands[self.current_player],
            completed_tricks=self.completed_tricks,
            current_trick=self.current_trick,
            trump_suit=self.trump_suit,
            team_tricks=self.team_tricks,
            hakem=self.hakem,
        )

    def step(self, action: int) -> EnvStepResult:
        """
        Play one card for the current player.

        Args:
            action:
                Card id from 0 to 51.

        Returns:
            EnvStepResult containing next observation, reward, done, and info.
        """
        if self.done:
            raise RuntimeError("Cannot step after hand is done. Call reset().")

        if self.current_player is None:
            raise RuntimeError("Environment has not been reset.")

        if self.trump_suit is None:
            raise RuntimeError("Trump suit has not been selected.")

        chosen_card = id_to_card(action)
        hand = self.hands[self.current_player]

        if not is_legal_play(chosen_card, hand, self.current_trick):
            raise ValueError(
                f"Illegal action. Player {self.current_player} cannot play {chosen_card}."
            )

        hand.remove(chosen_card)
        self.current_trick.append((self.current_player, chosen_card))

        info = {
            "played_card": chosen_card,
            "player": self.current_player,
            "trick_completed": False,
            "trick_winner": None,
            "winning_team": None,
            "team_tricks": self.team_tricks.copy(),
        }

        # If the trick is complete, determine winner.
        if len(self.current_trick) == 4:
            trick_winner = determine_trick_winner(
                current_trick=self.current_trick,
                trump_suit=self.trump_suit,
            )

            winning_team = TEAM_BY_PLAYER[trick_winner]
            self.team_tricks[winning_team] += 1

            self.completed_tricks.append(self.current_trick.copy())
            self.current_trick = []
            self.current_player = trick_winner

            info["trick_completed"] = True
            info["trick_winner"] = trick_winner
            info["winning_team"] = winning_team
            info["team_tricks"] = self.team_tricks.copy()

            if self.team_tricks[winning_team] == 7:
                self.done = True
                self.winning_team = winning_team

                reward = self.get_terminal_reward_for_player(info["player"])

                return EnvStepResult(
                    observation=None,
                    reward=reward,
                    done=True,
                    info=info,
                )

        else:
            self.current_player = (self.current_player + 1) % 4

        reward = 0

        return EnvStepResult(
            observation=self.get_observation(),
            reward=reward,
            done=False,
            info=info,
        )

    def get_terminal_reward_for_player(self, player_id: int) -> int:
        """
        Return final reward from the perspective of the player who just acted.
        """
        if self.winning_team is None:
            return 0

        player_team = TEAM_BY_PLAYER[player_id]

        if player_team == self.winning_team:
            return 1

        return -1

    def legal_actions(self) -> List[int]:
        """
        Return legal action ids for the current player.
        """
        observation = self.get_observation()

        return [
            card_id
            for card_id, is_legal in enumerate(observation.legal_action_mask)
            if is_legal == 1
        ]