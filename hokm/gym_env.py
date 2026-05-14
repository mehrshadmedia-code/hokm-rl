from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hokm.env import HokmEnv
from hokm.game import TEAM_BY_PLAYER
from hokm.observation import observation_to_flat_vector
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy


class HokmGymEnv(gym.Env):
    """
    Gymnasium-compatible Hokm environment.

    This is a first single-agent wrapper:
    - Learning agent controls one player, default Player 0.
    - Other players are controlled by fixed policies.
    - The environment automatically plays opponent/partner turns
      until it is the learning player's turn again or the hand ends.

    Action space:
        Discrete(52), where each action is a card id.

    Observation space:
        Flat numeric vector from PlayerObservation.

    Reward:
        +1 if learning player's team wins the hand.
        -1 if learning player's team loses the hand.
         0 during the hand.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        learning_player: int = 0,
        opponent_policy_name: str = "simple",
        partner_policy_name: str = "simple",
        seed: Optional[int] = None,
    ):
        super().__init__()

        if learning_player not in range(4):
            raise ValueError("learning_player must be from 0 to 3.")

        self.learning_player = learning_player
        self.opponent_policy_name = opponent_policy_name
        self.partner_policy_name = partner_policy_name
        self.base_seed = seed
        self.episode_count = 0

        self.env: Optional[HokmEnv] = None
        self.policies: Dict[int, object] = {}

        self.action_space = spaces.Discrete(52)

        # observation_to_flat_vector length:
        # player_id: 1
        # hand_vector: 52
        # played_cards_vector: 52
        # current_trick_vector: 52
        # legal_action_mask: 52
        # trump_vector: 4
        # team_tricks: 2
        # is_hakem: 1
        obs_length = 1 + 52 + 52 + 52 + 52 + 4 + 2 + 1

        self.observation_space = spaces.Box(
            low=0,
            high=52,
            shape=(obs_length,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        episode_seed = self._make_episode_seed(seed)

        self.env = HokmEnv(
            seed=episode_seed,
            hakem=0,
        )

        self.env.reset()
        self.policies = self._make_policies(seed=episode_seed)

        self._auto_play_until_learning_player_or_done()

        observation = self._get_learning_observation()

        info = {
            "learning_player": self.learning_player,
            "trump_suit": self.env.trump_suit,
            "legal_actions": self.get_legal_actions(),
            "action_mask": self.action_mask(),
        }

        self.episode_count += 1

        return observation, info

    def step(self, action: int):
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        if self.env.done:
            raise RuntimeError("Cannot step after episode is done. Call reset().")

        if self.env.current_player != self.learning_player:
            raise RuntimeError(
                "It is not the learning player's turn. "
                "Internal auto-play should have handled this."
            )

        legal_actions = self.get_legal_actions()

        if action not in legal_actions:
            # Penalize illegal action and terminate.
            observation = self._get_learning_observation()
            reward = -1.0
            terminated = True
            truncated = False
            info = {
                "illegal_action": True,
                "legal_actions": legal_actions,
                "action_mask": self.action_mask(),
            }
            return observation, reward, terminated, truncated, info


        team_tricks_before = self.env.team_tricks.copy()

        self.env.step(action)

        self._auto_play_until_learning_player_or_done()

        terminated = self.env.done
        truncated = False

        reward = self._shaped_reward_since(team_tricks_before)

        if terminated:
            reward += self._terminal_reward_for_learning_player()
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_learning_observation()


        info = {
            "illegal_action": False,
            "learning_player": self.learning_player,
            "winning_team": self.env.winning_team,
            "team_tricks": self.env.team_tricks.copy(),
            "legal_actions": [] if terminated else self.get_legal_actions(),
            "action_mask": np.zeros(52, dtype=np.int8) if terminated else self.action_mask(),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.env is None:
            print("Environment has not been reset.")
            return

        print("=" * 60)
        print("HokmGymEnv")
        print("=" * 60)
        print(f"Learning player: {self.learning_player}")
        print(f"Current player: {self.env.current_player}")
        print(f"Trump suit: {self.env.trump_suit}")
        print(f"Team tricks: {self.env.team_tricks}")
        print(f"Current trick: {self.env.current_trick}")
        print(f"Done: {self.env.done}")
        print(f"Winning team: {self.env.winning_team}")

    def action_mask(self) -> np.ndarray:
        """
        Return legal action mask for MaskablePPO later.

        1 means legal.
        0 means illegal.
        """
        if self.env is None or self.env.done:
            return np.zeros(52, dtype=np.int8)

        mask = np.zeros(52, dtype=np.int8)

        for action in self.get_legal_actions():
            mask[action] = 1

        return mask
    

    def action_masks(self) -> np.ndarray:
        """
        Compatibility method for sb3-contrib MaskablePPO.

        MaskablePPO expects this method name.
        True/1 means the action is valid.
        False/0 means the action is invalid.
        """
        return self.action_mask().astype(bool)
    

    def get_legal_actions(self):
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        return self.env.legal_actions()

    def _get_learning_observation(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        observation = self.env.get_observation()
        flat = observation_to_flat_vector(observation)

        return np.array(flat, dtype=np.float32)

    def _auto_play_until_learning_player_or_done(self) -> None:
        """
        Let fixed policies play until it is the learning player's turn again.
        """
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        while not self.env.done and self.env.current_player != self.learning_player:
            player_id = self.env.current_player
            policy = self.policies[player_id]

            action = policy.choose_action(
                player_id=player_id,
                hand=self.env.hands[player_id],
                current_trick=self.env.current_trick,
                trump_suit=self.env.trump_suit,
            )

            self.env.step(action)

    def _terminal_reward_for_learning_player(self) -> float:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        if self.env.winning_team is None:
            return 0.0

        learning_team = TEAM_BY_PLAYER[self.learning_player]

        if learning_team == self.env.winning_team:
            return 1.0

        return -1.0

    def _shaped_reward_since(self, team_tricks_before: Dict[int, int]) -> float:
            """
            Small reward signal for tricks won/lost since the learning player's last action.
            """
            if self.env is None:
                raise RuntimeError("Environment has not been reset.")

            learning_team = TEAM_BY_PLAYER[self.learning_player]
            opponent_team = 1 - learning_team

            learning_team_gained = (
                self.env.team_tricks[learning_team] - team_tricks_before[learning_team]
            )
            opponent_team_gained = (
                self.env.team_tricks[opponent_team] - team_tricks_before[opponent_team]
            )

            reward = 0.0
            reward += 0.05 * learning_team_gained
            reward -= 0.05 * opponent_team_gained

            return reward
    
    def _make_policies(self, seed: int) -> Dict[int, object]:
        policies = {}

        learning_team = TEAM_BY_PLAYER[self.learning_player]

        for player_id in range(4):
            if player_id == self.learning_player:
                continue

            player_team = TEAM_BY_PLAYER[player_id]

            if player_team == learning_team:
                policies[player_id] = self._make_policy(
                    self.partner_policy_name,
                    seed=seed + player_id,
                )
            else:
                policies[player_id] = self._make_policy(
                    self.opponent_policy_name,
                    seed=seed + player_id,
                )

        return policies

    @staticmethod
    def _make_policy(policy_name: str, seed: int):
        if policy_name == "simple":
            return SimpleRulePolicy()

        if policy_name == "random":
            return RandomLegalPolicy(seed=seed)

        raise ValueError(f"Unknown policy name: {policy_name}")

    def _make_episode_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return seed

        if self.base_seed is not None:
            return self.base_seed + self.episode_count

        return self.episode_count