from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hokm.env import HokmEnv
from hokm.game import TEAM_BY_PLAYER
from hokm.observation import observation_to_flat_vector
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy


class HokmTeamGymEnv(gym.Env):
    """
    Gymnasium environment where the learning model controls Team 0:
    - Player 0
    - Player 2

    Opponents:
    - Player 1
    - Player 3

    The same policy is used for both Player 0 and Player 2.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent_policy_name: str = "simple",
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.team_players = {0, 2}
        self.opponent_policy_name = opponent_policy_name
        self.base_seed = seed
        self.episode_count = 0

        self.env: Optional[HokmEnv] = None
        self.opponent_policies: Dict[int, object] = {}

        self.action_space = spaces.Discrete(52)

        # Current feature observation size:
        # 1 + 52 + 52 + 52 + 52 + 4 + 2 + 1 + 10 = 226
        obs_length = 1 + 52 + 52 + 52 + 52 + 4 + 2 + 1 + 10

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

        self.env = HokmEnv(seed=episode_seed, hakem=0)
        self.env.reset()

        self.opponent_policies = {
            1: self._make_policy(self.opponent_policy_name, episode_seed + 1),
            3: self._make_policy(self.opponent_policy_name, episode_seed + 3),
        }

        self._auto_play_until_team_turn_or_done()

        observation = self._get_team_observation()

        info = {
            "team_players": sorted(self.team_players),
            "current_player": self.env.current_player,
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

        if self.env.current_player not in self.team_players:
            raise RuntimeError(
                "It is not Team 0's turn. Auto-play should have handled this."
            )

        legal_actions = self.get_legal_actions()

        if action not in legal_actions:
            observation = self._get_team_observation()
            reward = -1.0
            terminated = True
            truncated = False
            info = {
                "illegal_action": True,
                "legal_actions": legal_actions,
                "action_mask": self.action_mask(),
                "current_player": self.env.current_player,
            }
            return observation, reward, terminated, truncated, info

        team_tricks_before = self.env.team_tricks.copy()

        acting_player = self.env.current_player
        self.env.step(action)

        self._auto_play_until_team_turn_or_done()

        terminated = self.env.done
        truncated = False

        reward = self._shaped_reward_since(team_tricks_before)

        if terminated:
            reward += self._terminal_reward_for_team()
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_team_observation()

        info = {
            "illegal_action": False,
            "acting_player": acting_player,
            "current_player": None if terminated else self.env.current_player,
            "winning_team": self.env.winning_team,
            "team_tricks": self.env.team_tricks.copy(),
            "legal_actions": [] if terminated else self.get_legal_actions(),
            "action_mask": np.zeros(52, dtype=np.int8) if terminated else self.action_mask(),
        }

        return observation, float(reward), terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        if self.env is None or self.env.done:
            return np.zeros(52, dtype=np.int8)

        mask = np.zeros(52, dtype=np.int8)

        for action in self.get_legal_actions():
            mask[action] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        return self.action_mask().astype(bool)

    def get_legal_actions(self):
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        return self.env.legal_actions()

    def _get_team_observation(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        observation = self.env.get_observation()
        flat = observation_to_flat_vector(observation)

        return np.array(flat, dtype=np.float32)

    def _auto_play_until_team_turn_or_done(self) -> None:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        while not self.env.done and self.env.current_player not in self.team_players:
            player_id = self.env.current_player
            policy = self.opponent_policies[player_id]

            action = policy.choose_action(
                player_id=player_id,
                hand=self.env.hands[player_id],
                current_trick=self.env.current_trick,
                trump_suit=self.env.trump_suit,
            )

            self.env.step(action)

    def _terminal_reward_for_team(self) -> float:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        if self.env.winning_team is None:
            return 0.0

        return 1.0 if self.env.winning_team == 0 else -1.0

    def _shaped_reward_since(self, team_tricks_before: Dict[int, int]) -> float:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")

        team_0_gained = self.env.team_tricks[0] - team_tricks_before[0]
        team_1_gained = self.env.team_tricks[1] - team_tricks_before[1]

        reward = 0.0
        reward += 0.05 * team_0_gained
        reward -= 0.05 * team_1_gained

        return reward

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