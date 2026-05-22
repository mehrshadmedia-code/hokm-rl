import argparse
import tkinter as tk
from pathlib import Path
from typing import Dict, Set

import numpy as np
from sb3_contrib import MaskablePPO

from hokm.env import HokmEnv
from hokm.observation import observation_to_flat_vector
from hokm.policies import RandomLegalPolicy, SimpleRulePolicy
import traceback

SUIT_SYMBOLS = {
    "hearts": "H",
    "diamonds": "D",
    "clubs": "C",
    "spades": "S",
}

SUIT_ORDER = {
    "hearts": 0,
    "diamonds": 1,
    "clubs": 2,
    "spades": 3,
}

RANK_ORDER = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def format_card(card) -> str:
    return f"{card.rank}-{SUIT_SYMBOLS[card.suit]}"


def format_cards(cards) -> str:
    sorted_cards = sorted(
        cards,
        key=lambda card: (SUIT_ORDER[card.suit], RANK_ORDER[card.rank]),
    )
    return "  ".join(format_card(card) for card in sorted_cards)


def make_policy(policy_name: str, seed: int):
    if policy_name == "simple":
        return SimpleRulePolicy()

    if policy_name == "random":
        return RandomLegalPolicy(seed=seed)

    raise ValueError(f"Unknown policy name: {policy_name}")


class HokmVisualGame:
    def __init__(
        self,
        root: tk.Tk,
        model_path: str,
        model_players: Set[int],
        opponent_policy_name: str,
        seed: int,
        deterministic: bool,
        show_all_hands: bool,
    ):
        self.root = root
        self.root.title("Hokm RL Visual Viewer")

        self.model = MaskablePPO.load(model_path)
        self.model_players = model_players
        self.opponent_policy_name = opponent_policy_name
        self.seed = seed
        self.deterministic = deterministic
        self.show_all_hands = show_all_hands

        self.env = HokmEnv(seed=seed, hakem=0)
        self.env.reset()

        self.bot_policies: Dict[int, object] = {
            player_id: make_policy(opponent_policy_name, seed + player_id)
            for player_id in range(4)
            if player_id not in model_players
        }

        self.last_message = "Game started. Click Next Play."
        self.move_number = 0

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        self.board_text = tk.Text(
            self.root,
            height=28,
            width=110,
            font=("Courier", 13),
            wrap="word",
        )
        self.board_text.pack(padx=12, pady=12)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.next_button = tk.Button(
            self.button_frame,
            text="Next Play",
            font=("Arial", 14, "bold"),
            command=self.next_play,
            width=15,
        )
        self.next_button.grid(row=0, column=0, padx=5)

        self.auto_button = tk.Button(
            self.button_frame,
            text="Auto Play",
            font=("Arial", 14),
            command=self.auto_play,
            width=15,
        )
        self.auto_button.grid(row=0, column=1, padx=5)

        self.reset_button = tk.Button(
            self.button_frame,
            text="Reset",
            font=("Arial", 14),
            command=self.reset_game,
            width=15,
        )
        self.reset_button.grid(row=0, column=2, padx=5)

    def choose_action_for_current_player(self) -> int:
        player_id = self.env.current_player

        if player_id in self.model_players:
            observation = self.env.get_observation()
            obs = np.array(observation_to_flat_vector(observation), dtype=np.float32)
            action_mask = np.array(
                [value == 1 for value in observation.legal_action_mask],
                dtype=bool,
            )

            action, _ = self.model.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=action_mask,
            )

            action = int(action)

            if observation.legal_action_mask[action] != 1:
                legal_actions = [
                    index
                    for index, value in enumerate(observation.legal_action_mask)
                    if value == 1
                ]
                raise ValueError(
                    f"Model selected illegal action {action}. "
                    f"Legal actions are: {legal_actions}"
                )

            return action

        policy = self.bot_policies[player_id]

        action = policy.choose_action(
            player_id=player_id,
            hand=self.env.hands[player_id],
            current_trick=self.env.current_trick,
            trump_suit=self.env.trump_suit,
        )

        return int(action)
    
    def next_play(self):
        print("BUTTON CLICKED: Next Play", flush=True)

        try:
            if self.env.done:
                self.last_message = "Game is finished. Click Reset to play again."
                self.refresh()
                return

            player_id = self.env.current_player
            action = self.choose_action_for_current_player()

            if action is None:
                raise RuntimeError(f"No action returned for player {player_id}.")

            print(f"Before step: player={player_id}, action={action}", flush=True)

            result = self.env.step(action)

            played_card = result.info["played_card"]
            self.move_number += 1

            actor = "MODEL" if player_id in self.model_players else self.opponent_policy_name.upper()

            message = (
                f"Move {self.move_number}: Player {player_id} ({actor}) played "
                f"{format_card(played_card)}"
            )

            if result.info["trick_completed"]:
                message += (
                    f" | Trick winner: Player {result.info['trick_winner']} "
                    f"| Score: {result.info['team_tricks']}"
                )

            if self.env.done:
                message += f" | GAME OVER. Winning team: Team {self.env.winning_team}"

            print(message, flush=True)

            self.last_message = message
            self.refresh()
            self.root.update_idletasks()

        except Exception as exc:
            self.last_message = f"ERROR: {exc}"
            print("=" * 80)
            print("Error inside Next Play button:")
            traceback.print_exc()
            print("=" * 80)
            self.refresh()
            self.root.update_idletasks()

    def auto_play(self):
        if self.env.done:
            self.refresh()
            return

        self.next_play()

        if not self.env.done:
            self.root.after(700, self.auto_play)

    def reset_game(self):
        self.seed += 1
        self.env = HokmEnv(seed=self.seed, hakem=0)
        self.env.reset()

        self.bot_policies = {
            player_id: make_policy(self.opponent_policy_name, self.seed + player_id)
            for player_id in range(4)
            if player_id not in self.model_players
        }

        self.move_number = 0
        self.last_message = f"New game started with seed {self.seed}."
        self.history_text.delete("1.0", tk.END)
        self.add_history(self.last_message)
        self.next_button.config(state="normal")
        self.refresh()
    
    def refresh(self):
        model_players_text = ", ".join(str(player) for player in sorted(self.model_players))

        lines = []
        lines.append("=" * 90)
        lines.append("HOKM RL VISUAL VIEWER")
        lines.append("=" * 90)
        lines.append(f"Seed: {self.seed}")
        lines.append(f"Model player(s): {model_players_text}")
        lines.append(f"Trump: {self.env.trump_suit}")
        lines.append(f"Team 0: Players 0 & 2")
        lines.append(f"Team 1: Players 1 & 3")
        lines.append(f"Team tricks: {self.env.team_tricks}")
        lines.append(f"Current player: {self.env.current_player}")
        lines.append(f"Done: {self.env.done}")
        lines.append("-" * 90)

        lines.append("CURRENT TRICK:")
        if self.env.current_trick:
            for player_id, card in self.env.current_trick:
                lines.append(f"  Player {player_id}: {format_card(card)}")
        else:
            lines.append("  empty")

        lines.append("-" * 90)

        for player_id in range(4):
            is_current = player_id == self.env.current_player and not self.env.done
            is_model = player_id in self.model_players

            title = f"PLAYER {player_id}"
            if is_model:
                title += " [MODEL]"
            else:
                title += f" [{self.opponent_policy_name.upper()}]"

            if is_current:
                title += "  <-- CURRENT TURN"

            lines.append(title)

            if self.show_all_hands or is_model:
                lines.append(f"  Hand: {format_cards(self.env.hands[player_id])}")
            else:
                lines.append(f"  Hand: {len(self.env.hands[player_id])} cards hidden")

            lines.append("")

        lines.append("-" * 90)
        lines.append("LAST MESSAGE:")
        lines.append(self.last_message)
        lines.append("=" * 90)

        board = "\n".join(lines)

        self.board_text.delete("1.0", tk.END)
        self.board_text.insert(tk.END, board)
        self.board_text.see(tk.END)

        if self.env.done:
            self.next_button.config(state="disabled")
        else:
            self.next_button.config(state="normal")

        self.root.update_idletasks()
        
    def add_history(self, text: str):
        self.history_text.insert(tk.END, text + "\n")
        self.history_text.see(tk.END)

    def next_trick(self):
        if self.env.done:
            self.refresh()
            return

        starting_completed_tricks = len(self.env.completed_tricks)

        while not self.env.done and len(self.env.completed_tricks) == starting_completed_tricks:
            self.next_play()

        self.refresh()

def parse_model_players(text: str) -> Set[int]:
    players = {int(part.strip()) for part in text.split(",") if part.strip()}

    if not players:
        raise ValueError("At least one model player is required.")

    for player_id in players:
        if player_id not in range(4):
            raise ValueError("Model players must be between 0 and 3.")

    return players


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/stage_b_features_simple.zip",
        help="Path to trained MaskablePPO model.",
    )

    parser.add_argument(
        "--model-players",
        type=str,
        default="0",
        help='Comma-separated model players. Example: "0" or "0,2".',
    )

    parser.add_argument(
        "--opponent-policy-name",
        type=str,
        default="simple",
        choices=["simple", "random"],
        help="Policy used by non-model players.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Game seed.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic model actions.",
    )

    parser.add_argument(
        "--hide-opponent-hands",
        action="store_true",
        help="Hide non-model player hands.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    root = tk.Tk()

    app = HokmVisualGame(
        root=root,
        model_path=str(model_path),
        model_players=parse_model_players(args.model_players),
        opponent_policy_name=args.opponent_policy_name,
        seed=args.seed,
        deterministic=not args.stochastic,
        show_all_hands=not args.hide_opponent_hands,
    )
    print("GUI opened. Click Next Play and watch terminal output.", flush=True)
    root.mainloop()


if __name__ == "__main__":
    main()