"""
Ultimate Tic-Tac-Toe Gymnasium Environment

A Gymnasium-compatible environment for Ultimate Tic-Tac-Toe.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import Player, UltimateTicTacToeBoard, Position


class UltimateTicTacToeEnv(gym.Env):
    """
    Ultimate Tic-Tac-Toe environment compatible with Gymnasium.

    This environment implements the standard Gymnasium interface:
    - reset(): Reset the environment to initial state
    - step(action): Take an action and return (observation, reward, done, truncated, info)
    - render(): Render the current state
    - seed(): Set random seed for reproducibility

    Player X always trainer, Player O is an opponent (can be random or another agent).
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Ultimate Tic-Tac-Toe environment.

        Args:
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        super().__init__()

        self.render_mode = render_mode
        self.board = UltimateTicTacToeBoard()

        # Action space: 81 possible moves (9 sub-boards × 9 positions)
        # Actions are encoded as: sub_board * 9 + position
        self.action_space = spaces.Discrete(81)

        # Observation space:
        #  - "board": 9×9 数値 (0: empty, 1: X, 2: O)
        #  - "action_mask": 81 次元バイナリ (1: 打てる, 0: 打てない)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=2, shape=(9, 9), dtype=np.int8),
                "action_mask": spaces.MultiBinary(81),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Let the Gymnasium base class handle core seeding & bookkeeping
        super().reset(seed=seed, options=options)

        # Choose X or O deterministically via the numpy RNG
        player = Player(self.np_random.integers(1, 3))  # 1 → X, 2 → O
        self.board.reset(current_player=player)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Action to take (0-80, encoded as sub_board * 9 + position)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Store the player who made this move (before the move is made)
        current_player = self.board.current_player

        # Make move
        # Create Position object from action
        position_obj = Position(action)
        self.board.make_move(position_obj)
        # Note: make_move() switches current_player at the end

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        # Calculate reward based on the player who made the move
        reward = self._calculate_reward(current_player)

        # Check if episode is done
        terminated = self.board.game_over

        return observation, reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment.

        Returns:
            Rendered image (for rgb_array mode) or None (for human mode)
        """
        if self.render_mode == "human":
            raise NotImplementedError(
                "Human rendering is not implemented yet. Use 'rgb_array' mode instead."
            )
        elif self.render_mode == "rgb_array":
            raise NotImplementedError(
                "RGB array rendering is not implemented yet. This method should return an RGB image."
            )
        else:
            return None

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation.

        Returns:
            Dict with keys:
              - "board": current board state (9×9 np.int8)
              - "action_mask": 1D np.uint8 array of length 81 (1 = legal)
        """
        return {
            "board": self.board.board.copy(),
            "action_mask": self._get_action_mask(),
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.

        Returns:
            Dictionary containing additional info
        """
        return {
            "meta_board": self.board.subboard_winner.copy(),
            "current_player": self.board.current_player.value,
            "legal_moves": self.board.get_legal_moves(),
            "game_over": self.board.game_over,
            "winner": (
                self.board.winner.value if self.board.winner != Player.EMPTY else None
            ),
            "last_move": self.board.last_move,
        }

    def _calculate_reward(self, current_player: Player) -> float:
        """
        Calculate reward for the current state.

        Returns:
            Reward value: +1 for win, 0 for draw, -1 for loss
        """
        if not self.board.game_over:
            return 0.0

        if self.board.winner == Player.EMPTY:
            # Draw
            return 0.0
        elif self.board.winner == current_player:
            return 1.0
        else:
            return -1.0

    def _get_action_mask(self) -> np.ndarray:
        """
        Returns
        -------
        mask : np.ndarray(uint8, shape=(81,))
            1 → 今打てる
            0 → 打てない（サブボードが埋まっている など）
        """
        legal_moves = self.board.get_legal_moves()
        mask = np.zeros(81, dtype=np.int8)

        for position in legal_moves:
            mask[position.board_id] = 1

        return mask

    def close(self) -> None:
        """Close the environment."""
        pass
