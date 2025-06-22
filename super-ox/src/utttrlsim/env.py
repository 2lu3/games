"""
Ultimate Tic-Tac-Toe Gymnasium Environment

A Gymnasium-compatible environment for Ultimate Tic-Tac-Toe.
"""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import Player, UltimateTicTacToeBoard


class UltimateTicTacToeEnv(gym.Env):
    """
    Ultimate Tic-Tac-Toe environment compatible with Gymnasium.

    This environment implements the standard Gymnasium interface:
    - reset(): Reset the environment to initial state
    - step(action): Take an action and return (observation, reward, done, truncated, info)
    - render(): Render the current state
    - seed(): Set random seed for reproducibility
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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

        # Observation space: 9x9 board state
        # 0: Empty, 1: Player X, 2: Player O
        self.observation_space = spaces.Box(low=0, high=2, shape=(9, 9), dtype=np.int8)

        # Additional info space for meta-board
        self.meta_observation_space = spaces.Box(
            low=0, high=2, shape=(3, 3), dtype=np.int8
        )

        # Random number generator
        self.np_random = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Set up random number generator
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)

        self.board.reset()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Action to take (0-80, encoded as sub_board * 9 + position)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Decode action
        sub_board = action // 9
        position = action % 9

        # Validate action
        if not (0 <= action < 81):
            raise ValueError(f"Invalid action: {action}. Must be 0-80.")

        # Make move
        success = self.board.make_move(sub_board, position)

        if not success:
            # Invalid move - penalize heavily
            observation = self._get_observation()
            info = self._get_info()
            return observation, -100.0, True, False, info

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self.board.game_over

        return observation, reward, done, False, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment.

        Returns:
            Rendered image (for rgb_array mode) or None (for human mode)
        """
        if self.render_mode == "human":
            print(self.board.render())
            return None
        elif self.render_mode == "rgb_array":
            # For now, return a simple representation
            # In a full implementation, this would return an actual image
            return self._render_rgb_array()
        else:
            return None

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            Current board state as 9x9 numpy array
        """
        return self.board.board.copy()

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.

        Returns:
            Dictionary containing additional info
        """
        return {
            "meta_board": self.board.meta_board.copy(),
            "current_player": self.board.current_player.value,
            "legal_moves": self.board.get_legal_moves(),
            "game_over": self.board.game_over,
            "winner": (
                self.board.winner.value if self.board.winner != Player.EMPTY else None
            ),
            "last_move": self.board.last_move,
        }

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current state.

        Returns:
            Reward value
        """
        if not self.board.game_over:
            return 0.0

        if self.board.winner == Player.EMPTY:
            # Draw
            return 0.0
        elif self.board.winner == Player.X:
            # X wins (assuming agent is X)
            return 1.0
        else:
            # O wins (assuming agent is O)
            return -1.0

    def _render_rgb_array(self) -> np.ndarray:
        """
        Render the board as an RGB array.

        Returns:
            RGB array representation of the board
        """
        # Simple implementation - create a colored representation
        # In a full implementation, this would create an actual image
        height, width = 300, 300
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill with white background
        img.fill(255)

        # Draw grid lines
        for i in range(1, 9):
            # Vertical lines
            x = (i * width) // 9
            img[:, x - 1 : x + 1] = [100, 100, 100]

            # Horizontal lines
            y = (i * height) // 9
            img[y - 1 : y + 1, :] = [100, 100, 100]

        # Draw thicker lines for meta-board boundaries
        for i in range(1, 3):
            # Vertical meta-lines
            x = (i * width) // 3
            img[:, x - 2 : x + 2] = [50, 50, 50]

            # Horizontal meta-lines
            y = (i * height) // 3
            img[y - 2 : y + 2, :] = [50, 50, 50]

        # Draw pieces
        cell_height = height // 9
        cell_width = width // 9

        for row in range(9):
            for col in range(9):
                cell_value = self.board.board[row, col]
                if cell_value != 0:
                    center_y = row * cell_height + cell_height // 2
                    center_x = col * cell_width + cell_width // 2
                    radius = min(cell_height, cell_width) // 4

                    if cell_value == 1:  # X
                        color = [255, 0, 0]  # Red
                    else:  # O
                        color = [0, 0, 255]  # Blue

                    # Draw circle
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx * dx + dy * dy <= radius * radius:
                                y = center_y + dy
                                x = center_x + dx
                                if 0 <= y < height and 0 <= x < width:
                                    img[y, x] = color

        return img

    def get_legal_actions(self) -> np.ndarray:
        """
        Get mask of legal actions.

        Returns:
            Boolean array where True indicates legal actions
        """
        legal_moves = self.board.get_legal_moves()
        mask = np.zeros(81, dtype=bool)

        for sub_board, position in legal_moves:
            action = sub_board * 9 + position
            mask[action] = True

        return mask

    def close(self) -> None:
        """Close the environment."""
        pass
