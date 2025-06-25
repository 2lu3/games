"""
Random Agent for Ultimate Tic-Tac-Toe

A simple random agent that selects legal moves uniformly at random.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym

from ..board import UltimateTicTacToeBoard, Position


class RandomAgent:
    """
    Random agent that selects legal moves uniformly at random.

    This agent serves as a baseline for comparison with more sophisticated
    agents and algorithms.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random agent.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def select_action(self, env: gym.Env) -> int:
        """
        Select a random legal action from environment.

        Args:
            env: UltimateTicTacToeEnv instance

        Returns:
            Action index (0-80)
        """
        legal_moves = env.board.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available")

        pos = random.choice(legal_moves)
        return pos.board_id

    def get_action_probs(self, board: UltimateTicTacToeBoard) -> np.ndarray:
        """
        Get action probabilities for all possible actions.

        Args:
            board: Current board state

        Returns:
            Array of probabilities for each action (0-80)
        """
        legal_moves = board.get_legal_moves()
        probs = np.zeros(81)

        if legal_moves:
            prob = 1.0 / len(legal_moves)
            for position in legal_moves:
                action = position.board_id
                probs[action] = prob

        return probs
