"""
Random Opponent Policy for Ultimate Tic-Tac-Toe

Provides a simple random policy that selects legal moves uniformly.
"""

import numpy as np

from ..board import Position, UltimateTicTacToeBoard


def random_policy(board: UltimateTicTacToeBoard, rng: np.random.Generator) -> Position:
    """
    Return a random legal move for the opponent.
    
    Args:
        board: Current board state
        rng: Random number generator
        
    Returns:
        Random legal position
    """
    legal_moves = board.get_legal_moves()
    return rng.choice(legal_moves) 