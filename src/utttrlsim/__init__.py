"""
Ultimate Tic-Tac-Toe Reinforcement Learning Simulator

A Gymnasium-compatible environment for Ultimate Tic-Tac-Toe with RL agents.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .board import UltimateTicTacToeBoard
from .env import UltimateTicTacToeEnv

__all__ = ["UltimateTicTacToeEnv", "UltimateTicTacToeBoard"]
