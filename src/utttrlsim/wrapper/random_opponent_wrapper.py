"""
Opponent Wrapper for Ultimate Tic-Tac-Toe

This wrapper manages the interaction between a PPO agent and an arbitrary opponent agent.
The PPO agent always plays as Player X, and the opponent agent plays as Player O.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple

from ..board import Player, Position
from ..agent.random_agent import RandomAgent


class RandomOpponentWrapper(gym.Wrapper):
    """
    Wrapper that pits a PPO agent against an arbitrary opponent agent.

    The wrapper ensures that:
    1. PPO agent always plays as Player X (first player)
    2. Opponent agent always plays as Player O (second player)
    3. Each step consists of PPO move followed by opponent move (if game continues)
    4. Rewards are calculated from PPO agent's perspective
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the OpponentWrapper.

        Args:
            env: The base Ultimate Tic-Tac-Toe environment
            opponent_agent: Opponent agent instance (must have select_action_from_env or predict)
        """
        super().__init__(env)
        self.opponent = RandomAgent()
        self.prev_player = None
        self.step_count = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and opponent.

        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(**kwargs)
        self.prev_player = self.env.unwrapped.board.current_player
        self.step_count = 0
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: PPO move followed by opponent agent move.

        Args:
            action: Action from PPO agent (0-80)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Step 1: PPO agent makes a move
        observation, _, done, truncated, info = self.env.step(action)

        # If game is over after PPO move, calculate reward from PPO perspective
        if done:
            final_reward = self._calculate_reward_from_ppo_perspective()
            return observation, final_reward, done, truncated, info

        # Step 2: Opponent agent makes a move (if game continues)
        if not done and self.env.unwrapped.board.current_player == Player.O:
            # 汎用化: select_action_from_env または predict または select_action
            if hasattr(self.opponent, "select_action_from_env"):
                opponent_action = self.opponent.select_action_from_env(self.env)
            elif hasattr(self.opponent, "predict"):
                # SB3系モデルの場合
                opponent_action, _ = self.opponent.predict(
                    observation, deterministic=True
                )
            elif hasattr(self.opponent, "select_action"):
                # RandomAgent等の場合
                opponent_action = self.opponent.select_action(self.env.unwrapped)
            else:
                raise AttributeError(
                    "Opponent agent must have select_action_from_env, predict, or select_action method."
                )
            observation, _, done, truncated, info = self.env.step(opponent_action)
            final_reward = self._calculate_reward_from_ppo_perspective()
            return observation, final_reward, done, truncated, info

        # If we reach here, something unexpected happened
        return observation, 0.0, done, truncated, info

    def _calculate_reward_from_ppo_perspective(self) -> float:
        """
        Calculate reward from PPO agent's perspective (Player X).

        This method ensures that rewards are always calculated from the PPO agent's
        perspective, regardless of which player made the last move.

        Returns:
            Reward value: +1 for PPO win, 0 for draw, -1 for PPO loss
        """
        if not self.env.unwrapped.board.game_over:
            return 0.0

        if self.env.unwrapped.board.winner == Player.EMPTY:
            # Draw
            return 0.0
        elif self.env.unwrapped.board.winner == Player.X:
            # PPO agent (Player X) wins
            return 1.0
        else:
            # PPO agent (Player X) loses (opponent wins)
            return -1.0

    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for the current state.

        Returns:
            Boolean array indicating legal actions
        """
        return self.env.unwrapped.get_action_mask()
