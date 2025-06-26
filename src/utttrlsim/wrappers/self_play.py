"""
SelfPlayWrapper for Ultimate Tic-Tac-Toe

This wrapper converts the environment to a single-agent learning perspective
by automatically handling opponent moves and converting rewards to 0-sum.
"""

from typing import Any, Callable, Dict, Protocol, Tuple
import gymnasium as gym
import numpy as np

from ..board import Player, Position, UltimateTicTacToeBoard
from ..policies.random import random_policy


class UltimateTicTacToeEnv(gym.Env):
    board: UltimateTicTacToeBoard
    np_random: np.random.Generator


class SelfPlayWrapper(gym.Wrapper):
    """
    Wrapper that converts Ultimate Tic-Tac-Toe to single-agent learning perspective.

    This wrapper:
    1. Automatically executes opponent moves
    2. Converts rewards to 0-sum from agent perspective
    3. Optionally flips board observations for O player
    4. Ensures agent always plays when step() is called
    """

    def __init__(
        self,
        env: UltimateTicTacToeEnv,
        agent_piece: Player,
        opponent_policy: (
            Callable[[UltimateTicTacToeBoard, np.random.Generator], Position] | None
        ) = None,
        flip_observation: bool = True,
    ):
        """
        Initialize SelfPlayWrapper.

        Args:
            env: Base Ultimate Tic-Tac-Toe environment
            agent_piece: Which piece the learning agent plays (Player.X or Player.O)
            opponent_policy: Function that returns opponent moves
            flip_observation: Whether to flip board for O player
        """
        super().__init__(env)

        self.agent_piece = agent_piece
        self.opponent_policy = (
            opponent_policy if opponent_policy is not None else random_policy
        )
        self.flip_observation = flip_observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment and ensure agent plays first.

        Returns:
            Tuple of (observation, info) where observation is from agent perspective
        """
        obs, info = super().reset(seed=seed, options=options)

        # Flip board if needed for O player
        if self.flip_observation and self.agent_piece == Player.O:
            obs = self._flip_board_in_obs(obs)

        # If opponent goes first, execute opponent moves until agent's turn
        if (
            self.env.board.current_player != self.agent_piece
            and not self.env.board.game_over
        ):
            opp_pos = self.opponent_policy(self.env.board, self.env.np_random)
            opp_act: int = opp_pos.board_id
            obs, _, _, _, info = super().step(opp_act)

            # Flip board if needed for O player
            if self.flip_observation and self.agent_piece == Player.O:
                obs = self._flip_board_in_obs(obs)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take agent action, then execute one opponent move until agent's turn again.

        Args:
            action: Agent's move (0-80)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            where reward is from agent perspective (0-sum)
        """
        # Ensure it's agent's turn
        if self.env.board.current_player != self.agent_piece:
            raise ValueError(
                f"Expected agent's turn (Player {self.agent_piece.name}), "
                f"got Player {self.env.board.current_player.name}"
            )

        # Execute agent's move
        obs, rew, term, trunc, info = self.env.step(action)
        agent_reward = rew  # Agent's reward (same sign as original)

        # Flip board if needed for O player (after agent's move)
        if self.flip_observation and self.agent_piece == Player.O:
            obs = self._flip_board_in_obs(obs)

        # Execute exactly one opponent move (game is strictly alternating turns)
        if not (term or trunc) and self.env.board.current_player != self.agent_piece:
            opp_pos = self.opponent_policy(self.env.board, self.env.np_random)
            obs, opp_rew, term, trunc, info = self.env.step(opp_pos.board_id)
            agent_reward -= opp_rew  # 0‑sum: opponent's reward is negative for agent

            # Flip board if needed for O player (after opponent's move)
            if self.flip_observation and self.agent_piece == Player.O:
                obs = self._flip_board_in_obs(obs)

        return obs, agent_reward, term, trunc, info

    def _flip_board_in_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Flip board observation for O player perspective.

        Args:
            obs: Original observation

        Returns:
            Observation with flipped board (X and O swapped) and action mask
        """
        new_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        
        # Board flip
        x = Player.X.value
        o = Player.O.value
        new_obs["board"] = np.where(new_obs["board"] == x, o, np.where(new_obs["board"] == o, x, new_obs["board"]))

        return new_obs

    def get_action_mask(self):
        return self.env.get_action_mask()

