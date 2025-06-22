"""
Tests for UltimateTicTacToeEnv
"""

import gymnasium as gym
import numpy as np
import pytest

from utttrlsim.board import Player
from utttrlsim.env import UltimateTicTacToeEnv


class TestUltimateTicTacToeEnv:
    """Test cases for UltimateTicTacToeEnv"""

    def test_initialization(self):
        """Test environment initialization"""
        # _Arrange_
        # (No setup needed for initialization test)

        # _Act_
        env = UltimateTicTacToeEnv()

        # _Assert_
        assert env.action_space.n == 81
        assert env.observation_space.shape == (9, 9)
        assert env.observation_space.dtype == np.int8
        assert np.all(env.observation_space.low == 0)
        assert np.all(env.observation_space.high == 2)

    def test_reset(self):
        """Test environment reset"""
        # _Arrange_
        env = UltimateTicTacToeEnv()
        env.step(40)  # sub_board 4, position 4
        env.step(0)  # sub_board 0, position 0

        # _Act_
        observation, info = env.reset()

        # _Assert_
        # Check observation
        assert observation.shape == (9, 9)
        assert np.all(observation == 0)

        # Check info
        assert "meta_board" in info
        assert "current_player" in info
        assert "legal_moves" in info
        assert "game_over" in info
        assert "winner" in info
        assert "last_move" in info

        # Check board state
        assert np.all(env.board.board == 0)
        assert np.all(env.board.meta_board == 0)
        assert env.board.current_player == Player.X
        assert not env.board.game_over

    def test_step_valid_move(self):
        """Test making valid moves"""
        # _Arrange_
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()
        action = 40  # sub_board 4, position 4

        # _Act_
        observation, reward, done, truncated, info = env.step(action)

        # _Assert_
        # Check observation
        assert observation.shape == (9, 9)
        assert observation[4, 4] == 1  # X's move

        # Check reward (should be 0 for non-terminal state)
        assert reward == 0.0

        # Check done (should be False)
        assert not done

        # Check info
        assert info["current_player"] == 2  # O's turn
        assert info["last_move"] == (4, 4)
        assert not info["game_over"]

    def test_step_invalid_move(self):
        """Test making invalid moves"""
        # _Arrange_
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()
        env.step(40)  # Make a valid move first

        # _Act_
        # Try to make the same move again (invalid)
        observation, reward, done, truncated, info = env.step(40)

        # _Assert_
        # Should be penalized and game should end
        assert reward == -100.0
        assert done

    def test_step_out_of_bounds(self):
        """Test making out-of-bounds moves"""
        # _Arrange_
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()

        # _Act_ & _Assert_
        # Try invalid action - should raise ValueError
        with pytest.raises(ValueError):
            env.step(100)

    def test_legal_actions(self):
        """Test legal actions mask"""
        # _Arrange_
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()

        # _Act_
        legal_actions = env.get_legal_actions()

        # _Assert_
        # Should have 81 actions, all legal at start
        assert legal_actions.shape == (81,)
        assert np.all(legal_actions)

        # Additional test: make a move and check legal actions
        env.step(40)
        legal_actions = env.get_legal_actions()

        # Should have 8 legal actions in sub-board 4
        assert np.sum(legal_actions) == 8
        # Check that action 40 is not legal anymore
        assert not legal_actions[40]

    def test_game_termination(self):
        """Test game termination conditions"""
        # _Arrange_
        env = UltimateTicTacToeEnv()

        # _Act_
        observation, info = env.reset()

        # _Assert_
        # Create a winning pattern (simplified test)
        # This is a complex test - for now, just test basic functionality
        assert not info["game_over"]

    def test_render_human(self):
        """Test human rendering mode"""
        # _Arrange_
        env = UltimateTicTacToeEnv(render_mode="human")
        observation, info = env.reset()

        # _Act_
        result = env.render()

        # _Assert_
        # Render should not raise an error
        assert result is None

    def test_render_rgb_array(self):
        """Test RGB array rendering mode"""
        # _Arrange_
        env = UltimateTicTacToeEnv(render_mode="rgb_array")
        observation, info = env.reset()
        env.step(40)  # Make a move

        # _Act_
        result = env.render()

        # _Assert_
        # Render should return an image
        assert result is not None
        assert result.shape == (300, 300, 3)
        assert result.dtype == np.uint8

    def test_seed(self):
        """Test seeding functionality"""
        # _Arrange_
        env = UltimateTicTacToeEnv()

        # _Act_
        observation, info = env.reset(seed=42)

        # _Assert_
        # Check that environment was reset properly
        assert observation.shape == (9, 9)
        assert np.all(observation == 0)

    def test_close(self):
        """Test environment closing"""
        # _Arrange_
        env = UltimateTicTacToeEnv()

        # _Act_
        env.close()

        # _Assert_
        # Close should not raise an error (no explicit assert needed)
        pass

    def test_gymnasium_compatibility(self):
        """Test that environment is compatible with Gymnasium"""
        # _Arrange_
        # (No setup needed for interface testing)

        # _Act_
        env = UltimateTicTacToeEnv()

        # _Assert_
        # Check that environment follows Gymnasium interface
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "render")
        assert hasattr(env, "close")

        # Check that spaces are valid
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
