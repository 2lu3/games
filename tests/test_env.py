"""
Tests for UltimateTicTacToeEnv
"""

import gymnasium as gym
import numpy as np
import pytest

from utttrlsim.board import Player, Position
from utttrlsim.env import UltimateTicTacToeEnv


class TestUltimateTicTacToeEnv:
    """Test cases for UltimateTicTacToeEnv"""

    def test_initialization(self):
        """Test environment initialization"""
        env = UltimateTicTacToeEnv()

        assert env.action_space.n == 81
        assert env.observation_space.shape == (9, 9)
        assert env.observation_space.dtype == np.int8
        assert np.all(env.observation_space.low == 0)
        assert np.all(env.observation_space.high == 2)

    def test_reset(self):
        """Test environment reset"""
        # Arrange
        env = UltimateTicTacToeEnv()
        # Make some moves to change the environment state
        env.step(40)  # sub_board 4, position 4
        env.step(0)  # sub_board 0, position 0

        # Act
        # Reset the environment
        observation, info = env.reset()

        # Assert
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
        assert np.all(env.board.subboard_winner == 0)
        assert env.board.current_player == Player.X
        assert not env.board.game_over

    def test_step_valid_move(self):
        """Test making valid moves"""
        # Arrange
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()
        action = 40  # sub_board 4, position 4

        # Act
        # Make a valid move
        observation, reward, done, truncated, info = env.step(action)

        # Assert
        # Check observation
        assert observation.shape == (9, 9)
        np.testing.assert_equal(
            observation[4, 4], 
            1, 
            err_msg="observation[4, 4]は1であるべき - Xの着手後"
        )

        # Check reward (should be 0 for non-terminal state)
        assert reward == 0.0

        # Check done (should be False)
        assert not done

        # Check info
        assert info["current_player"] == 2  # O's turn
        assert isinstance(info["last_move"], Position)
        assert info["last_move"].board_id == 40
        assert not info["game_over"]

    def test_step_invalid_move(self):
        """Test making invalid moves"""
        # Arrange
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()
        # Make a move first
        env.step(40)

        # Act
        # Try to make the same move again (invalid)
        observation, reward, done, truncated, info = env.step(40)

        # Assert
        # Should be penalized and game should end
        assert reward == -100.0
        assert done

    def test_step_out_of_bounds(self):
        """Test making out-of-bounds moves"""
        env = UltimateTicTacToeEnv()

        observation, info = env.reset()

        # Try invalid action
        with pytest.raises(ValueError):
            env.step(100)

    def test_legal_actions(self):
        """Test legal actions mask"""
        # Arrange
        env = UltimateTicTacToeEnv()
        observation, info = env.reset()

        # Act
        legal_actions = env.get_legal_actions()

        # Assert
        # Should have 81 actions, all legal at start
        assert legal_actions.shape == (81,)
        assert np.all(legal_actions)

        # Arrange (second phase)
        # Make a move
        env.step(40)

        # Act (second phase)
        # Get new legal actions
        legal_actions = env.get_legal_actions()

        # Assert (second phase)
        # Should have 8 legal actions in sub-board 4 (the cell position of the last move)
        # The last move was at position 40, which corresponds to cell (1, 1) in sub-board 4
        # So the next player must play in sub-board 4 (which is at grid position (1, 1))
        assert np.sum(legal_actions) == 8

        # Check that action 40 is not legal anymore
        assert not legal_actions[40]

        # Check that only actions in sub-board 4 are legal
        # Sub-board 4 corresponds to positions [30, 31, 32, 39, 40, 41, 48, 49, 50]
        # But position 40 is already taken, so legal actions are [30, 31, 32, 39, 41, 48, 49, 50]
        expected_legal_actions = [30, 31, 32, 39, 41, 48, 49, 50]
        for action in range(81):
            if legal_actions[action]:
                assert action in expected_legal_actions

    def test_position_class(self):
        """Test Position class functionality"""
        # Test initialization with board_id
        pos1 = Position(40)
        assert pos1.board_id == 40
        assert pos1.board_x == 4
        assert pos1.board_y == 4
        assert pos1.sub_grid_x == 1
        assert pos1.sub_grid_y == 1
        assert pos1.sub_grid_id == 4
        assert pos1.cell_x == 1
        assert pos1.cell_y == 1
        assert pos1.cell_id == 4
        
        # Test initialization with grid and cell coordinates
        pos2 = Position(1, 1, 1, 1)
        assert pos2.board_id == 40
        assert pos2 == pos1
        
        # Test corner position
        pos3 = Position(0)
        assert pos3.board_x == 0
        assert pos3.board_y == 0
        assert pos3.sub_grid_x == 0
        assert pos3.sub_grid_y == 0
        assert pos3.cell_x == 0
        assert pos3.cell_y == 0

    def test_sub_grid_mapping(self):
        """Test the mapping of sub-grids to board positions"""
        # Create a mapping of sub-grid coordinates to board positions
        sub_grid_positions = {}
        
        for board_id in range(81):
            pos = Position(board_id)
            sub_grid_key = (pos.sub_grid_x, pos.sub_grid_y)
            if sub_grid_key not in sub_grid_positions:
                sub_grid_positions[sub_grid_key] = []
            sub_grid_positions[sub_grid_key].append(board_id)
        
        # Test that sub-grid (1, 1) contains the expected positions
        sub_grid_1_1 = sub_grid_positions[(1, 1)]
        expected_positions = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        assert sorted(sub_grid_1_1) == expected_positions

    def test_game_termination(self):
        """Test game termination conditions"""
        env = UltimateTicTacToeEnv()

        observation, info = env.reset()

        # Create a winning pattern (simplified test)
        # This is a complex test - for now, just test basic functionality
        assert not info["game_over"]

    def test_render_human(self):
        """Test human rendering mode"""
        env = UltimateTicTacToeEnv(render_mode="human")

        observation, info = env.reset()

        # Render should not raise an error
        result = env.render()
        assert result is None

    def test_render_rgb_array(self):
        """Test RGB array rendering mode"""
        # Arrange
        env = UltimateTicTacToeEnv(render_mode="rgb_array")
        observation, info = env.reset()
        # Make a move to create a testable state
        env.step(40)

        # Act
        # Render should return an image
        result = env.render()

        # Assert
        assert result is not None
        assert result.shape == (300, 300, 3)
        assert result.dtype == np.uint8

    def test_seed(self):
        """Test seeding functionality"""
        env = UltimateTicTacToeEnv()

        # Set seed via reset
        observation, info = env.reset(seed=42)

        # Check that environment was reset properly
        assert observation.shape == (9, 9)
        assert np.all(observation == 0)

    def test_close(self):
        """Test environment closing"""
        env = UltimateTicTacToeEnv()

        # Close should not raise an error
        env.close()

    def test_gymnasium_compatibility(self):
        """Test that environment is compatible with Gymnasium"""
        env = UltimateTicTacToeEnv()

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
