"""
Property-based tests for Ultimate Tic-Tac-Toe board using Hypothesis.

Tests random legal move generation, differential consistency checks,
and boundary value testing to catch edge cases that manual tests might miss.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant
from typing import List, Set

from utttrlsim.board import UltimateTicTacToeBoard, Player, Position


class TestBoardProperties:
    """Property-based tests for board functionality."""

    @given(st.integers(min_value=0, max_value=80))
    def test_position_coordinate_conversion_roundtrip(self, board_id: int):
        """Test that position coordinate conversions are consistent."""
        pos = Position(board_id)
        
        # Test that all coordinate properties are within valid ranges
        assert 0 <= pos.board_x <= 8
        assert 0 <= pos.board_y <= 8
        assert 0 <= pos.sub_grid_x <= 2
        assert 0 <= pos.sub_grid_y <= 2
        assert 0 <= pos.cell_x <= 2
        assert 0 <= pos.cell_y <= 2
        assert 0 <= pos.sub_grid_id <= 8
        assert 0 <= pos.cell_id <= 8
        
        # Test roundtrip consistency
        pos2 = Position(pos.sub_grid_x, pos.sub_grid_y, pos.cell_x, pos.cell_y)
        assert pos == pos2
        assert pos.board_id == pos2.board_id

    @given(
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
    )
    def test_position_from_grid_coordinates(self, grid_x: int, grid_y: int, cell_x: int, cell_y: int):
        """Test position creation from grid coordinates."""
        pos = Position(grid_x, grid_y, cell_x, cell_y)
        
        assert pos.sub_grid_x == grid_x
        assert pos.sub_grid_y == grid_y
        assert pos.cell_x == cell_x
        assert pos.cell_y == cell_y
        assert 0 <= pos.board_id <= 80

    @given(st.lists(st.integers(min_value=0, max_value=80), min_size=1, max_size=50))
    def test_legal_moves_sequence_consistency(self, move_sequence: List[int]):
        """Test that a sequence of legal moves maintains board consistency."""
        board = UltimateTicTacToeBoard()
        original_board = board.copy()
        positions_played = set()
        
        for move_id in move_sequence:
            legal_moves = board.get_legal_moves()
            
            # Skip if no legal moves (game over)
            if not legal_moves:
                break
                
            # Find a legal move from the attempted move
            legal_move_ids = {pos.board_id for pos in legal_moves}
            if move_id not in legal_move_ids:
                # Use the first legal move instead
                move_id = legal_moves[0].board_id
            
            position = Position(move_id)
            
            # Record state before move
            previous_player = board.current_player
            previous_legal_moves = set(legal_moves)
            
            # Make the move
            board.make_move(position)
            positions_played.add(move_id)
            
            # Check invariants after move
            assert position not in board.get_legal_moves(), "Position should not be legal after being played"
            assert board.board[position.board_y, position.board_x] == previous_player.value
            assert board.current_player != previous_player, "Player Should switch after move"
            assert len(positions_played) <= 81, "Cannot play more than 81 moves"
            
            # Board should not have any invalid values
            assert np.all((board.board >= 0) & (board.board <= 2)), "Board values must be 0, 1, or 2"

    @given(st.integers(min_value=0, max_value=80))
    def test_single_move_differential_consistency(self, move_id: int):
        """Test differential consistency for a single move."""
        board = UltimateTicTacToeBoard()
        legal_moves = board.get_legal_moves()
        
        # Skip if the move is not legal
        legal_move_ids = {pos.board_id for pos in legal_moves}
        assume(move_id in legal_move_ids)
        
        position = Position(move_id)
        
        # Create copies before move
        board_copy1 = board.copy()
        board_copy2 = board.copy()
        
        # Make move on original
        original_player = board.current_player
        board.make_move(position)
        
        # Copies should be unchanged
        np.testing.assert_array_equal(board_copy1.board, board_copy2.board)
        assert board_copy1.current_player == board_copy2.current_player == original_player
        
        # Original should be different from copies
        assert not np.array_equal(board.board, board_copy1.board)
        assert board.current_player != board_copy1.current_player
        
        # Make same move on copy
        board_copy1.make_move(position)
        
        # Now original and copy1 should be identical
        np.testing.assert_array_equal(board.board, board_copy1.board)
        assert board.current_player == board_copy1.current_player
        assert board.last_move == board_copy1.last_move
        
        # copy2 should still be different
        assert not np.array_equal(board.board, board_copy2.board)

    @settings(max_examples=50)
    @given(st.lists(st.integers(min_value=0, max_value=80), min_size=2, max_size=20))
    def test_game_progression_invariants(self, move_sequence: List[int]):
        """Test invariants during game progression."""
        board = UltimateTicTacToeBoard()
        moves_made = 0
        
        for move_id in move_sequence:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Use a legal move
            legal_move_ids = {pos.board_id for pos in legal_moves}
            if move_id not in legal_move_ids:
                move_id = legal_moves[0].board_id
                
            position = Position(move_id)
            board.make_move(position)
            moves_made += 1
            
            # Check invariants
            assert moves_made <= 81, "Cannot make more than 81 moves"
            
            # If game is over, winner should be determined correctly
            if board.game_over:
                winner = board.winner
                assert winner in [Player.EMPTY, Player.X, Player.O]
                
                # If there's a winner, they should have a winning pattern on meta-board
                if winner != Player.EMPTY:
                    meta_board = board.subboard_winner
                    # Check for winning pattern (simplified check)
                    found_win = False
                    # Check rows
                    for i in range(3):
                        if np.all(meta_board[i, :] == winner.value):
                            found_win = True
                    # Check columns  
                    for j in range(3):
                        if np.all(meta_board[:, j] == winner.value):
                            found_win = True
                    # Check diagonals
                    if np.all(np.diag(meta_board) == winner.value):
                        found_win = True
                    if np.all(np.diag(np.fliplr(meta_board)) == winner.value):
                        found_win = True
                    
                    assert found_win, f"Winner {winner} should have winning pattern on meta-board"

    def test_empty_board_has_81_legal_moves(self):
        """Test that an empty board has exactly 81 legal moves."""
        board = UltimateTicTacToeBoard()
        legal_moves = board.get_legal_moves()
        
        assert len(legal_moves) == 81
        move_ids = {pos.board_id for pos in legal_moves}
        assert move_ids == set(range(81))

    @given(st.integers(min_value=0, max_value=80))
    def test_invalid_move_rejection(self, move_id: int):
        """Test that invalid moves are properly rejected."""
        board = UltimateTicTacToeBoard()
        position = Position(move_id)
        
        # Make the first move
        board.make_move(position)
        
        # Try to make the same move again - should raise ValueError
        with pytest.raises(ValueError):
            board.make_move(position)

    @given(st.lists(st.integers(min_value=0, max_value=2), min_size=9, max_size=9))
    def test_sub_board_win_detection(self, pattern: List[int]):
        """Test sub-board win detection with various patterns."""
        board = UltimateTicTacToeBoard()
        
        # Pattern is already valid player values (0, 1, 2)
        pattern_array = np.array(pattern).reshape(3, 3)
        
        # Check if X (1) or O (2) has a winning pattern
        has_x_win = board._check_win_pattern_for_player(pattern_array, Player.X)
        has_o_win = board._check_win_pattern_for_player(pattern_array, Player.O)
        
        # Set up the board with this pattern in sub-board (0,0)
        for i in range(3):
            for j in range(3):
                board.board[i, j] = pattern[i * 3 + j]
        
        # Check sub-board winner detection
        meta_board = board.subboard_winner
        
        if has_x_win and not has_o_win:
            assert meta_board[0, 0] == Player.X.value
        elif has_o_win and not has_x_win:
            assert meta_board[0, 0] == Player.O.value
        elif has_x_win and has_o_win:
            # Both players have winning patterns - in real gameplay this shouldn't happen
            # but in this test scenario, the board logic determines the winner
            # Check that some winner is determined (not empty)
            assert meta_board[0, 0] in [Player.X.value, Player.O.value]
        else:
            # No clear single winner or draw
            assert meta_board[0, 0] == Player.EMPTY.value


class TestAdvancedGameProperties:
    """Advanced property tests for game state management."""
    
    @settings(max_examples=20)
    @given(st.lists(st.integers(min_value=0, max_value=80), min_size=3, max_size=15))
    def test_random_game_sequence(self, move_ids: List[int]):
        """Test random game sequences maintain consistency."""
        board = UltimateTicTacToeBoard()
        move_history = []
        
        for move_id in move_ids:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # Convert to valid move
            legal_move_ids = {pos.board_id for pos in legal_moves}
            if move_id not in legal_move_ids:
                move_id = legal_moves[0].board_id
            
            position = Position(move_id)
            prev_player = board.current_player
            
            # Make move
            board.make_move(position)
            move_history.append((position, prev_player))
            
            # Verify board state
            assert board.board[position.board_y, position.board_x] == prev_player.value
            assert board.current_player != prev_player
            
            # Verify legal moves are valid
            current_legal_moves = board.get_legal_moves()
            for pos in current_legal_moves:
                assert board.board[pos.board_y, pos.board_x] == Player.EMPTY.value
        
        # Final consistency check
        assert np.all((board.board >= 0) & (board.board <= 2))
        
    @given(st.integers(min_value=1, max_value=10))
    def test_board_copy_independence(self, num_moves: int):
        """Test that board copies are independent."""
        board = UltimateTicTacToeBoard()
        original_copy = board.copy()
        
        moves_made = 0
        while moves_made < num_moves and not board.game_over:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Make move on original
            board.make_move(legal_moves[0])
            moves_made += 1
            
            # Original copy should remain unchanged
            np.testing.assert_array_equal(
                original_copy.board, 
                np.zeros((9, 9), dtype=np.int8),
                err_msg="Original copy should remain empty"
            )
            assert original_copy.current_player == Player.X
            assert original_copy.last_move is None


class TestBoundaryValues:
    """Test boundary values and edge cases."""
    
    def test_corner_positions(self):
        """Test moves at board corners."""
        board = UltimateTicTacToeBoard()
        corners = [
            Position(0),    # Top-left
            Position(8),    # Top-right  
            Position(72),   # Bottom-left
            Position(80),   # Bottom-right
        ]
        
        for corner in corners:
            board.reset()
            legal_moves = board.get_legal_moves()
            assert corner in legal_moves
            
            board.make_move(corner)
            assert board.board[corner.board_y, corner.board_x] != Player.EMPTY.value

    def test_center_positions(self):
        """Test moves at center positions."""
        board = UltimateTicTacToeBoard()
        centers = [
            Position(40),   # Global center
            Position(4),    # Top-middle sub-board center
            Position(76),   # Bottom-middle sub-board center
        ]
        
        for center in centers:
            board.reset()
            board.make_move(center)
            assert board.board[center.board_y, center.board_x] != Player.EMPTY.value

    @given(st.integers(min_value=0, max_value=80))
    def test_position_bounds(self, pos_id: int):
        """Test that all position IDs produce valid coordinates."""
        pos = Position(pos_id)
        
        # All coordinates should be within valid bounds
        assert 0 <= pos.board_x <= 8
        assert 0 <= pos.board_y <= 8
        assert 0 <= pos.sub_grid_x <= 2
        assert 0 <= pos.sub_grid_y <= 2
        assert 0 <= pos.cell_x <= 2
        assert 0 <= pos.cell_y <= 2
        
        # Board coordinates should match sub-grid and cell coordinates
        expected_board_x = pos.sub_grid_x * 3 + pos.cell_x
        expected_board_y = pos.sub_grid_y * 3 + pos.cell_y
        assert pos.board_x == expected_board_x
        assert pos.board_y == expected_board_y

    def test_maximum_game_length(self):
        """Test that games can reach maximum length without crashing."""
        board = UltimateTicTacToeBoard()
        moves_made = 0
        
        # Try to make moves until game is over or board is full
        while not board.game_over and moves_made < 81:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Make the first legal move
            board.make_move(legal_moves[0])
            moves_made += 1
            
            # Ensure we don't get stuck in infinite loop
            assert moves_made <= 81
        
        # Game should either be over or board should be analyzed correctly
        if moves_made == 81:
            # All positions should be filled
            assert np.all(board.board != Player.EMPTY.value)
        
        # Board state should be consistent
        assert np.all((board.board >= 0) & (board.board <= 2))