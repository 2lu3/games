"""
Edge case and boundary value tests for Ultimate Tic-Tac-Toe.

These tests focus on specific edge cases and boundary conditions that are
important for robustness but might not be easily covered by property-based testing.
"""

import numpy as np
import pytest
from src.utttrlsim.board import UltimateTicTacToeBoard, Player, Position


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_position_boundary_values(self):
        """Test Position class with boundary values"""
        # Test minimum position
        pos_min = Position(0)
        assert pos_min.board_id == 0
        assert pos_min.board_x == 0
        assert pos_min.board_y == 0
        assert pos_min.sub_grid_x == 0
        assert pos_min.sub_grid_y == 0
        assert pos_min.cell_x == 0
        assert pos_min.cell_y == 0
        
        # Test maximum position
        pos_max = Position(80)
        assert pos_max.board_id == 80
        assert pos_max.board_x == 8
        assert pos_max.board_y == 8
        assert pos_max.sub_grid_x == 2
        assert pos_max.sub_grid_y == 2
        assert pos_max.cell_x == 2
        assert pos_max.cell_y == 2
        
        # Test invalid positions
        with pytest.raises(AssertionError):
            Position(-1)
        with pytest.raises(AssertionError):
            Position(81)
    
    def test_position_coordinate_boundaries(self):
        """Test Position creation with coordinate boundaries"""
        # Test valid coordinate boundaries
        Position(0, 0, 0, 0)  # All minimum
        Position(2, 2, 2, 2)  # All maximum
        
        # Test invalid coordinate boundaries
        with pytest.raises(AssertionError):
            Position(-1, 0, 0, 0)  # Invalid grid_x
        with pytest.raises(AssertionError):
            Position(0, -1, 0, 0)  # Invalid grid_y
        with pytest.raises(AssertionError):
            Position(0, 0, -1, 0)  # Invalid cell_x
        with pytest.raises(AssertionError):
            Position(0, 0, 0, -1)  # Invalid cell_y
        with pytest.raises(AssertionError):
            Position(3, 0, 0, 0)   # Invalid grid_x
        with pytest.raises(AssertionError):
            Position(0, 3, 0, 0)   # Invalid grid_y
        with pytest.raises(AssertionError):
            Position(0, 0, 3, 0)   # Invalid cell_x
        with pytest.raises(AssertionError):
            Position(0, 0, 0, 3)   # Invalid cell_y
    
    def test_moves_after_game_over(self):
        """Test that moves cannot be made after game is over"""
        board = UltimateTicTacToeBoard()
        
        # Create a quick winning pattern for X in the meta-board
        # Win sub-boards (0,0), (1,0), (2,0) to win top row of meta-board
        
        # Sub-board (0,0) - X wins with top row
        for i in range(3):
            pos = Position(0, 0, i, 0)
            board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Sub-board (1,0) - X wins with top row  
        for i in range(3):
            pos = Position(1, 0, i, 0)
            board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Sub-board (2,0) - X wins with top row
        for i in range(3):
            pos = Position(2, 0, i, 0)
            board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Verify game is over and X won
        assert board.game_over
        assert board.winner == Player.X
        
        # Try to make a move after game is over
        with pytest.raises(RuntimeError, match="game is already over"):
            board.make_move(Position(40))
    
    def test_invalid_moves(self):
        """Test making invalid moves"""
        board = UltimateTicTacToeBoard()
        
        # Make first move
        first_move = Position(40)  # Center position
        board.make_move(first_move)
        
        # Try to make same move again (position already occupied)
        with pytest.raises(ValueError, match="Invalid move"):
            board.make_move(first_move)
        
        # Try to make move in wrong sub-board
        # After move at (40), next move should be in sub-board (1,1)
        wrong_subboard_move = Position(0, 0, 0, 0)  # This is in sub-board (0,0)
        with pytest.raises(ValueError, match="Invalid move"):
            board.make_move(wrong_subboard_move)
    
    def test_sub_board_full_no_winner(self):
        """Test scenario where sub-board is full but has no winner"""
        board = UltimateTicTacToeBoard()
        
        # Create a full sub-board (0,0) with no winner
        # Pattern: X O X
        #          O X O  
        #          X O X
        pattern = [
            Player.X, Player.O, Player.X,
            Player.O, Player.X, Player.O,
            Player.X, Player.O, Player.X
        ]
        
        for i, player in enumerate(pattern):
            pos = Position(0, 0, i % 3, i // 3)
            board.board[pos.board_y, pos.board_x] = player.value
        
        # Set up board state to direct play to the full sub-board
        board.last_move = Position(0, 0, 0, 0)  # Would direct to sub-board (0,0)
        board.current_player = Player.O
        
        # Legal moves should not include the full sub-board (0,0)
        legal_moves = board.get_legal_moves()
        for move in legal_moves:
            assert (move.sub_grid_x, move.sub_grid_y) != (0, 0)
        
        # Should have moves available in other sub-boards
        assert len(legal_moves) > 0
    
    def test_all_sub_boards_won_or_full(self):
        """Test game ending when all sub-boards are won or full"""
        board = UltimateTicTacToeBoard()
        
        # Create a scenario where all sub-boards are either won or full
        # Let's say X wins sub-boards (0,0), (1,1), (2,2) - diagonal win
        winning_positions = [
            [(0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0)],  # Top row win in (0,0)
            [(1, 1), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 2, 0)],  # Top row win in (1,1)  
            [(2, 2), (2, 2, 0, 0), (2, 2, 1, 0), (2, 2, 2, 0)],  # Top row win in (2,2)
        ]
        
        for sub_grid, *positions in winning_positions:
            for pos_args in positions:
                pos = Position(*pos_args)
                board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Fill remaining sub-boards as full with no winner
        remaining_subboards = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        full_pattern = [
            Player.X, Player.O, Player.X,
            Player.O, Player.X, Player.O,
            Player.X, Player.O, Player.X
        ]
        
        for grid_x, grid_y in remaining_subboards:
            for i, player in enumerate(full_pattern):
                pos = Position(grid_x, grid_y, i % 3, i // 3)
                board.board[pos.board_y, pos.board_x] = player.value
        
        # Game should be over with X as winner (diagonal in meta-board)
        assert board.game_over
        assert board.winner == Player.X
    
    def test_draw_scenario(self):
        """Test a draw scenario where no player wins the meta-board"""
        board = UltimateTicTacToeBoard()
        
        # Create a scenario where all sub-boards are full/won but no meta-board winner
        # Let's create a pattern that ensures no winning line:
        # Meta-board layout (what each sub-board position represents):
        # (0,0) (0,1) (0,2)
        # (1,0) (1,1) (1,2)  
        # (2,0) (2,1) (2,2)
        #
        # X wins: (0,0), (0,1), (1,2), (2,1) - no three in a row
        # O wins: (0,2), (1,0), (2,0), (2,2) - no three in a row
        # Draw: (1,1) - 1 sub-board full with no winner
        
        x_wins = [(0, 0), (0, 1), (1, 2), (2, 1)]
        o_wins = [(0, 2), (1, 0), (2, 0), (2, 2)]
        draw_subboard = (1, 1)
        
        # Set X wins (top row in each sub-board)
        for grid_x, grid_y in x_wins:
            for i in range(3):
                pos = Position(grid_x, grid_y, i, 0)
                board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Set O wins (top row in each sub-board)
        for grid_x, grid_y in o_wins:
            for i in range(3):
                pos = Position(grid_x, grid_y, i, 0)
                board.board[pos.board_y, pos.board_x] = Player.O.value
        
        # Set draw sub-board as full with no winner
        draw_pattern = [
            Player.X, Player.O, Player.X,
            Player.O, Player.X, Player.O,
            Player.X, Player.O, Player.X
        ]
        
        grid_x, grid_y = draw_subboard
        for i, player in enumerate(draw_pattern):
            pos = Position(grid_x, grid_y, i % 3, i // 3)
            board.board[pos.board_y, pos.board_x] = player.value
        
        # Game should be over with no winner (draw)
        assert board.game_over
        assert board.winner == Player.EMPTY
    
    def test_single_cell_sub_board_win(self):
        """Test winning a sub-board with exactly 3 in a row"""
        board = UltimateTicTacToeBoard()
        
        # Test each possible winning pattern in sub-board (0,0)
        winning_patterns = [
            # Rows
            [(0, 0), (1, 0), (2, 0)],  # Top row
            [(0, 1), (1, 1), (2, 1)],  # Middle row
            [(0, 2), (1, 2), (2, 2)],  # Bottom row
            # Columns  
            [(0, 0), (0, 1), (0, 2)],  # Left column
            [(1, 0), (1, 1), (1, 2)],  # Middle column
            [(2, 0), (2, 1), (2, 2)],  # Right column
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],  # Main diagonal
            [(2, 0), (1, 1), (0, 2)],  # Anti-diagonal
        ]
        
        for pattern in winning_patterns:
            board.reset()
            
            # Fill the winning pattern for X in sub-board (0,0)
            for cell_x, cell_y in pattern:
                pos = Position(0, 0, cell_x, cell_y)
                board.board[pos.board_y, pos.board_x] = Player.X.value
            
            # Check that sub-board (0,0) is won by X
            assert board.subboard_winner[0, 0] == Player.X.value
    
    def test_legal_moves_when_directed_to_won_subboard(self):
        """Test legal moves when directed to an already won sub-board"""
        board = UltimateTicTacToeBoard()
        
        # Win sub-board (1,1) for X (top row)
        for i in range(3):
            pos = Position(1, 1, i, 0)
            board.board[pos.board_y, pos.board_x] = Player.X.value
        
        # Set last move to direct play to the won sub-board (1,1)
        board.last_move = Position(0, 0, 1, 1)  # Cell (1,1) directs to sub-board (1,1)
        board.current_player = Player.O
        
        # Legal moves should not include the won sub-board (1,1)
        legal_moves = board.get_legal_moves()
        for move in legal_moves:
            assert (move.sub_grid_x, move.sub_grid_y) != (1, 1)
        
        # Should have moves available in other sub-boards
        assert len(legal_moves) > 0
        
        # All legal moves should be in empty cells of non-won sub-boards
        for move in legal_moves:
            assert board.board[move.board_y, move.board_x] == Player.EMPTY.value
    
    def test_coordinate_mapping_consistency(self):
        """Test that coordinate mappings are consistent across all positions"""
        for board_id in range(81):
            pos = Position(board_id)
            
            # Test that coordinate calculations are consistent
            expected_board_x = board_id % 9
            expected_board_y = board_id // 9
            
            assert pos.board_x == expected_board_x
            assert pos.board_y == expected_board_y
            
            # Test sub-grid calculations
            expected_sub_grid_x = expected_board_x // 3
            expected_sub_grid_y = expected_board_y // 3
            
            assert pos.sub_grid_x == expected_sub_grid_x
            assert pos.sub_grid_y == expected_sub_grid_y
            
            # Test cell calculations
            expected_cell_x = expected_board_x % 3
            expected_cell_y = expected_board_y % 3
            
            assert pos.cell_x == expected_cell_x
            assert pos.cell_y == expected_cell_y
            
            # Test round-trip conversion
            pos2 = Position(pos.sub_grid_x, pos.sub_grid_y, pos.cell_x, pos.cell_y)
            assert pos.board_id == pos2.board_id
    
    def test_render_output_format(self):
        """Test that render output has correct format"""
        board = UltimateTicTacToeBoard()
        
        # Test empty board render
        rendered = board.render()
        lines = rendered.split('\n')
        
        # Should have 11 lines (3 groups of 3 lines + 2 separator lines)
        assert len(lines) == 11
        
        # Each game line should have specific length
        game_lines = [lines[i] for i in range(11) if i not in [3, 7]]
        for line in game_lines:
            assert len(line) == 21  # 9 cells + 8 spaces + 4 separators
        
        # Separator lines should be dashes
        assert lines[3] == "-" * 23
        assert lines[7] == "-" * 23
        
        # Make some moves and test render still works
        board.make_move(Position(0))   # Top-left, directs play to sub-board (0,0)
        board.make_move(Position(1))   # Legal move in sub-board (0,0)
        board.make_move(Position(10))  # Legal move in sub-board (1,0) directed by previous move
        
        rendered = board.render()
        assert 'X' in rendered
        assert 'O' in rendered
        assert '.' in rendered