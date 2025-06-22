"""
Property-based and boundary value tests for Ultimate Tic-Tac-Toe using Hypothesis.

These tests generate random legal moves and verify consistency properties
that would be difficult to test manually.
"""

import numpy as np
from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from src.utttrlsim.board import UltimateTicTacToeBoard, Player, Position


# Strategy for generating valid board positions (0-80)
board_positions = st.integers(min_value=0, max_value=80)

# Strategy for generating sub-grid coordinates
sub_grid_coords = st.integers(min_value=0, max_value=2)

# Strategy for generating players
players = st.sampled_from([Player.X, Player.O])


class TestPropertyBasedBoard:
    """Property-based tests for UltimateTicTacToeBoard"""

    @given(board_positions)
    def test_position_coordinate_consistency(self, board_id):
        """Test that Position coordinate calculations are consistent"""
        pos = Position(board_id)
        
        # Test round-trip conversion
        pos2 = Position(pos.sub_grid_x, pos.sub_grid_y, pos.cell_x, pos.cell_y)
        assert pos.board_id == pos2.board_id
        
        # Test coordinate bounds
        assert 0 <= pos.board_x <= 8
        assert 0 <= pos.board_y <= 8
        assert 0 <= pos.sub_grid_x <= 2
        assert 0 <= pos.sub_grid_y <= 2
        assert 0 <= pos.cell_x <= 2
        assert 0 <= pos.cell_y <= 2
        
        # Test coordinate relationships
        assert pos.board_x == pos.sub_grid_x * 3 + pos.cell_x
        assert pos.board_y == pos.sub_grid_y * 3 + pos.cell_y
        assert pos.sub_grid_id == pos.sub_grid_x + pos.sub_grid_y * 3

    @given(sub_grid_coords, sub_grid_coords, sub_grid_coords, sub_grid_coords)
    def test_position_from_coordinates(self, grid_x, grid_y, cell_x, cell_y):
        """Test Position initialization from coordinates"""
        pos = Position(grid_x, grid_y, cell_x, cell_y)
        
        assert pos.sub_grid_x == grid_x
        assert pos.sub_grid_y == grid_y
        assert pos.cell_x == cell_x
        assert pos.cell_y == cell_y
        assert 0 <= pos.board_id <= 80

    @given(st.lists(board_positions, min_size=1, max_size=20))
    def test_legal_moves_always_valid(self, move_sequence):
        """Test that legal moves are always valid and making them doesn't break the board"""
        board = UltimateTicTacToeBoard()
        
        for move_id in move_sequence:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Find a legal move from our sequence, or skip
            legal_move = None
            for legal in legal_moves:
                if legal.board_id == move_id:
                    legal_move = legal
                    break
            
            if legal_move is None:
                continue
                
            # Save state before move
            old_state = board.copy()
            
            # Make the move
            board.make_move(legal_move)
            
            # Verify move was made correctly
            assert board.board[legal_move.board_y, legal_move.board_x] != Player.EMPTY.value
            assert board.last_move == legal_move
            
            # Verify board state is still valid
            assert np.all((board.board >= 0) & (board.board <= 2))

    @given(st.lists(board_positions, min_size=2, max_size=10))
    def test_move_alternation_property(self, move_sequence):
        """Test that players alternate correctly"""
        board = UltimateTicTacToeBoard()
        expected_player = Player.X
        
        for move_id in move_sequence:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Find a legal move
            legal_move = None
            for legal in legal_moves:
                if legal.board_id == move_id:
                    legal_move = legal
                    break
            
            if legal_move is None:
                continue
            
            # Check current player before move
            assert board.current_player == expected_player
            
            # Make move
            board.make_move(legal_move)
            
            # Player should have switched
            expected_player = Player.O if expected_player == Player.X else Player.X

    def test_first_move_has_all_positions_legal(self):
        """Test that first move allows all 81 positions"""
        board = UltimateTicTacToeBoard()
        legal_moves = board.get_legal_moves()
        
        assert len(legal_moves) == 81
        board_ids = {move.board_id for move in legal_moves}
        assert board_ids == set(range(81))

    @given(board_positions)
    def test_board_copy_independence(self, first_move_id):
        """Test that board copies are independent"""
        board = UltimateTicTacToeBoard()
        
        # Make a move if it's legal
        legal_moves = board.get_legal_moves()
        first_move = None
        for move in legal_moves:
            if move.board_id == first_move_id:
                first_move = move
                break
        
        if first_move is None:
            return  # Skip if move not legal
            
        board.make_move(first_move)
        
        # Create copy
        board_copy = board.copy()
        
        # Modify original
        legal_moves = board.get_legal_moves()
        if legal_moves:
            board.make_move(legal_moves[0])
            
            # Copy should be unchanged
            assert not np.array_equal(board.board, board_copy.board)
            assert board.current_player != board_copy.current_player
            assert board.last_move != board_copy.last_move

    @given(st.lists(board_positions, min_size=1, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_game_over_consistency(self, move_sequence):
        """Test that game_over and winner properties are consistent"""
        board = UltimateTicTacToeBoard()
        
        for move_id in move_sequence:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Find a legal move
            legal_move = None
            for legal in legal_moves:
                if legal.board_id == move_id:
                    legal_move = legal
                    break
            
            if legal_move is None:
                continue
                
            board.make_move(legal_move)
        
        # Test consistency between game_over and winner
        if board.game_over:
            winner = board.winner
            # Winner should be a valid player or EMPTY (draw)
            assert winner in [Player.X, Player.O, Player.EMPTY]
        else:
            # If game not over, winner should be EMPTY
            assert board.winner == Player.EMPTY


class UltimateTicTacToeStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for Ultimate Tic-Tac-Toe"""
    
    def __init__(self):
        super().__init__()
        self.board = UltimateTicTacToeBoard()
        self.move_history = []
    
    @initialize()
    def initialize_board(self):
        """Initialize with a fresh board"""
        self.board = UltimateTicTacToeBoard()
        self.move_history = []
    
    @rule(move_choice=st.data())
    def make_random_legal_move(self, move_choice):
        """Make a random legal move"""
        if self.board.game_over:
            return
            
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return
            
        # Choose a random legal move using Hypothesis data strategy
        move = move_choice.draw(st.sampled_from(legal_moves))
        old_player = self.board.current_player
        
        self.board.make_move(move)
        self.move_history.append((move, old_player))
    
    @rule()
    def reset_board(self):
        """Reset the board and clear history"""
        self.board.reset()
        self.move_history = []
    
    @invariant()
    def board_state_valid(self):
        """Invariant: Board state is always valid"""
        # All values should be 0, 1, or 2
        assert np.all((self.board.board >= 0) & (self.board.board <= 2))
        
        # Current player should be valid
        assert self.board.current_player in [Player.X, Player.O]
        
        # If we have a last move, it should be a valid position
        if self.board.last_move is not None:
            assert 0 <= self.board.last_move.board_id <= 80
    
    @invariant()
    def legal_moves_are_empty_cells(self):
        """Invariant: All legal moves point to empty cells"""
        if self.board.game_over:
            return
            
        legal_moves = self.board.get_legal_moves()
        for move in legal_moves:
            assert self.board.board[move.board_y, move.board_x] == Player.EMPTY.value
    
    @invariant()
    def move_history_consistency(self):
        """Invariant: Move history should be consistent with board state"""
        if not self.move_history:
            return
            
        # Count moves for each player
        x_moves = sum(1 for _, player in self.move_history if player == Player.X)
        o_moves = sum(1 for _, player in self.move_history if player == Player.O)
        
        # X should have made either the same number or one more move than O
        assert x_moves == o_moves or x_moves == o_moves + 1
        
        # Current player should be correct based on move history
        total_moves = len(self.move_history)
        if total_moves % 2 == 0:
            assert self.board.current_player == Player.X
        else:
            assert self.board.current_player == Player.O


class TestBoundaryValues:
    """Boundary value tests for edge cases"""
    
    def test_corner_positions(self):
        """Test all corner positions of the board"""
        corner_positions = [0, 2, 6, 8, 72, 74, 78, 80]
        
        for pos_id in corner_positions:
            board = UltimateTicTacToeBoard()
            pos = Position(pos_id)
            
            # Should be a legal first move
            legal_moves = board.get_legal_moves()
            assert pos in legal_moves
            
            # Make the move
            board.make_move(pos)
            
            # Verify position mapping
            assert board.board[pos.board_y, pos.board_x] == Player.X.value

    def test_center_positions(self):
        """Test center positions of board and sub-boards"""
        # Global center
        center_pos = Position(40)  # 4,4 in 9x9 board
        
        # Sub-board centers
        sub_centers = []
        for grid_x in range(3):
            for grid_y in range(3):
                sub_centers.append(Position(grid_x, grid_y, 1, 1))  # Center of each sub-board
        
        board = UltimateTicTacToeBoard()
        
        # Test global center
        board.make_move(center_pos)
        assert board.board[4, 4] == Player.X.value
        
        # Test that we're restricted to the correct sub-board
        legal_moves = board.get_legal_moves()
        target_sub_grid = (1, 1)  # Center cell points to center sub-board
        
        for move in legal_moves:
            assert (move.sub_grid_x, move.sub_grid_y) == target_sub_grid

    def test_sub_board_transition_boundaries(self):
        """Test moves that transition between sub-boards"""
        board = UltimateTicTacToeBoard()
        
        # Test each possible cell position and verify it directs to correct sub-board
        test_positions = [
            (Position(0, 0, 0, 0), (0, 0)),  # Top-left cell -> top-left sub-board
            (Position(0, 0, 2, 2), (2, 2)),  # Bottom-right cell -> bottom-right sub-board
            (Position(1, 1, 0, 1), (0, 1)),  # Middle cell -> specific sub-board
        ]
        
        for initial_pos, expected_sub_grid in test_positions:
            board.reset()
            board.make_move(initial_pos)
            
            legal_moves = board.get_legal_moves()
            if legal_moves:  # If target sub-board isn't won/full
                for move in legal_moves:
                    assert (move.sub_grid_x, move.sub_grid_y) == expected_sub_grid

    def test_full_sub_board_behavior(self):
        """Test behavior when sub-boards become full"""
        board = UltimateTicTacToeBoard()
        
        # Fill sub-board (0,0) completely without creating a winner
        # This requires careful placement to avoid 3-in-a-row
        fill_pattern = [
            (0, 0, 0, 0, Player.X),  # Top-left
            (0, 0, 1, 0, Player.O),  # Top-center  
            (0, 0, 2, 0, Player.X),  # Top-right
            (0, 0, 0, 1, Player.O),  # Middle-left
            (0, 0, 1, 1, Player.X),  # Center
            (0, 0, 2, 1, Player.O),  # Middle-right
            (0, 0, 0, 2, Player.X),  # Bottom-left
            (0, 0, 1, 2, Player.O),  # Bottom-center
            (0, 0, 2, 2, Player.X),  # Bottom-right
        ]
        
        # Set up this pattern directly
        for grid_x, grid_y, cell_x, cell_y, player in fill_pattern:
            pos = Position(grid_x, grid_y, cell_x, cell_y)
            board.board[pos.board_y, pos.board_x] = player.value
        
        # Set last move to direct play to sub-board (0,0)
        board.last_move = Position(0, 0, 0, 0)  # This would direct to (0,0)
        board.current_player = Player.O
        
        # Legal moves should not include sub-board (0,0) since it's full
        legal_moves = board.get_legal_moves()
        for move in legal_moves:
            assert (move.sub_grid_x, move.sub_grid_y) != (0, 0)

    def test_maximum_game_length(self):
        """Test behavior in very long games"""
        board = UltimateTicTacToeBoard()
        move_count = 0
        max_moves = 81  # Theoretical maximum
        
        # Play random legal moves until game ends or max moves reached
        while not board.game_over and move_count < max_moves:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Make first legal move
            board.make_move(legal_moves[0])
            move_count += 1
        
        # Game should either be over or board should be full
        assert board.game_over or move_count == max_moves
        
        # If game is over, winner should be valid
        if board.game_over:
            assert board.winner in [Player.X, Player.O, Player.EMPTY]


# Test runner for the state machine
TestUltimateTicTacToeStateMachine = UltimateTicTacToeStateMachine.TestCase


class TestDifferentialConsistency:
    """Tests for differential consistency - comparing different implementations or states"""
    
    @given(st.lists(board_positions, min_size=1, max_size=20))
    def test_board_state_consistency(self, move_sequence):
        """Test that board state is consistent with move history"""
        board = UltimateTicTacToeBoard()
        moves_made = []
        
        for move_id in move_sequence:
            if board.game_over:
                break
                
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            # Find a legal move
            legal_move = None
            for legal in legal_moves:
                if legal.board_id == move_id:
                    legal_move = legal
                    break
                    
            if legal_move is None:
                continue
                
            old_player = board.current_player
            board.make_move(legal_move)
            moves_made.append((legal_move, old_player))
        
        # Verify that each move is reflected in the board state
        for move, player in moves_made:
            assert board.board[move.board_y, move.board_x] == player.value
    
    @given(st.lists(board_positions, min_size=2, max_size=15))
    def test_copy_vs_rebuild_consistency(self, move_sequence):
        """Test that copying a board gives same result as rebuilding it"""
        original_board = UltimateTicTacToeBoard()
        moves_made = []
        
        # Play sequence on original board
        for move_id in move_sequence:
            if original_board.game_over:
                break
                
            legal_moves = original_board.get_legal_moves()
            if not legal_moves:
                break
                
            legal_move = None
            for legal in legal_moves:
                if legal.board_id == move_id:
                    legal_move = legal
                    break
                    
            if legal_move is None:
                continue
                
            moves_made.append((legal_move, original_board.current_player))
            original_board.make_move(legal_move)
        
        # Create copy
        copied_board = original_board.copy()
        
        # Rebuild board from scratch with same moves
        rebuilt_board = UltimateTicTacToeBoard()
        for move, _ in moves_made:
            if not rebuilt_board.game_over:
                legal_moves = rebuilt_board.get_legal_moves()
                if move in legal_moves:
                    rebuilt_board.make_move(move)
        
        # All three boards should have same state
        assert np.array_equal(original_board.board, copied_board.board)
        assert np.array_equal(original_board.board, rebuilt_board.board)
        assert original_board.current_player == copied_board.current_player
        assert original_board.game_over == copied_board.game_over
        assert original_board.winner == copied_board.winner