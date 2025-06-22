import pytest
import numpy as np
from utttrlsim.board import UltimateTicTacToeBoard, Player, Position

class TestUltimateTicTacToeBoard:
    """Test cases for UltimateTicTacToeBoard"""

    @pytest.fixture
    def board(self):
        """Fixture to provide a fresh UltimateTicTacToeBoard instance for each test"""
        return UltimateTicTacToeBoard()

    def test_initialization(self, board):
        """Test board initialization"""
        assert board.board.shape == (9, 9)
        assert board.subboard_winner.shape == (3, 3)
        assert np.all(board.board == 0)
        assert np.all(board.subboard_winner == 0)
        assert board.current_player == Player.X
        assert board.last_move is None
        assert not board.game_over
        assert board.winner == Player.EMPTY

    def test_reset(self, board):
        """Test board reset"""
        # Arrange
        # Make some moves to change the board state
        board.make_move(Position(40))  # board_id 40 = (4, 4) in board coordinates
        board.make_move(Position(30))   # board_id 30 = (3, 3) in sub-board (1,1)

        # Act
        # Reset the board
        board.reset()

        # Assert
        assert np.all(board.board == 0)
        assert np.all(board.subboard_winner == 0)
        assert board.current_player == Player.X
        assert board.last_move is None
        assert not board.game_over
        assert board.winner == Player.EMPTY

    def test_legal_moves_first_move(self, board):
        """Test legal moves for first move"""
        legal_moves = board.get_legal_moves()

        # First move should have 81 legal moves
        assert len(legal_moves) == 81

        # All moves should be valid positions from 0 to 80
        board_ids = [pos.board_id for pos in legal_moves]
        assert set(board_ids) == set(range(81))

    def test_legal_moves_after_move(self, board):
        """Test legal moves after making a move"""
        # Arrange
        # (No specific arrangement needed - using fresh board from fixture)
        
        # Act
        # Make a move at board position 40 (center of the board)
        # This corresponds to sub_grid (1,1), cell (1,1)
        board.make_move(Position(40))

        # Assert
        # Next move should be restricted to sub-board corresponding to cell (1,1)
        # which is sub-board at (1,1)
        legal_moves = board.get_legal_moves()

        # Should have 8 legal moves in the target sub-board
        assert len(legal_moves) == 8

        # All legal moves should be in the same sub-grid as the last move's cell position
        target_sub_grid_x = 1  # cell_x of position 40
        target_sub_grid_y = 1  # cell_y of position 40
        
        for pos in legal_moves:
            assert pos.sub_grid_x == target_sub_grid_x
            assert pos.sub_grid_y == target_sub_grid_y
            assert pos.board_id != 40  # Position 40 is already taken

    def test_make_move(self, board):
        """Test making moves"""
        # Arrange
        # Set up the position to make a move
        position = Position(40)  # board_id 40
        
        # Act
        # Make a valid move
        board.make_move(position)

        # Assert
        # Check that the move was made
        assert board.board[position.board_y, position.board_x] == Player.X.value
        assert board.last_move == position
        assert board.current_player == Player.O

    def test_make_invalid_move(self, board):
        """Test making invalid moves"""
        # Arrange
        # Set up a board state with one move already made
        position = Position(40)
        board.make_move(position)

        # Act & Assert
        # Try to make the same move again - should raise ValueError
        with pytest.raises(ValueError):
            board.make_move(position)

    def test_make_move_wrong_sub_board(self, board):
        """Test making a move on wrong sub-board when target is specified"""
        # Arrange
        # Make first move at position 40 (center)
        # This forces next move to be in sub-board (1,1) 
        first_position = Position(40)  # center, cell (1,1)
        board.make_move(first_position)
        
        # Act & Assert
        # Try to make a move in a different sub-board (e.g., sub-board (0,0))
        # Position 0 is in sub-board (0,0), which should be invalid
        wrong_sub_board_position = Position(0)  # sub-board (0,0)
        with pytest.raises(ValueError):
            board.make_move(wrong_sub_board_position)

    def test_make_move_after_game_over(self, board):
        """Test making a move after the game has ended"""
        # Arrange
        # Create a winning condition for X by controlling the meta-board
        # Set up X to win the top row of meta-board: sub-boards (0,0), (1,0), (2,0)
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # Create winning patterns in sub-boards 0, 1, 2 (top row of meta-board)
        for sub_grid_x in range(3):
            start_x = sub_grid_x * 3
            # Make top row of each sub-board won by X
            for i in range(3):
                board_state[0, start_x + i] = Player.X.value
        
        # Set the board state to create a game-over condition
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(2)
        
        # Verify game is over
        assert board.game_over
        assert board.winner == Player.X
        
        # Act & Assert
        # Try to make a move after game is over - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot make move: game is already over"):
            board.make_move(Position(10))

    def test_make_move_on_won_sub_board(self, board):
        """Test making a move on a sub-board that's already won"""
        # Arrange
        # Create a state where sub-board (0,0) is won by X
        board_state = np.zeros((9, 9), dtype=np.int8)
        # Win sub-board (0,0) with X on top row
        board_state[0, 0] = Player.X.value  # Position(0)
        board_state[0, 1] = Player.X.value  # Position(1)
        board_state[0, 2] = Player.X.value  # Position(2)
        
        # Set last move to direct play to the won sub-board (0,0)
        # Position 9 has cell coordinates (0,0), directing to sub-board (0,0)
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(9)  # cell (0,0) -> target sub-board (0,0)
        
        # Act & Assert
        # Try to make a move in the won sub-board - should raise ValueError
        # Since sub-board (0,0) is won, moves there should be invalid
        with pytest.raises(ValueError):
            board.make_move(Position(3))  # Position 3 is in sub-board (0,0)

    def test_make_move_on_full_sub_board(self, board):
        """Test making a move on a sub-board that's full but not won"""
        # Arrange
        # Fill sub-board (0,0) completely without creating a winner
        board_state = np.zeros((9, 9), dtype=np.int8)
        # Fill sub-board (0,0) alternating X and O to avoid wins
        pattern = [Player.X.value, Player.O.value, Player.X.value,
                  Player.O.value, Player.X.value, Player.O.value,
                  Player.X.value, Player.O.value, Player.X.value]
        
        for i in range(3):
            for j in range(3):
                board_state[i, j] = pattern[i * 3 + j]
        
        # Set last move to direct play to the full sub-board (0,0)
        # Position 9 has cell coordinates (0,0), directing to sub-board (0,0)
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(9)  # cell (0,0) -> target sub-board (0,0)
        
        # Act & Assert
        # Try to make a move in the full sub-board - should raise ValueError
        # Since sub-board (0,0) is full, moves there should be invalid
        with pytest.raises(ValueError):
            board.make_move(Position(0))  # Position 0 is in sub-board (0,0)

    def test_make_move_invalid_position_id(self, board):
        """Test making a move with invalid position ID"""
        # Act & Assert
        # Try to create positions with invalid IDs
        with pytest.raises(AssertionError):
            Position(-1)  # Negative position ID
            
        with pytest.raises(AssertionError):
            Position(81)  # Position ID too large (max is 80)

    def test_make_move_invalid_grid_coordinates(self, board):
        """Test making a move with invalid grid coordinates"""
        # Act & Assert
        # Try to create positions with invalid grid coordinates
        with pytest.raises(AssertionError):
            Position(-1, 0, 0, 0)  # Invalid grid_x
            
        with pytest.raises(AssertionError):
            Position(0, -1, 0, 0)  # Invalid grid_y
            
        with pytest.raises(AssertionError):
            Position(0, 0, -1, 0)  # Invalid cell_x
            
        with pytest.raises(AssertionError):
            Position(0, 0, 0, -1)  # Invalid cell_y
            
        with pytest.raises(AssertionError):
            Position(3, 0, 0, 0)  # grid_x too large
            
        with pytest.raises(AssertionError):
            Position(0, 3, 0, 0)  # grid_y too large
            
        with pytest.raises(AssertionError):
            Position(0, 0, 3, 0)  # cell_x too large
            
        with pytest.raises(AssertionError):
            Position(0, 0, 0, 3)  # cell_y too large

    def test_make_move_edge_case_all_sub_boards_won_or_full(self, board):
        """Test behavior when all sub-boards are won or full"""
        # Arrange
        # Create a state where all sub-boards are either won or full
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # Win some sub-boards and fill others
        # Sub-board (0,0): Won by X (top row)
        board_state[0, 0:3] = Player.X.value
        
        # Sub-board (1,0): Won by O (left column)
        board_state[0:3, 3] = Player.O.value
        
        # Sub-board (2,0): Won by X (diagonal)
        board_state[0, 6] = Player.X.value
        board_state[1, 7] = Player.X.value
        board_state[2, 8] = Player.X.value
        
        # Fill remaining sub-boards without winners
        for grid_y in range(3):
            for grid_x in range(3):
                if (grid_x, grid_y) not in [(0, 0), (1, 0), (2, 0)]:
                    start_y = grid_y * 3
                    start_x = grid_x * 3
                    # Fill with alternating pattern to avoid wins
                    pattern = [Player.X.value, Player.O.value, Player.X.value,
                              Player.O.value, Player.X.value, Player.O.value,
                              Player.X.value, Player.O.value, Player.X.value]
                    for i in range(3):
                        for j in range(3):
                            board_state[start_y + i, start_x + j] = pattern[i * 3 + j]
        
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(10)
        
        # Act & Assert
        # Game should be over, so any move should raise RuntimeError
        assert board.game_over
        with pytest.raises(RuntimeError, match="Cannot make move: game is already over"):
            board.make_move(Position(50))

    def test_make_move_invalid_argument_types(self, board):
        """Test making a move with invalid argument types"""
        # Act & Assert
        # Try to make moves with invalid types
        with pytest.raises((TypeError, ValueError)):
            board.make_move("invalid")  # String instead of Position
            
        with pytest.raises((TypeError, ValueError)):
            board.make_move(42)  # Integer instead of Position
            
        with pytest.raises((TypeError, ValueError)):
            board.make_move(None)  # None instead of Position

    def test_sub_board_win(self, board):
        """Test sub-board win detection"""
        # Arrange
        # Set up a situation where sub-board 0 has been won by X
        # Sub-board 0 top row: X X X
        # Sub-board 0 middle row: O O O  
        # Sub-board 0 bottom row: . . .
        board_state = np.zeros((9, 9), dtype=np.int8)
        # Sub-board 0 top row positions: 0, 1, 2
        board_state[0, 0] = Player.X.value  # Position(0)
        board_state[0, 1] = Player.X.value  # Position(1)
        board_state[0, 2] = Player.X.value  # Position(2)
        # Sub-board 0 middle row positions: 9, 10, 11  
        board_state[1, 0] = Player.O.value  # Position(9)
        board_state[1, 1] = Player.O.value  # Position(10)
        board_state[1, 2] = Player.O.value  # Position(11)

        # Act
        # Set the board state directly
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(2)

        # Assert
        # Sub-board 0 should be won by X
        assert board.subboard_winner[0, 0] == Player.X.value

        # Since sub-board 0 is won, next moves can be played anywhere except sub-board 0
        legal_moves = board.get_legal_moves()
        # Check that moves can be made in other sub-boards
        sub_grids_with_moves = set()
        for pos in legal_moves:
            sub_grids_with_moves.add((pos.sub_grid_x, pos.sub_grid_y))
        
        # Should not be able to play in sub-board 0
        assert (0, 0) not in sub_grids_with_moves

    def test_game_win(self, board):
        """Test game win detection by controlling meta-board"""
        # Arrange
        # Create a situation where X wins the top row of meta-board
        # Set sub-boards (0,0), (1,0), (2,0) as won by X
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # Create winning patterns in sub-boards 0, 1, 2
        for sub_grid_x in range(3):
            start_x = sub_grid_x * 3
            # Make top row of each sub-board won by X
            for i in range(3):
                board_state[0, start_x + i] = Player.X.value

        # Act
        # Set the board state directly
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(2)

        # Assert
        assert board.game_over
        assert board.winner == Player.X

    def test_draw_detection(self, board):
        """Test draw detection"""
        # Basic test - empty board should not be game over
        assert not board.game_over

    def test_render(self, board):
        """Test board rendering"""
        # Arrange
        # Make some moves to create a testable board state
        board.make_move(Position(40))  # Center position
        board.make_move(Position(30))   # Valid position in sub-board (1,1)

        # Act
        rendered = board.render()

        # Assert
        # Check that render produces a string
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_copy(self, board):
        """Test board copying"""
        # Arrange
        # Make some moves to create a testable board state
        board.make_move(Position(40))
        board.make_move(Position(30))

        # Act
        # Copy the board
        board_copy = board.copy()

        # Assert
        # Check that copy is independent
        assert np.array_equal(board.board, board_copy.board)
        assert np.array_equal(board.subboard_winner, board_copy.subboard_winner)

        # Make a move on the original
        legal_moves = board.get_legal_moves()
        if legal_moves:
            board.make_move(legal_moves[0])

            # Copy should be unchanged
            assert not np.array_equal(board.board, board_copy.board)

    def test_debug_coordinates(self, board):
        """Debug test to understand coordinate mapping"""
        # Arrange
        # Set up a position to test
        position = Position(40)
        
        # Act
        # Make a move at position 40 (center of board)
        board.make_move(position)

        # Assert
        # Print board state to understand coordinate mapping
        print("Board after move at position 40:")
        print(board.board)
        print("Meta board:")
        print(board.subboard_winner)
        print("Last move:", board.last_move)
        print("Current player:", board.current_player)

        # Check what position was actually filled
        # Position 40 should be at board coordinates (4, 4)
        assert board.board[4, 4] == Player.X.value

        # Check legal moves after first move
        legal_moves = board.get_legal_moves()
        print("Legal moves after first move:", [pos.board_id for pos in legal_moves])
        print("Number of legal moves:", len(legal_moves))

        # Make another valid move
        if legal_moves:
            next_position = legal_moves[0]
            board.make_move(next_position)
            print("Board after second move:")
            print(board.board)

    def test_sub_board_full(self, board):
        """Test that when a sub-board is full, next move can be anywhere"""
        # Arrange
        # Fill sub-board 0 completely without creating a winner
        board_state = np.zeros((9, 9), dtype=np.int8)
        # Fill sub-board 0 (positions 0-8) alternating X and O
        pattern = [Player.X.value, Player.O.value, Player.X.value,
                  Player.O.value, Player.X.value, Player.O.value,
                  Player.X.value, Player.O.value, Player.X.value]
        
        for i in range(3):
            for j in range(3):
                board_state[i, j] = pattern[i * 3 + j]

        # Act
        # Set the board state directly
        board.board = board_state
        board.current_player = Player.O
        board.last_move = Position(8)

        # Assert
        # Sub-board 0 is full but not won, so next moves can be anywhere except sub-board 0
        legal_moves = board.get_legal_moves()
        
        # Check that moves can be made in other sub-boards
        sub_grids_with_moves = set()
        for pos in legal_moves:
            sub_grids_with_moves.add((pos.sub_grid_x, pos.sub_grid_y))
        
        # Should not be able to play in sub-board 0 (it's full)
        assert (0, 0) not in sub_grids_with_moves
        # Should be able to play in other sub-boards
        assert len(sub_grids_with_moves) > 0

    def test_legal_moves_restricted(self, board):
        """Test that legal moves are restricted to target sub-board when it's not full/won"""
        # Arrange
        # Set up a situation where sub-board 0 has some moves but is not full/won
        board_state = np.zeros((9, 9), dtype=np.int8)
        board_state[0, 0] = Player.X.value  # Position(0)
        board_state[0, 1] = Player.O.value  # Position(1)

        # Act
        # Set the board state with last move directing play to sub-board 1
        # Position(1) has cell coordinates (1, 0), so next play goes to sub-grid (1, 0)
        board.board = board_state
        board.current_player = Player.X
        board.last_move = Position(1)

        # Assert
        # Next moves should be restricted to sub-board at (1, 0)
        legal_moves = board.get_legal_moves()
        
        # All legal moves should be in sub-grid (1, 0)
        for pos in legal_moves:
            assert pos.sub_grid_x == 1
            assert pos.sub_grid_y == 0
        
        # Should have 9 moves available in the target sub-board (all empty)
        assert len(legal_moves) == 9

    def test_position_class(self, board):
        """Test Position class functionality"""
        # Test initialization with board_id
        pos1 = Position(40)
        assert pos1.board_id == 40
        assert pos1.board_x == 4
        assert pos1.board_y == 4
        assert pos1.sub_grid_x == 1
        assert pos1.sub_grid_y == 1
        assert pos1.cell_x == 1
        assert pos1.cell_y == 1
        
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
