import pytest
import numpy as np
from utttrlsim.board import UltimateTicTacToeBoard, Player, Position

class TestUltimateTicTacToeBoard:
    """Test cases for UltimateTicTacToeBoard"""

    @pytest.fixture
    def board(self):
        """Fixture to provide a fresh UltimateTicTacToeBoard instance for each test"""
        return UltimateTicTacToeBoard()

    # Helper functions for test data generation
    def fill_sub_board(self, board_state: np.ndarray, grid_x: int, grid_y: int, pattern: list):
        """
        Fill a sub-board with a specific pattern.
        
        Args:
            board_state: The 9x9 board state array to modify
            grid_x, grid_y: Sub-grid coordinates (0-2)  
            pattern: List of 9 values to fill the sub-board (row-wise)
        """
        for i in range(3):
            for j in range(3):
                board_state[grid_y * 3 + i, grid_x * 3 + j] = pattern[i * 3 + j]

    def create_sub_board_win(self, board_state: np.ndarray, grid_x: int, grid_y: int, player: Player, win_type: str = "row"):
        """
        Create a winning pattern in a sub-board.
        
        Args:
            board_state: The 9x9 board state array to modify
            grid_x, grid_y: Sub-grid coordinates (0-2)
            player: Player who wins this sub-board
            win_type: Type of win - "row", "col", "diag1", "diag2"
        """
        base_y = grid_y * 3
        base_x = grid_x * 3
        
        if win_type == "row":
            # Top row win
            for j in range(3):
                board_state[base_y, base_x + j] = player.value
        elif win_type == "col":
            # Left column win
            for i in range(3):
                board_state[base_y + i, base_x] = player.value
        elif win_type == "diag1":
            # Main diagonal win
            for i in range(3):
                board_state[base_y + i, base_x + i] = player.value
        elif win_type == "diag2":
            # Anti-diagonal win
            for i in range(3):
                board_state[base_y + i, base_x + (2 - i)] = player.value

    def setup_board_state(self, board: UltimateTicTacToeBoard, board_state: np.ndarray, 
                         current_player: Player, last_move: Position):
        """
        Set up the board with a specific state.
        
        Args:
            board: Board instance to modify
            board_state: The 9x9 board state array
            current_player: Current player
            last_move: Last move made
        """
        board.board = board_state.copy()
        board.current_player = current_player
        board.last_move = last_move

    def create_empty_board_state(self) -> np.ndarray:
        """Create an empty 9x9 board state."""
        return np.zeros((9, 9), dtype=np.int8)

    def test_initialization(self, board):
        """Test board initialization"""
        assert board.board.shape == (9, 9)
        assert board.subboard_winner.shape == (3, 3)
        np.testing.assert_array_equal(board.board, 0, err_msg="初期化後のboard配列はすべて0であるべき")
        np.testing.assert_array_equal(board.subboard_winner, 0, err_msg="初期化後のsubboard_winner配列はすべて0であるべき")
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
        np.testing.assert_array_equal(board.board, 0, err_msg="リセット後のboard配列はすべて0であるべき")
        np.testing.assert_array_equal(board.subboard_winner, 0, err_msg="リセット後のsubboard_winner配列はすべて0であるべき")
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
        np.testing.assert_equal(
            board.board[position.board_y, position.board_x], 
            Player.X.value,
            err_msg=f"position ({position.board_y}, {position.board_x})の値はPlayer.X.value({Player.X.value})であるべき"
        )
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

    def test_sub_board_win(self, board):
        """Test sub-board win detection"""
        # Arrange
        # Set up a situation where sub-board 0 has been won by X
        board_state = self.create_empty_board_state()
        # Create X win in top row of sub-board (0,0)
        self.create_sub_board_win(board_state, 0, 0, Player.X, "row")
        # Add some O moves in middle row of sub-board (0,0)
        board_state[1, 0] = Player.O.value  # Position(9)
        board_state[1, 1] = Player.O.value  # Position(10)
        board_state[1, 2] = Player.O.value  # Position(11)

        # Act
        # Set the board state directly
        self.setup_board_state(board, board_state, Player.O, Position(2))

        # Assert
        # Sub-board 0 should be won by X
        np.testing.assert_equal(
            board.subboard_winner[0, 0], 
            Player.X.value,
            err_msg=f"subboard_winner[0, 0]はPlayer.X.value({Player.X.value})であるべき - サブボード0はXが勝利"
        )

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
        board_state = self.create_empty_board_state()
        
        # Create winning patterns in sub-boards 0, 1, 2 (top row of meta-board)
        for sub_grid_x in range(3):
            self.create_sub_board_win(board_state, sub_grid_x, 0, Player.X, "row")

        # Act
        # Set the board state directly
        self.setup_board_state(board, board_state, Player.O, Position(2))

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
        np.testing.assert_array_equal(board.board, board_copy.board, err_msg="コピー直後はboard配列が同じであるべき")
        np.testing.assert_array_equal(board.subboard_winner, board_copy.subboard_winner, err_msg="コピー直後はsubboard_winner配列が同じであるべき")

        # Make a move on the original
        legal_moves = board.get_legal_moves()
        if legal_moves:
            board.make_move(legal_moves[0])

            # Copy should be unchanged - arrays should NOT be equal after modifying original
            try:
                np.testing.assert_array_equal(board.board, board_copy.board)
                assert False, "元のボードに変更後、コピーは変更されないべき - 配列が等しくないはず"
            except AssertionError:
                # Expected: arrays should not be equal after modifying original
                # This means the copy is independent, which is what we want
                pass

            # Verify that the copy is indeed unchanged by checking a specific position
            # The original board should have the new move, but the copy should not
            assert board.board[legal_moves[0].board_y, legal_moves[0].board_x] != 0, "元のボードには新しい着手があるべき"
            assert board_copy.board[legal_moves[0].board_y, legal_moves[0].board_x] == 0, "コピーされたボードには新しい着手がないべき"

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
        np.testing.assert_equal(
            board.board[4, 4], 
            Player.X.value,
            err_msg=f"board[4, 4]はPlayer.X.value({Player.X.value})であるべき - position 40の着手後"
        )

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
        board_state = self.create_empty_board_state()
        # Fill sub-board 0 (positions 0-8) alternating X and O
        pattern = [Player.X.value, Player.O.value, Player.X.value,
                  Player.O.value, Player.X.value, Player.O.value,
                  Player.X.value, Player.O.value, Player.X.value]
        
        self.fill_sub_board(board_state, 0, 0, pattern)

        # Act
        # Set the board state directly
        self.setup_board_state(board, board_state, Player.O, Position(8))

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
        board_state = self.create_empty_board_state()
        board_state[0, 0] = Player.X.value  # Position(0)
        board_state[0, 1] = Player.O.value  # Position(1)

        # Act
        # Set the board state with last move directing play to sub-board 1
        # Position(1) has cell coordinates (1, 0), so next play goes to sub-grid (1, 0)
        self.setup_board_state(board, board_state, Player.X, Position(1))

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
