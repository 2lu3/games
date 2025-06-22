import numpy as np
from src.utttrlsim.board import UltimateTicTacToeBoard, Player, Position

class TestUltimateTicTacToeBoard:
    """Test cases for UltimateTicTacToeBoard"""

    def test_initialization(self):
        """Test board initialization"""
        board = UltimateTicTacToeBoard()

        assert board.board.shape == (9, 9)
        assert board.subboard_winner.shape == (3, 3)
        assert np.all(board.board == 0)
        assert np.all(board.subboard_winner == 0)
        assert board.current_player == Player.X
        assert board.last_move is None
        assert not board.game_over
        assert board.winner == Player.EMPTY

    def test_reset(self):
        """Test board reset"""
        board = UltimateTicTacToeBoard()

        # Make some moves
        board.make_move(Position(40))  # board_id 40 = (4, 4) in board coordinates
        board.make_move(Position(0))   # board_id 0 = (0, 0) in board coordinates

        # Reset
        board.reset()

        assert np.all(board.board == 0)
        assert np.all(board.subboard_winner == 0)
        assert board.current_player == Player.X
        assert board.last_move is None
        assert not board.game_over
        assert board.winner == Player.EMPTY

    def test_legal_moves_first_move(self):
        """Test legal moves for first move"""
        board = UltimateTicTacToeBoard()
        legal_moves = board.get_legal_moves()

        # First move should have 81 legal moves
        assert len(legal_moves) == 81

        # All moves should be valid positions from 0 to 80
        board_ids = [pos.board_id for pos in legal_moves]
        assert set(board_ids) == set(range(81))

    def test_legal_moves_after_move(self):
        """Test legal moves after making a move"""
        board = UltimateTicTacToeBoard()

        # Make a move at board position 40 (center of the board)
        # This corresponds to sub_grid (1,1), cell (1,1)
        board.make_move(Position(40))

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

    def test_make_move(self):
        """Test making moves"""
        board = UltimateTicTacToeBoard()

        # Make a valid move
        position = Position(40)  # board_id 40
        success = board.make_move(position)
        assert success

        # Check that the move was made
        np.testing.assert_equal(board.board[position.board_y, position.board_x], Player.X.value)
        assert board.last_move == position
        assert board.current_player == Player.O

    def test_make_invalid_move(self):
        """Test making invalid moves"""
        board = UltimateTicTacToeBoard()

        # Make a move
        position = Position(40)
        board.make_move(position)

        # Try to make the same move again
        success = board.make_move(position)
        assert not success

    def test_sub_board_win(self):
        """Test sub-board win detection"""
        board = UltimateTicTacToeBoard()

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

        # Set the board state
        board.set_board_state(board_state, Player.O, Position(2))

        # Sub-board 0 should be won by X
        np.testing.assert_equal(board.subboard_winner[0, 0], Player.X.value)

        # Since sub-board 0 is won, next moves can be played anywhere except sub-board 0
        legal_moves = board.get_legal_moves()
        # Check that moves can be made in other sub-boards
        sub_grids_with_moves = set()
        for pos in legal_moves:
            sub_grids_with_moves.add((pos.sub_grid_x, pos.sub_grid_y))
        
        # Should not be able to play in sub-board 0
        assert (0, 0) not in sub_grids_with_moves

    def test_game_win(self):
        """Test game win detection by controlling meta-board"""
        board = UltimateTicTacToeBoard()

        # Create a situation where X wins the top row of meta-board
        # Set sub-boards (0,0), (1,0), (2,0) as won by X
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # Create winning patterns in sub-boards 0, 1, 2
        for sub_grid_x in range(3):
            start_x = sub_grid_x * 3
            # Make top row of each sub-board won by X
            for i in range(3):
                board_state[0, start_x + i] = Player.X.value

        # Set the board state
        board.set_board_state(board_state, Player.O, Position(2))

        assert board.game_over
        assert board.winner == Player.X

    def test_draw_detection(self):
        """Test draw detection"""
        board = UltimateTicTacToeBoard()

        # Basic test - empty board should not be game over
        assert not board.game_over

    def test_render(self):
        """Test board rendering"""
        board = UltimateTicTacToeBoard()

        # Make some moves
        board.make_move(Position(40))  # Center position
        board.make_move(Position(0))   # Top-left position

        rendered = board.render()

        # Check that render produces a string
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_copy(self):
        """Test board copying"""
        board = UltimateTicTacToeBoard()

        # Make some moves
        board.make_move(Position(40))
        board.make_move(Position(0))

        # Copy the board
        board_copy = board.copy()

        # Check that copy is independent
        assert np.array_equal(board.board, board_copy.board)
        assert np.array_equal(board.subboard_winner, board_copy.subboard_winner)

        # Make a move on the original
        legal_moves = board.get_legal_moves()
        if legal_moves:
            board.make_move(legal_moves[0])

            # Copy should be unchanged
            assert not np.array_equal(board.board, board_copy.board)

    def test_debug_coordinates(self):
        """Debug test to understand coordinate mapping"""
        board = UltimateTicTacToeBoard()

        # Make a move at position 40 (center of board)
        position = Position(40)
        board.make_move(position)

        # Print board state to understand coordinate mapping
        print("Board after move at position 40:")
        print(board.board)
        print("Meta board:")
        print(board.subboard_winner)
        print("Last move:", board.last_move)
        print("Current player:", board.current_player)

        # Check what position was actually filled
        # Position 40 should be at board coordinates (4, 4)
        np.testing.assert_equal(board.board[4, 4], Player.X.value)

        # Check legal moves after first move
        legal_moves = board.get_legal_moves()
        print("Legal moves after first move:", [pos.board_id for pos in legal_moves])
        print("Number of legal moves:", len(legal_moves))

        # Make another valid move
        if legal_moves:
            next_position = legal_moves[0]
            success = board.make_move(next_position)
            print("Success making next move:", success)

            if success:
                print("Board after second move:")
                print(board.board)

    def test_sub_board_full(self):
        """Test that when a sub-board is full, next move can be anywhere"""
        board = UltimateTicTacToeBoard()

        # Fill sub-board 0 completely without creating a winner
        board_state = np.zeros((9, 9), dtype=np.int8)
        # Fill sub-board 0 (positions 0-8) alternating X and O
        pattern = [Player.X.value, Player.O.value, Player.X.value,
                  Player.O.value, Player.X.value, Player.O.value,
                  Player.X.value, Player.O.value, Player.X.value]
        
        for i in range(3):
            for j in range(3):
                board_state[i, j] = pattern[i * 3 + j]

        # Set the board state
        board.set_board_state(board_state, Player.O, Position(8))

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

    def test_legal_moves_restricted(self):
        """Test that legal moves are restricted to target sub-board when it's not full/won"""
        board = UltimateTicTacToeBoard()

        # Set up a situation where sub-board 0 has some moves but is not full/won
        board_state = np.zeros((9, 9), dtype=np.int8)
        board_state[0, 0] = Player.X.value  # Position(0)
        board_state[0, 1] = Player.O.value  # Position(1)

        # Set the board state with last move directing play to sub-board 1
        # Position(1) has cell coordinates (1, 0), so next play goes to sub-grid (1, 0)
        board.set_board_state(board_state, Player.X, Position(1))

        # Next moves should be restricted to sub-board at (1, 0)
        legal_moves = board.get_legal_moves()
        
        # All legal moves should be in sub-grid (1, 0)
        for pos in legal_moves:
            assert pos.sub_grid_x == 1
            assert pos.sub_grid_y == 0
        
        # Should have 9 moves available in the target sub-board (all empty)
        assert len(legal_moves) == 9

    def test_position_class(self):
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
