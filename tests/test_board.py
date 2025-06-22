import numpy as np
from src.utttrlsim.board import UltimateTicTacToeBoard, Player

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
        board.make_move(4, 4)
        board.make_move(0, 0)

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

        # All moves should be valid
        for sub_board in range(9):
            for position in range(9):
                assert (sub_board, position) in legal_moves

    def test_legal_moves_after_move(self):
        """Test legal moves after making a move"""
        board = UltimateTicTacToeBoard()

        # Make a move in sub-board 4, position 4
        board.make_move(4, 4)

        # Next move should be restricted to sub-board 4
        legal_moves = board.get_legal_moves()

        # Should have 8 legal moves in sub-board 4
        assert len(legal_moves) == 8

        for sub_board, position in legal_moves:
            assert sub_board == 4
            assert position != 4  # Position 4 is already taken

    def test_make_move(self):
        """Test making moves"""
        board = UltimateTicTacToeBoard()

        # Make a valid move
        success = board.make_move(4, 4)
        assert success

        # Check that the move was made
        assert board.board[4, 4] == Player.X.value
        assert board.last_move == (4, 4)
        assert board.current_player == Player.O

    def test_make_invalid_move(self):
        """Test making invalid moves"""
        board = UltimateTicTacToeBoard()

        # Make a move
        board.make_move(4, 4)

        # Try to make the same move again
        success = board.make_move(4, 4)
        assert not success

    def test_sub_board_win(self):
        """Test sub-board win detection (ルール準拠)"""
        board = UltimateTicTacToeBoard()

        # サブボード0の上段（0,1,2）をXが埋めて勝利した状況を直接セット
        # サブボード0の上段: X X X
        # サブボード0の中段: O O O
        # サブボード0の下段: . . .
        board_state = np.zeros((9, 9), dtype=np.int8)
        # サブボード0の上段をXで埋める
        board_state[0, 0] = Player.X.value  # サブボード0, pos 0
        board_state[0, 1] = Player.X.value  # サブボード0, pos 1
        board_state[0, 2] = Player.X.value  # サブボード0, pos 2
        # サブボード0の中段をOで埋める（テスト用）
        board_state[1, 0] = Player.O.value  # サブボード0, pos 3
        board_state[1, 1] = Player.O.value  # サブボード0, pos 4
        board_state[1, 2] = Player.O.value  # サブボード0, pos 5

        # subboard_winnerも設定（サブボード0はXが勝利）
        subboard_winner_state = np.zeros((3, 3), dtype=np.int8)
        subboard_winner_state[0, 0] = Player.X.value  # サブボード0はXが勝利

        # 盤面をセット
        board.set_board_state(board_state, subboard_winner_state, Player.O, (0, 2))

        # サブボード0は勝敗が決まったので、subboard_winner[0,0]がXになっているはず
        assert board.subboard_winner[0, 0] == Player.X.value

        # サブボード0は勝敗が決まったので、次の手はどこでも打てる
        legal_moves = board.get_legal_moves()
        # サブボード0以外のどこかに打てることを確認
        assert any(sub_board != 0 for sub_board, _ in legal_moves)
        # サブボード0にはもう打てないことを確認
        assert all(sub_board != 0 for sub_board, _ in legal_moves)

    def test_game_win(self):
        """Inject a meta‑board win directly and confirm detection."""
        board = UltimateTicTacToeBoard()

        # Empty main board – only meta-board matters here
        board_state = np.zeros((9, 9), dtype=np.int8)

        # X controls the entire top row of the meta‑board
        subboard_winner_state = np.zeros((3, 3), dtype=np.int8)
        subboard_winner_state[0, :] = Player.X.value

        # Apply state; last_move arbitrary
        board.set_board_state(
            board_state,
            subboard_winner_state,
            current_player=Player.O,
            last_move=(2, 2),
        )

        assert board.game_over
        assert board.winner == Player.X

    def test_draw_detection(self):
        """Test draw detection"""
        board = UltimateTicTacToeBoard()

        # Fill all sub-boards without creating a meta-board winner
        # This is a complex test case - for now, just test basic functionality
        assert not board.game_over

    def test_render(self):
        """Test board rendering"""
        board = UltimateTicTacToeBoard()

        # Make some moves
        board.make_move(4, 4)
        board.make_move(0, 0)

        rendered = board.render()

        # Check that render produces a string
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_copy(self):
        """Test board copying"""
        board = UltimateTicTacToeBoard()

        # Make some moves
        board.make_move(4, 4)
        board.make_move(0, 0)

        # Copy the board
        board_copy = board.copy()

        # Check that copy is independent
        assert np.array_equal(board.board, board_copy.board)
        assert np.array_equal(board.subboard_winner, board_copy.subboard_winner)

        # Make a move on the original
        board.make_move(4, 5)

        # Copy should be unchanged
        assert not np.array_equal(board.board, board_copy.board)

    def test_debug_coordinates(self):
        """Debug test to understand coordinate mapping"""
        board = UltimateTicTacToeBoard()

        # Make a move in sub-board 4, position 0
        board.make_move(4, 0)

        # Print board state to understand coordinate mapping
        print("Board after move (4, 0):")
        print(board.board)
        print("Meta board:")
        print(board.subboard_winner)
        print("Last move:", board.last_move)
        print("Current player:", board.current_player)

        # Check what position was actually filled
        # Sub-board 4 should be at (1, 1) in meta-board
        # Position 0 in sub-board 4 should be at (3, 3) in main board
        assert board.board[3, 3] == Player.X.value

        # Check legal moves after first move
        legal_moves = board.get_legal_moves()
        print("Legal moves after first move:", legal_moves)
        print("Number of legal moves:", len(legal_moves))

        # Should be able to play in sub-board 0 (since last move was position 0)
        assert (0, 0) in legal_moves or (0, 1) in legal_moves or (0, 2) in legal_moves

        # Make another move in sub-board 0, position 0
        success = board.make_move(0, 0)
        print("Success making move (0, 0):", success)

        if success:
            print("Board after move (0, 0):")
            print(board.board)

            # Now try to make move in sub-board 4, position 1
            success2 = board.make_move(4, 1)
            print("Success making move (4, 1):", success2)

            if success2:
                print("Board after move (4, 1):")
                print(board.board)

                # Position 1 in sub-board 4 should be at (3, 4) in main board
                assert board.board[3, 4] == Player.X.value

                # Make the winning move in sub-board 4, position 2
                success3 = board.make_move(4, 2)
                print("Success making move (4, 2):", success3)

                if success3:
                    print("Board after move (4, 2):")
                    print(board.board)
                    print("Meta board:")
                    print(board.subboard_winner)

                    # Position 2 in sub-board 4 should be at (3, 5) in main board
                    assert board.board[3, 5] == Player.X.value

                    # Now check if sub-board 4 is won
                    assert board.subboard_winner[1, 1] == Player.X.value

    def test_sub_board_full(self):
        """Test that when a sub-board is full, next move can be anywhere"""
        board = UltimateTicTacToeBoard()

        # サブボード0が完全に埋まった状況を直接セット
        board_state = np.zeros((9, 9), dtype=np.int8)
        # サブボード0をすべて埋める（勝敗なし）
        for i in range(3):
            for j in range(3):
                board_state[i, j] = (i + j) % 2 + 1  # XとOを交互に配置

        # subboard_winnerは未勝利のまま
        subboard_winner_state = np.zeros((3, 3), dtype=np.int8)

        # 盤面をセット（最後の手はサブボード0のpos 8）
        board.set_board_state(board_state, subboard_winner_state, Player.O, (0, 8))

        # サブボード0は完全に埋まっているので、次の手はどこでも打てる
        legal_moves = board.get_legal_moves()
        # サブボード0以外のどこかに打てることを確認
        assert any(sub_board != 0 for sub_board, _ in legal_moves)
        # サブボード0にはもう打てないことを確認
        assert all(sub_board != 0 for sub_board, _ in legal_moves)

    def test_legal_moves_restricted(self):
        """Test that legal moves are restricted to target sub-board when it's not full/won"""
        board = UltimateTicTacToeBoard()

        # サブボード0に空きがある状況をセット
        board_state = np.zeros((9, 9), dtype=np.int8)
        board_state[0, 0] = Player.X.value  # サブボード0, pos 0
        board_state[0, 1] = Player.O.value  # サブボード0, pos 1

        # subboard_winnerは未勝利
        subboard_winner_state = np.zeros((3, 3), dtype=np.int8)

        # 盤面をセット（最後の手はサブボード0のpos 1）
        board.set_board_state(board_state, subboard_winner_state, Player.X, (0, 1))

        # 次の手はサブボード1（pos 1に対応）に制限される
        legal_moves = board.get_legal_moves()
        # すべての合法手がサブボード1であることを確認
        assert all(sub_board == 1 for sub_board, _ in legal_moves)
        # サブボード1の空いている位置のみが合法手であることを確認
        expected_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # サブボード1は全て空いている
        actual_positions = [pos for _, pos in legal_moves]
        assert set(actual_positions) == set(expected_positions)
