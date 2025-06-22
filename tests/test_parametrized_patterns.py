"""
Parametrized Testing Examples for Restriction/Release Logic Patterns

This module demonstrates how to use @pytest.mark.parametrize to compress duplicate tests
for logic patterns related to restrictions and releases (制限・解放).
"""

import numpy as np
import pytest
from src.utttrlsim.board import UltimateTicTacToeBoard, Player, Position
from src.utttrlsim.env import UltimateTicTacToeEnv


class TestParametrizedRestrictionRelease:
    """Parametrized tests for restriction/release logic patterns"""

    @pytest.mark.parametrize("initial_position,expected_restricted_subgrid,expected_moves_count", [
        # 制限パターン：初期位置から特定のサブグリッドに制限される
        (Position(0), (0, 0), 8),    # 左上角 → サブグリッド(0,0)制限
        (Position(4), (1, 1), 8),    # 中央上 → サブグリッド(1,1)制限
        (Position(8), (2, 2), 8),    # 右上角 → サブグリッド(2,2)制限
        (Position(40), (1, 1), 8),   # 全体中央 → サブグリッド(1,1)制限
        (Position(80), (2, 2), 8),   # 右下角 → サブグリッド(2,2)制限
    ])
    def test_move_restriction_patterns(self, initial_position, expected_restricted_subgrid, expected_moves_count):
        """Test that moves are restricted to specific sub-grids after initial moves"""
        board = UltimateTicTacToeBoard()
        
        # 初期状態：全ての位置が利用可能
        initial_legal_moves = board.get_legal_moves()
        assert len(initial_legal_moves) == 81
        
        # 制限を引き起こす手を打つ
        success = board.make_move(initial_position)
        assert success
        
        # 制限後：指定されたサブグリッドのみが利用可能
        restricted_legal_moves = board.get_legal_moves()
        assert len(restricted_legal_moves) == expected_moves_count
        
        # すべての合法手が期待されるサブグリッドにある
        for move in restricted_legal_moves:
            assert move.sub_grid_x == expected_restricted_subgrid[0]
            assert move.sub_grid_y == expected_restricted_subgrid[1]

    @pytest.mark.parametrize("board_state_setup,expected_release_condition", [
        # 解放パターン1：サブボードが勝利で解放
        ("subboard_won", "won_subboard_unavailable"),
        # 解放パターン2：サブボードが満杯で解放
        ("subboard_full", "full_subboard_unavailable"),
        # 解放パターン3：サブボードが勝利且つ満杯で解放
        ("subboard_won_and_full", "won_full_subboard_unavailable"),
    ])
    def test_move_release_patterns(self, board_state_setup, expected_release_condition):
        """Test that moves are released when sub-boards become unavailable"""
        board = UltimateTicTacToeBoard()
        
        if board_state_setup == "subboard_won":
            # サブボード0をXで勝利状態にする
            board_state = np.zeros((9, 9), dtype=np.int8)
            board_state[0, 0] = Player.X.value  # Position(0)
            board_state[0, 1] = Player.X.value  # Position(1)
            board_state[0, 2] = Player.X.value  # Position(2)
            board.set_board_state(board_state, Player.O, Position(0))
            
        elif board_state_setup == "subboard_full":
            # サブボード0を満杯にする（勝者なし）
            board_state = np.zeros((9, 9), dtype=np.int8)
            pattern = [Player.X.value, Player.O.value, Player.X.value,
                      Player.O.value, Player.X.value, Player.O.value,
                      Player.X.value, Player.O.value, Player.X.value]
            for i in range(3):
                for j in range(3):
                    board_state[i, j] = pattern[i * 3 + j]
            board.set_board_state(board_state, Player.O, Position(0))
            
        elif board_state_setup == "subboard_won_and_full":
            # サブボード0を勝利且つ満杯にする
            board_state = np.zeros((9, 9), dtype=np.int8)
            # トップ行をXで勝利
            board_state[0, 0] = Player.X.value
            board_state[0, 1] = Player.X.value
            board_state[0, 2] = Player.X.value
            # 残りを埋める
            board_state[1, 0] = Player.O.value
            board_state[1, 1] = Player.O.value
            board_state[1, 2] = Player.O.value
            board_state[2, 0] = Player.X.value
            board_state[2, 1] = Player.X.value
            board_state[2, 2] = Player.O.value
            board.set_board_state(board_state, Player.O, Position(0))
        
        # 解放されたことを確認：サブボード0では打てない
        legal_moves = board.get_legal_moves()
        subgrids_with_moves = set()
        for pos in legal_moves:
            subgrids_with_moves.add((pos.sub_grid_x, pos.sub_grid_y))
        
        # サブボード(0,0)では打てない
        assert (0, 0) not in subgrids_with_moves
        # 他のサブボードでは打てる
        assert len(subgrids_with_moves) > 0

    @pytest.mark.parametrize("invalid_move_scenario,expected_result", [
        # 無効手のパターン
        ("same_position_twice", False),           # 同じ位置に2回打つ
        ("restricted_subboard_violation", False), # 制限されたサブボード外に打つ
        ("won_subboard_move", False),            # 勝利済みサブボードに打つ
        ("full_subboard_move", False),           # 満杯サブボードに打つ
    ])
    def test_invalid_move_patterns(self, invalid_move_scenario, expected_result):
        """Test various invalid move scenarios"""
        board = UltimateTicTacToeBoard()
        
        if invalid_move_scenario == "same_position_twice":
            # 同じ位置に2回打とうとする
            position = Position(40)
            board.make_move(position)
            result = board.make_move(position)
            
        elif invalid_move_scenario == "restricted_subboard_violation":
            # 制限されたサブボード外に打とうとする
            board.make_move(Position(0))  # サブボード(0,0)に制限される
            # 別のサブボードに打とうとする
            result = board.make_move(Position(40))  # サブボード(1,1)
            
        elif invalid_move_scenario == "won_subboard_move":
            # 勝利済みサブボードに打とうとする
            board_state = np.zeros((9, 9), dtype=np.int8)
            board_state[0, 0] = Player.X.value
            board_state[0, 1] = Player.X.value
            board_state[0, 2] = Player.X.value
            board.set_board_state(board_state, Player.O, Position(1))
            # 勝利済みサブボード(0,0)に打とうとする
            result = board.make_move(Position(18))  # サブボード(0,0)の位置
            
        elif invalid_move_scenario == "full_subboard_move":
            # 満杯サブボードに打とうとする
            board_state = np.zeros((9, 9), dtype=np.int8)
            pattern = [Player.X.value, Player.O.value, Player.X.value,
                      Player.O.value, Player.X.value, Player.O.value,
                      Player.X.value, Player.O.value, Player.X.value]
            for i in range(3):
                for j in range(3):
                    board_state[i, j] = pattern[i * 3 + j]
            board.set_board_state(board_state, Player.O, Position(1))
            # 満杯サブボード(0,0)に打とうとする
            result = board.make_move(Position(4))  # サブボード(0,0)の位置
        
        assert result == expected_result

    @pytest.mark.parametrize("player,expected_game_state", [
        # ゲーム終了パターン
        (Player.X, "x_wins"),
        (Player.O, "o_wins"),
    ])
    def test_game_termination_patterns(self, player, expected_game_state):
        """Test various game termination scenarios"""
        board = UltimateTicTacToeBoard()
        
        # メタボードでプレイヤーが勝利する状態を作る
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # 上段の3つのサブボードでプレイヤーが勝利
        for sub_grid_x in range(3):
            start_x = sub_grid_x * 3
            for i in range(3):
                board_state[0, start_x + i] = player.value
        
        board.set_board_state(board_state, Player.O, Position(2))
        
        # ゲーム終了確認
        assert board.game_over
        assert board.winner == player


class TestParametrizedEnvironmentPatterns:
    """Parametrized tests for environment restriction/release patterns"""

    @pytest.mark.parametrize("render_mode,expected_output_type", [
        # レンダリングモードのパターン
        ("human", type(None)),
        ("rgb_array", np.ndarray),
    ])
    def test_render_mode_patterns(self, render_mode, expected_output_type):
        """Test different rendering modes"""
        env = UltimateTicTacToeEnv(render_mode=render_mode)
        env.reset()
        env.step(40)
        
        result = env.render()
        assert type(result) == expected_output_type
        
        if render_mode == "rgb_array":
            assert result.shape == (300, 300, 3)
            assert result.dtype == np.uint8

    @pytest.mark.parametrize("invalid_action,expected_behavior", [
        # 無効アクションのパターン
        (-1, "value_error"),
        (81, "value_error"),
        (100, "value_error"),
    ])
    def test_invalid_action_patterns(self, invalid_action, expected_behavior):
        """Test various invalid action scenarios in environment"""
        env = UltimateTicTacToeEnv()
        env.reset()
        
        if expected_behavior == "value_error":
            with pytest.raises(ValueError):
                env.step(invalid_action)

    @pytest.mark.parametrize("game_sequence,expected_legal_actions_count", [
        # ゲームシーケンスによる合法手の変化パターン
        ([40], 8),           # 中央に打った後：8手
        ([0, 36], 8),        # 角に打って応答した後：8手
        ([40, 0, 36], 8),    # 3手打った後：8手
    ])
    def test_legal_actions_sequence_patterns(self, game_sequence, expected_legal_actions_count):
        """Test legal actions count after different move sequences"""
        env = UltimateTicTacToeEnv()
        env.reset()
        
        for action in game_sequence:
            env.step(action)
        
        legal_actions = env.get_legal_actions()
        assert np.sum(legal_actions) == expected_legal_actions_count

    @pytest.mark.parametrize("seed_value", [42, 123, 999])
    def test_reproducible_environment_patterns(self, seed_value):
        """Test that environment behavior is reproducible with same seed"""
        env1 = UltimateTicTacToeEnv()
        env2 = UltimateTicTacToeEnv()
        
        obs1, info1 = env1.reset(seed=seed_value)
        obs2, info2 = env2.reset(seed=seed_value)
        
        # 同じシードで同じ状態になる
        assert np.array_equal(obs1, obs2)
        assert info1["current_player"] == info2["current_player"]
        
        # 同じアクションで同じ結果になる
        obs1, reward1, done1, truncated1, info1 = env1.step(40)
        obs2, reward2, done2, truncated2, info2 = env2.step(40)
        
        assert np.array_equal(obs1, obs2)
        assert reward1 == reward2
        assert done1 == done2
        assert truncated1 == truncated2


class TestParametrizedEdgeCases:
    """Parametrized tests for edge cases in restriction/release logic"""

    @pytest.mark.parametrize("board_fill_percentage,expected_available_moves", [
        # ボードの埋まり具合による利用可能手のパターン
        (0.1, "many_available"),      # 10%埋まり：多くの手が利用可能
        (0.5, "some_available"),      # 50%埋まり：一部の手が利用可能
        (0.9, "few_available"),       # 90%埋まり：少数の手が利用可能
    ])
    def test_board_fill_progression_patterns(self, board_fill_percentage, expected_available_moves):
        """Test legal moves as board fills up"""
        board = UltimateTicTacToeBoard()
        
        # ボードを指定の割合まで埋める
        total_positions = 81
        positions_to_fill = int(total_positions * board_fill_percentage)
        
        # ランダムに位置を選んで埋める（簡単な実装）
        filled_positions = set()
        player = Player.X
        
        for _ in range(positions_to_fill):
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # 最初の合法手を選ぶ
            position = legal_moves[0]
            if position.board_id not in filled_positions:
                board.make_move(position)
                filled_positions.add(position.board_id)
                player = Player.O if player == Player.X else Player.X
        
        # 残りの合法手を確認
        remaining_legal_moves = board.get_legal_moves()
        
        if expected_available_moves == "many_available":
            assert len(remaining_legal_moves) > 50
        elif expected_available_moves == "some_available":
            assert 10 <= len(remaining_legal_moves) <= 50
        elif expected_available_moves == "few_available":
            assert len(remaining_legal_moves) <= 10

    @pytest.mark.parametrize("winning_pattern,winner", [
        # 勝利パターンの種類
        ("top_row", Player.X),
        ("middle_row", Player.O),
        ("bottom_row", Player.X),
        ("left_column", Player.O),
        ("middle_column", Player.X),
        ("right_column", Player.O),
        ("main_diagonal", Player.X),
        ("anti_diagonal", Player.O),
    ])
    def test_winning_patterns(self, winning_pattern, winner):
        """Test different winning patterns on meta board"""
        board = UltimateTicTacToeBoard()
        board_state = np.zeros((9, 9), dtype=np.int8)
        
        # 各勝利パターンに応じてサブボードを設定
        winning_positions = {
            "top_row": [(0, 0), (1, 0), (2, 0)],
            "middle_row": [(0, 1), (1, 1), (2, 1)],
            "bottom_row": [(0, 2), (1, 2), (2, 2)],
            "left_column": [(0, 0), (0, 1), (0, 2)],
            "middle_column": [(1, 0), (1, 1), (1, 2)],
            "right_column": [(2, 0), (2, 1), (2, 2)],
            "main_diagonal": [(0, 0), (1, 1), (2, 2)],
            "anti_diagonal": [(2, 0), (1, 1), (0, 2)],
        }
        
        # 勝利パターンのサブボードを設定
        for sub_x, sub_y in winning_positions[winning_pattern]:
            # サブボードの上段をwinnerで埋める
            start_x = sub_x * 3
            start_y = sub_y * 3
            for i in range(3):
                board_state[start_y, start_x + i] = winner.value
        
        board.set_board_state(board_state, Player.O, Position(2))
        
        # ゲーム終了と勝者を確認
        assert board.game_over
        assert board.winner == winner