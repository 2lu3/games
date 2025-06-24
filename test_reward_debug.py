#!/usr/bin/env python3
"""
報酬計算のデバッグスクリプト
"""

import sys
import pathlib

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from utttrlsim.env import UltimateTicTacToeEnv
from utttrlsim.random_opponent_wrapper import OpponentWrapper
from utttrlsim.agents.random import RandomAgent
from utttrlsim.board import Player


def test_basic_reward():
    """基本的な報酬計算のテスト"""
    print("=== 基本的な報酬計算テスト ===")

    # 通常の環境
    env = UltimateTicTacToeEnv()
    env.reset()

    print(f"初期状態: current_player = {env.board.current_player}")
    print(f"初期状態: game_over = {env.board.game_over}")

    # 最初の手を打つ
    observation, reward, done, truncated, info = env.step(0)
    print(f"手0を打った後: reward = {reward}, done = {done}")
    print(f"手0を打った後: current_player = {env.board.current_player}")

    # 次の手を打つ
    observation, reward, done, truncated, info = env.step(1)
    print(f"手1を打った後: reward = {reward}, done = {done}")
    print(f"手1を打った後: current_player = {env.board.current_player}")


def test_opponent_wrapper_reward():
    """OpponentWrapperの報酬計算テスト"""
    print("\n=== OpponentWrapper報酬計算テスト ===")

    # OpponentWrapper環境
    base_env = UltimateTicTacToeEnv()
    opponent = RandomAgent(seed=42)
    wrapper = OpponentWrapper(base_env, opponent)
    wrapper.reset()

    print(f"初期状態: current_player = {wrapper.env.unwrapped.board.current_player}")

    # PPOエージェントの手を打つ
    observation, reward, done, truncated, info = wrapper.step(0)
    print(f"PPOが手0を打った後: reward = {reward}, done = {done}")
    print(
        f"PPOが手0を打った後: current_player = {wrapper.env.unwrapped.board.current_player}"
    )

    if not done:
        # 相手の手番
        print("相手の手番が実行されました")
        print(
            f"相手の手番後: current_player = {wrapper.env.unwrapped.board.current_player}"
        )


def test_legal_moves():
    """合法手のテスト"""
    print("\n=== 合法手テスト ===")

    env = UltimateTicTacToeEnv()
    env.reset()

    # 最初の手を打つ（どこでも打てる）
    observation, reward, done, truncated, info = env.step(0)
    print(f"手0を打った後: reward = {reward}, done = {done}")

    # 合法手を確認
    legal_moves = env.board.get_legal_moves()
    print(f"合法手の数: {len(legal_moves)}")

    # 手0の位置を確認
    pos0 = env.board.last_move
    print(f"手0の位置: {pos0}")
    print(f"手0のcell_x: {pos0.cell_x}, cell_y: {pos0.cell_y}")

    # 次の手は手0のcell位置に対応するサブボードに打つ必要がある
    target_sub_board = pos0.cell_x + pos0.cell_y * 3
    print(f"次の手を打てるサブボード: {target_sub_board}")

    # そのサブボード内の合法手を確認
    target_moves = [
        move for move in legal_moves if move.sub_grid_id == target_sub_board
    ]
    print(f"ターゲットサブボード{target_sub_board}内の合法手: {len(target_moves)}個")


def test_simple_winning_scenario():
    """簡単な勝利シナリオのテスト"""
    print("\n=== 簡単な勝利シナリオテスト ===")

    env = UltimateTicTacToeEnv()
    env.reset()

    # サブボード0（左上）で3つ並びを作る
    # 手0: サブボード0の左上
    # 手1: サブボード0の上中央（手0のcell位置に対応するサブボード0内）
    # 手2: サブボード0の右上（手1のcell位置に対応するサブボード0内）

    moves = [0, 1, 2]  # サブボード0の上段

    for i, move in enumerate(moves):
        observation, reward, done, truncated, info = env.step(move)
        print(f"手{move}を打った後: reward = {reward}, done = {done}")

        if done:
            print(f"ゲーム終了! 勝者: {env.board.winner}")
            break


if __name__ == "__main__":
    test_basic_reward()
    test_opponent_wrapper_reward()
    test_legal_moves()
    test_simple_winning_scenario()
