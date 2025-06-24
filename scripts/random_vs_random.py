#!/usr/bin/env python3
"""
Random Agent vs Random Agent 対戦スクリプト

Usage:
    python scripts/random_vs_random.py
    python scripts/random_vs_random.py --games 100 --seed 42 --verbose
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utttrlsim.agents.random import RandomAgent
from src.utttrlsim.env import UltimateTicTacToeEnv


def main():
    parser = argparse.ArgumentParser(description="Random Agent vs Random Agent 対戦")
    parser.add_argument(
        "--games", type=int, default=10, help="対戦回数 (デフォルト: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="乱数シード (デフォルト: None)"
    )
    parser.add_argument("--verbose", action="store_true", help="詳細出力を有効にする")

    args = parser.parse_args()

    # 対戦実行
    run_random_vs_random(args.games, args.seed, args.verbose)


def run_random_vs_random(num_games: int, seed: int = None, verbose: bool = False):
    """ランダムエージェント同士の対戦を実行"""

    print("Random Agent vs Random Agent")
    print("=" * 30)

    # 環境とエージェントを初期化
    env = UltimateTicTacToeEnv()
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    # シード設定
    if seed is not None:
        env.seed(seed)
        if verbose:
            print(f"Random seed: {seed}")

    # 統計初期化
    wins = [0, 0]  # [agent1 wins, agent2 wins]
    draws = 0

    # 対戦ループ
    for game in range(num_games):
        if verbose:
            print(f"\nGame {game + 1}/{num_games}")

        obs, info = env.reset()

        while True:
            # 現在のプレイヤーに応じてエージェントを選択
            if env.board.current_player.value == 1:  # Player.X
                action = agent1.select_action_from_env(env)
            else:  # Player.O
                action = agent2.select_action_from_env(env)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                # 結果を記録
                if reward == 1:
                    wins[0] += 1
                    if verbose:
                        print("Agent 1 wins")
                elif reward == -1:
                    wins[1] += 1
                    if verbose:
                        print("Agent 2 wins")
                else:
                    draws += 1
                    if verbose:
                        print("Draw")
                break

    # 結果表示
    print(f"\nResults after {num_games} games:")
    print(f"Agent 1 wins: {wins[0]}")
    print(f"Agent 2 wins: {wins[1]}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    main()
