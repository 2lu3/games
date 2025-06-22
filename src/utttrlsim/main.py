#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe CLI Simulator

Usage:
    python main.py --mode test
    python main.py --mode human
    python main.py --mode random --games 10
"""

import argparse
import sys
from typing import Optional

from .agents.random import RandomAgent
from .env import UltimateTicTacToeEnv


def main():
    parser = argparse.ArgumentParser(description="Ultimate Tic-Tac-Toe Simulator")
    parser.add_argument(
        "--mode",
        choices=["test", "human", "random"],
        default="test",
        help="Simulation mode",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (random mode only)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.mode == "test":
        run_tests()
    elif args.mode == "human":
        play_human_vs_random()
    elif args.mode == "random":
        play_random_vs_random(args.games, args.seed)


def run_tests():
    """環境の基本的な機能をテスト"""
    print("Running environment tests...")

    env = UltimateTicTacToeEnv()
    env.reset()

    # 基本的なステップテスト
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step test: reward={reward}, terminated={terminated}")

    # レンダリングテスト
    env.render()

    print("Basic tests completed successfully!")


def play_human_vs_random():
    """人間 vs ランダムエージェント対戦"""
    env = UltimateTicTacToeEnv()
    agent = RandomAgent()

    obs, info = env.reset()
    env.render()

    while True:
        if env.current_player == 1:  # 人間のターン
            print("\nあなたのターンです (X)")
            try:
                action = int(input("行動を入力してください (0-80): "))
                if not (0 <= action <= 80):
                    print("無効な行動です。0-80の範囲で入力してください。")
                    continue
            except ValueError:
                print("数値を入力してください。")
                continue
        else:  # エージェントのターン
            print("\nエージェントのターンです (O)")
            action = agent.act(obs, env.legal_actions)
            print(f"エージェントの行動: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            if reward == 1:
                print("あなたの勝利！")
            elif reward == -1:
                print("エージェントの勝利！")
            else:
                print("引き分け！")
            break


def play_random_vs_random(num_games: int, seed: Optional[int] = None):
    """ランダム vs ランダム対戦"""
    env = UltimateTicTacToeEnv()
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    if seed is not None:
        env.seed(seed)

    wins = [0, 0]  # [agent1 wins, agent2 wins]
    draws = 0

    for game in range(num_games):
        obs, info = env.reset()

        while True:
            if env.current_player == 1:
                action = agent1.act(obs, env.legal_actions)
            else:
                action = agent2.act(obs, env.legal_actions)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                if reward == 1:
                    wins[0] += 1
                elif reward == -1:
                    wins[1] += 1
                else:
                    draws += 1
                break

        if num_games <= 10:  # 少ないゲーム数の場合は結果を表示
            print(f"Game {game + 1}: ", end="")
            if reward == 1:
                print("Agent 1 wins")
            elif reward == -1:
                print("Agent 2 wins")
            else:
                print("Draw")

    print(f"\nResults after {num_games} games:")
    print(f"Agent 1 wins: {wins[0]}")
    print(f"Agent 2 wins: {wins[1]}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    main()
