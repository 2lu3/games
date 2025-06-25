#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 学習済みモデル vs RandomAgent 評価スクリプト
"""

import yaml
import torch
import os
import pathlib
import sys
import argparse
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 自作環境を Gym 登録しておく
from utttrlsim import env_registration
from utttrlsim.wrapper.random_opponent_wrapper import OpponentWrapper
from utttrlsim.agent.random_agent import RandomAgent


def mask_fn(env):
    """アクション マスク関数"""
    # get_action_mask を持つ本体を再帰的に探す
    while not hasattr(env, "get_action_mask"):
        env = getattr(env, "env", None)
        if env is None:
            raise AttributeError("get_action_mask を持つ環境が見つかりません")
    return env.get_action_mask()


def load_model(model_path: str, device: str = "auto"):
    """
    学習済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        device: デバイス指定

    Returns:
        読み込まれたモデル
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = MaskablePPO.load(model_path, device=device)
    return model


def create_evaluation_env(env_id: str, opponent_seed: int = None):
    """
    評価用環境を作成

    Args:
        env_id: 環境ID
        opponent_seed: 対戦相手の乱数シード

    Returns:
        評価用環境
    """
    base_env = gym.make(env_id)
    opponent_agent = RandomAgent(seed=opponent_seed)
    wrapped_env = OpponentWrapper(base_env, opponent_agent=opponent_agent)
    mask_env = ActionMasker(wrapped_env, mask_fn)
    return mask_env


def play_single_game(model, env, game_id: int) -> Dict:
    """
    1試合を実行

    Args:
        model: 学習済みモデル
        env: 評価用環境
        game_id: ゲームID

    Returns:
        ゲーム結果の辞書
    """
    observation, info = env.reset()
    step_count = 0
    max_steps = 100  # 安全限界

    game_log = []

    while not env.env.board.game_over and step_count < max_steps:
        # モデルの行動選択
        action, _ = model.predict(observation, deterministic=True)

        # 環境でステップ実行
        observation, reward, done, truncated, info = env.step(action)
        step_count += 1

        # ゲームログに記録
        game_log.append(
            {
                "step": step_count,
                "action": int(action),
                "reward": float(reward),
                "done": done,
                "current_player": env.env.board.current_player.value,
                "meta_board": env.env.board.subboard_winner.copy(),
            }
        )

        if done:
            break

    # ゲーム結果を判定
    winner = env.env.board.winner
    if winner.value == 1:  # Player X (モデル) の勝利
        result = "win"
        win_reward = 1.0
    elif winner.value == 2:  # Player O (RandomAgent) の勝利
        result = "loss"
        win_reward = -1.0
    else:  # 引き分け
        result = "draw"
        win_reward = 0.0

    return {
        "game_id": game_id,
        "result": result,
        "winner": winner.value,
        "steps": step_count,
        "win_reward": win_reward,
        "game_log": game_log,
        "final_board": env.env.board.board.copy(),
        "final_meta_board": env.env.board.subboard_winner.copy(),
    }


def evaluate_model(
    model_path: str,
    num_games: int = 1000,
    opponent_seed: int = 42,
    output_dir: str = None,
    target_win_rate: float = 0.7,
) -> Dict:
    """
    モデルを評価

    Args:
        model_path: モデルファイルのパス
        num_games: 評価ゲーム数
        opponent_seed: 対戦相手の乱数シード
        output_dir: 出力ディレクトリ
        target_win_rate: 目標勝率

    Returns:
        評価結果の辞書
    """
    # デバイス選択
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA backend")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS backend (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU backend")

    # モデル読み込み
    model = load_model(model_path, device=device)

    # 環境作成
    env = create_evaluation_env("UTTTRLSim-v0", opponent_seed=opponent_seed)

    # 評価実行
    print(f"Starting evaluation: {num_games} games vs RandomAgent...")

    results = []
    wins = 0
    losses = 0
    draws = 0

    for game_id in range(num_games):
        if game_id % 100 == 0:
            print(f"Progress: {game_id}/{num_games} games completed")

        game_result = play_single_game(model, env, game_id)
        results.append(game_result)

        if game_result["result"] == "win":
            wins += 1
        elif game_result["result"] == "loss":
            losses += 1
        else:
            draws += 1

    # 統計計算
    win_rate = wins / num_games
    loss_rate = losses / num_games
    draw_rate = draws / num_games

    avg_steps = np.mean([r["steps"] for r in results])
    std_steps = np.std([r["steps"] for r in results])

    # 結果出力
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total games: {num_games}")
    print(f"Wins: {wins} ({win_rate:.1%})")
    print(f"Losses: {losses} ({loss_rate:.1%})")
    print(f"Draws: {draws} ({draw_rate:.1%})")
    print(f"Average steps per game: {avg_steps:.1f} ± {std_steps:.1f}")
    print(f"Target win rate: {target_win_rate:.1%}")
    print(f"Achieved win rate: {win_rate:.1%}")

    if win_rate >= target_win_rate:
        print(f"✓ Target win rate achieved!")
    else:
        print(
            f"✗ Target win rate not achieved (need {target_win_rate - win_rate:.1%} more)"
        )

    print("=" * 60)

    # CSV出力
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # サマリーCSV
        summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Games", num_games])
            writer.writerow(["Wins", wins])
            writer.writerow(["Losses", losses])
            writer.writerow(["Draws", draws])
            writer.writerow(["Win Rate", f"{win_rate:.4f}"])
            writer.writerow(["Loss Rate", f"{loss_rate:.4f}"])
            writer.writerow(["Draw Rate", f"{draw_rate:.4f}"])
            writer.writerow(["Average Steps", f"{avg_steps:.2f}"])
            writer.writerow(["Steps Std Dev", f"{std_steps:.2f}"])
            writer.writerow(["Target Win Rate", f"{target_win_rate:.4f}"])
            writer.writerow(["Target Achieved", win_rate >= target_win_rate])

        # 詳細結果CSV
        details_path = os.path.join(output_dir, f"evaluation_details_{timestamp}.csv")
        with open(details_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Game ID", "Result", "Winner", "Steps", "Win Reward"])
            for result in results:
                writer.writerow(
                    [
                        result["game_id"],
                        result["result"],
                        result["winner"],
                        result["steps"],
                        result["win_reward"],
                    ]
                )

        print(f"Results saved to:")
        print(f"  Summary: {summary_path}")
        print(f"  Details: {details_path}")

    return {
        "total_games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "avg_steps": avg_steps,
        "std_steps": std_steps,
        "target_achieved": win_rate >= target_win_rate,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model vs RandomAgent"
    )
    parser.add_argument("model_path", help="Path to the trained model file")
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to play (default: 1000)",
    )
    parser.add_argument(
        "--opponent-seed",
        type=int,
        default=42,
        help="Random seed for opponent (default: 42)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for results (default: logs/evaluation)"
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.7,
        help="Target win rate (default: 0.7)",
    )

    args = parser.parse_args()

    # デフォルト出力ディレクトリ
    if args.output_dir is None:
        args.output_dir = "logs/evaluation"

    # 評価実行
    results = evaluate_model(
        model_path=args.model_path,
        num_games=args.num_games,
        opponent_seed=args.opponent_seed,
        output_dir=args.output_dir,
        target_win_rate=args.target_win_rate,
    )

    # CLIオプションで目標勝率を表示
    print(f"\nCLI Options:")
    print(f"  Model path: {args.model_path}")
    print(f"  Number of games: {args.num_games}")
    print(f"  Opponent seed: {args.opponent_seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Target win rate: {args.target_win_rate:.1%}")


if __name__ == "__main__":
    main()
