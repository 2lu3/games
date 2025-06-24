#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習モデル評価スクリプト（アクション マスク版）
"""

import gymnasium as gym
import torch
import yaml
import pathlib
import argparse
import sys
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.evaluation import evaluate_policy

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ★忘れずに環境登録
from utttrlsim import env_registration


def mask_fn(env):
    """アクション マスク関数"""
    # get_action_mask を持つ本体を再帰的に探す
    while not hasattr(env, "get_action_mask"):
        env = getattr(env, "env", None)
        if env is None:
            raise AttributeError("get_action_mask を持つ環境が見つかりません")
    return env.get_action_mask()


def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Tic-Tac-Toe RL Model Evaluation (Masked)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/policy_masked.zip",
        help="Path to trained masked model",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument(
        "--use-mask", action="store_true", help="Use action masking during evaluation"
    )
    args = parser.parse_args()

    # 設定読み込み
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        env_id = cfg["env_id"]
        model_path = cfg.get("model_path", args.model).replace(".zip", "_masked.zip")
    else:
        env_id = "UTTTRLSim-v0"
        model_path = args.model

    # デバイス選択
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Metal Performance Shaders (MPS) backend")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA backend")
    else:
        device = "cpu"
        print("Using CPU backend")

    # --- モデル読み込み ---
    print(f"Loading masked model from {model_path}")
    try:
        model = MaskablePPO.load(model_path, device=device)
        print("Masked model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Masked model file not found at {model_path}")
        print("Please train a masked model first using train_rl_masked.py")
        return

    # --- 環境作成 ---
    render_mode = "human" if args.render else None
    base_env = gym.make(env_id, render_mode=render_mode)

    # 評価時にマスクを使用するかどうか
    if args.use_mask:
        env = ActionMasker(gym.make(env_id, render_mode=render_mode), mask_fn)
        print("Using action masking during evaluation")
    else:
        env = base_env
        print("Not using action masking during evaluation")

    print(f"Evaluating masked model for {args.episodes} episodes...")

    # --- 評価実行 ---
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        render=args.render,
        deterministic=True,
    )

    # --- 結果出力 ---
    print(f"\n=== Evaluation Results (Masked Model) ===")
    print(f"Episodes: {args.episodes}")
    print(f"Action masking during evaluation: {args.use_mask}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # 追加統計情報
    if mean_reward > 0.5:
        print("🎉 Masked agent is performing well! (mostly winning)")
    elif mean_reward > -0.5:
        print("🤔 Masked agent is drawing frequently")
    else:
        print("😞 Masked agent needs more training (mostly losing)")

    env.close()


if __name__ == "__main__":
    main()
