#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習モデル評価スクリプト
"""

import gymnasium as gym
import torch
import yaml
import pathlib
import argparse
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ★忘れずに環境登録
from utttrlsim import env_registration


def main():
    parser = argparse.ArgumentParser(description="Ultimate Tic-Tac-Toe RL Model Evaluation")
    parser.add_argument("--model", type=str, default="models/policy.zip", 
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", 
                        help="Render episodes")
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        env_id = cfg["env_id"]
        model_path = cfg.get("model_path", args.model)
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
    print(f"Loading model from {model_path}")
    try:
        model = PPO.load(model_path, device=device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using train_rl.py")
        return
    
    # --- 環境作成 ---
    render_mode = "human" if args.render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    print(f"Evaluating model for {args.episodes} episodes...")
    
    # --- 評価実行 ---
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=args.episodes,
        render=args.render,
        deterministic=True
    )
    
    # --- 結果出力 ---
    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 追加統計情報
    if mean_reward > 0.5:
        print("🎉 Agent is performing well! (mostly winning)")
    elif mean_reward > -0.5:
        print("🤔 Agent is drawing frequently")
    else:
        print("😞 Agent needs more training (mostly losing)")
    
    env.close()


if __name__ == "__main__":
    main()