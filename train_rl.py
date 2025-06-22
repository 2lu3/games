#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習トレーニングスクリプト
"""

import yaml
import torch
import os
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# ← 自作環境を Gym 登録しておく
import env_registration

def main():
    # --- 設定読み込み（スクリプトと同じ場所の YAML を想定） ---
    config_path = pathlib.Path(__file__).with_suffix(".yaml")
    if not config_path.exists():
        config_path = pathlib.Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    os.makedirs(cfg["tensorboard_log"], exist_ok=True)
    
    # --- VecEnv 生成 ---
    print(f"Creating {cfg['n_envs']} environments...")
    vec_env = make_vec_env(
        cfg["env_id"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
    )
    
    # デバイス選択 (Apple Silicon対応)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Metal Performance Shaders (MPS) backend")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA backend")
    else:
        device = "cpu"
        print("Using CPU backend")
    
    # --- モデル作成 ---
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        device=device,
        verbose=1,
        tensorboard_log=cfg["tensorboard_log"],
        **cfg["ppo_params"]
    )
    
    # --- コールバック設定 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, cfg["total_steps"] // 10),  # 最低1回はチェックポイントを保存
        save_path=os.path.dirname(cfg["model_path"]),
        name_prefix="uttt_rl_model"
    )
    
    # --- 学習実行 ---
    print(f"Starting training for {cfg['total_steps']} steps...")
    model.learn(
        total_timesteps=cfg["total_steps"],
        log_interval=cfg.get("log_interval", 10),
        callback=checkpoint_callback
    )
    
    # --- モデル保存 ---
    print(f"Saving final model to {cfg['model_path']}")
    model.save(cfg["model_path"])
    
    print("Training completed!")
    print(f"TensorBoard logs: {cfg['tensorboard_log']}")
    print(f"Model saved: {cfg['model_path']}")


if __name__ == "__main__":
    main()