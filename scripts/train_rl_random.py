#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習トレーニングスクリプト（Random対戦版）
"""

import yaml
import torch
import os
import pathlib
import sys
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 自作環境を Gym 登録しておく
from utttrlsim import env_registration
from utttrlsim.random_opponent_wrapper import OpponentWrapper
from utttrlsim.agents.random import RandomAgent


def mask_fn(env):
    """アクション マスク関数"""
    # get_action_mask を持つ本体を再帰的に探す
    while not hasattr(env, "get_action_mask"):
        env = getattr(env, "env", None)
        if env is None:
            raise AttributeError("get_action_mask を持つ環境が見つかりません")
    return env.get_action_mask()


def create_opponent_env(env_id: str, opponent_agent):
    base_env = gym.make(env_id)
    wrapped_env = OpponentWrapper(base_env, opponent_agent=opponent_agent)
    return wrapped_env


def main():
    # --- 設定読み込み（プロジェクトルートの YAML を参照） ---
    config_path = project_root / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Random対戦版の設定を取得
    random_cfg = cfg.get("random_training", {})

    # ログディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 基本設定からlog_dirを取得
    base_log_dir = cfg.get("tensorboard_log", "logs/tb")
    log_dir = f"{base_log_dir}/random_vs_agent/{timestamp}"
    model_dir = f"models/random_vs_agent/{timestamp}"

    # ディレクトリ作成
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 環境作成（OpponentWrapper版） ---
    print(f"Creating OpponentWrapper environment...")

    opponent_agent = RandomAgent(seed=random_cfg.get("opponent_seed", 42))
    base_env = create_opponent_env(cfg["env_id"], opponent_agent)

    # ActionMaskerでラップ
    mask_env = ActionMasker(base_env, mask_fn)

    # デバイス選択 (Apple Silicon対応)
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA backend")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS backend (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU backend")

    # --- MaskablePPOモデル作成 ---
    print("Creating MaskablePPO model for Random opponent training...")

    # Random対戦版用のハイパーパラメータ
    ppo_params = cfg["ppo_params"].copy()
    random_ppo_params = random_cfg.get("ppo_params", {})
    ppo_params.update(random_ppo_params)

    model = MaskablePPO(
        "MlpPolicy",
        mask_env,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        **ppo_params,
    )

    # --- コールバック設定 ---
    # 基本設定からtotal_stepsを取得
    total_steps = cfg["total_steps"]
    checkpoint_freq = max(1, total_steps // 10)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=model_dir,
        name_prefix="uttt_rl_random_model",
    )

    # --- 学習実行 ---
    print(f"Starting Random opponent training for {total_steps} steps...")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")

    model.learn(
        total_timesteps=total_steps,
        log_interval=cfg.get("log_interval", 10),
        callback=checkpoint_callback,
    )

    # --- モデル保存 ---
    final_model_path = os.path.join(model_dir, "final_model.zip")
    print(f"Saving final Random opponent model to {final_model_path}")
    model.save(final_model_path)

    # --- 設定ファイルも保存 ---
    config_save_path = os.path.join(model_dir, "training_config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "base_config": cfg,
                "random_training_config": random_cfg,
                "training_info": {
                    "timestamp": timestamp,
                    "total_steps": total_steps,
                    "device": device,
                    "model_path": final_model_path,
                },
            },
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    print("Random opponent training completed!")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Final model saved: {final_model_path}")
    print(f"Config saved: {config_save_path}")

    # --- 学習結果の要約 ---
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Environment: {cfg['env_id']} with OpponentWrapper")
    print(f"Total steps: {total_steps}")
    print(f"Device: {device}")
    print(f"Opponent seed: {random_cfg.get('opponent_seed', 42)}")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
