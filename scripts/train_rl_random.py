#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習トレーニングスクリプト（Random対戦版）
"""

import os
import pathlib
import sys
from datetime import datetime

import gymnasium as gym
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# 自作環境を Gym 登録しておく
from utttrlsim import env_registration
from utttrlsim.board import Player
from utttrlsim.env import UltimateTicTacToeEnv
from utttrlsim.policies import random_policy
from utttrlsim.wrappers import SelfPlayWrapper

# プロジェクトルートをパスに追加
# project_root = pathlib.Path(__file__).parent.parent
# sys.path.insert(0, str(project_root / "src"))


def mask_fn(env):
    """アクション マスク関数"""
    # get_action_mask を持つ本体を再帰的に探す
    while not hasattr(env, "get_action_mask"):
        env = getattr(env, "env", None)
        if env is None:
            raise AttributeError("get_action_mask を持つ環境が見つかりません")
    return env.get_action_mask()


def make_env(env_id: str, agent_piece: Player, opponent_seed: int = None, rank: int = 0):
    """
    並列環境用の環境作成関数
    
    Args:
        env_id: 環境ID
        agent_piece: エージェントの駒（X or O）
        opponent_seed: 対戦相手の乱数シード
        rank: 環境のランク（並列環境の識別子）
    
    Returns:
        作成された環境
    """
    def _init():
        # 基本環境を作成
        base_env = UltimateTicTacToeEnv()
        
        # SelfPlayWrapperでラップ
        wrapped_env = SelfPlayWrapper(
            base_env,
            agent_piece=agent_piece,
            opponent_policy=random_policy,
            flip_observation=True,
        )
        
        # ActionMaskerでラップ
        mask_env = ActionMasker(wrapped_env, mask_fn)
        
        return mask_env
    
    return _init


def main():
    # --- 設定読み込み（プロジェクトルートの YAML を参照） ---
    config_path = "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Random対戦版の設定を取得
    random_cfg = cfg.get("random_training", {})

    # 並列環境数の取得
    n_envs = cfg.get("n_envs")  # デフォルトは4
    print(f"Using {n_envs} parallel environments")

    # ログディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 基本設定からlog_dirを取得
    base_log_dir = cfg.get("tensorboard_log", "logs/tb")
    log_dir = f"{base_log_dir}/random_vs_agent/{timestamp}"
    model_dir = f"models/random_vs_agent/{timestamp}"

    # ディレクトリ作成
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 並列環境作成 ---
    print(f"Creating {n_envs} parallel environments...")

    agent_piece = Player.X  # 学習エージェントはX固定
    opponent_seed = random_cfg.get("opponent_seed", 42)

    # SubprocVecEnvで並列環境を作成
    env = SubprocVecEnv([
        make_env("UTTTRLSim-v0", agent_piece, opponent_seed, i) 
        for i in range(n_envs)
    ])

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
        "MultiInputPolicy",
        env,
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
                    "n_envs": n_envs,
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
    print(f"Environment: {cfg['env_id']} with SelfPlayWrapper")
    print(f"Total steps: {total_steps}")
    print(f"Parallel environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Opponent seed: {opponent_seed}")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
