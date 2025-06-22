#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe 強化学習トレーニングスクリプト（アクション マスク版）
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

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ← 自作環境を Gym 登録しておく
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
    # --- 設定読み込み（プロジェクトルートの YAML を参照） ---
    config_path = project_root / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    os.makedirs(cfg["tensorboard_log"], exist_ok=True)
    
    # --- 環境作成（アクション マスク版） ---
    print(f"Creating environment with action masking...")
    
    # ベース環境を作成
    base_env = gym.make(cfg["env_id"])
    
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
    print("Creating MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        mask_env,
        device=device,
        verbose=1,
        tensorboard_log=cfg["tensorboard_log"],
        **cfg["ppo_params"]
    )
    
    # --- コールバック設定 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, cfg["total_steps"] // 10),  # 最低1回はチェックポイントを保存
        save_path=os.path.dirname(cfg["model_path"]),
        name_prefix="uttt_rl_masked_model"
    )
    
    # --- 学習実行 ---
    print(f"Starting training for {cfg['total_steps']} steps with action masking...")
    model.learn(
        total_timesteps=cfg["total_steps"],
        log_interval=cfg.get("log_interval", 10),
        callback=checkpoint_callback
    )
    
    # --- モデル保存 ---
    model_path = cfg["model_path"].replace(".zip", "_masked.zip")
    print(f"Saving final masked model to {model_path}")
    model.save(model_path)
    
    print("Training completed!")
    print(f"TensorBoard logs: {cfg['tensorboard_log']}")
    print(f"Masked model saved: {model_path}")


if __name__ == "__main__":
    main() 