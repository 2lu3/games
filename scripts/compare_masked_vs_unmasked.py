#!/usr/bin/env python3
"""
アクション マスクありとなしのモデル比較スクリプト
"""

import gymnasium as gym
import torch
import yaml
import pathlib
import argparse
import sys
import numpy as np
from stable_baselines3 import PPO
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

def evaluate_model(model, env, episodes=100, model_name="Model"):
    """モデルを評価して結果を返す"""
    print(f"Evaluating {model_name} for {episodes} episodes...")
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=episodes,
        render=False,
        deterministic=True
    )
    
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser(description="Compare masked vs unmasked models")
    parser.add_argument("--unmasked-model", type=str, default="models/policy.zip", 
                        help="Path to unmasked model")
    parser.add_argument("--masked-model", type=str, default="models/policy_masked.zip", 
                        help="Path to masked model")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of evaluation episodes")
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        env_id = cfg["env_id"]
    else:
        env_id = "UTTTRLSim-v0"
    
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
    models = {}
    
    # マスクなしモデル
    try:
        print(f"Loading unmasked model from {args.unmasked_model}")
        models['unmasked'] = PPO.load(args.unmasked_model, device=device)
        print("Unmasked model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Unmasked model not found at {args.unmasked_model}")
        models['unmasked'] = None
    
    # マスクありモデル
    try:
        print(f"Loading masked model from {args.masked_model}")
        models['masked'] = MaskablePPO.load(args.masked_model, device=device)
        print("Masked model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Masked model not found at {args.masked_model}")
        models['masked'] = None
    
    if not any(models.values()):
        print("Error: No models found!")
        return
    
    # --- 環境作成 ---
    base_env = gym.make(env_id)
    masked_env = ActionMasker(base_env, mask_fn)
    
    # --- 評価実行 ---
    results = {}
    
    # マスクなしモデルの評価
    if models['unmasked']:
        mean_reward, std_reward = evaluate_model(
            models['unmasked'], base_env, args.episodes, "Unmasked"
        )
        results['unmasked'] = (mean_reward, std_reward)
    
    # マスクありモデルの評価（マスクなし環境で）
    if models['masked']:
        mean_reward, std_reward = evaluate_model(
            models['masked'], base_env, args.episodes, "Masked (no mask env)"
        )
        results['masked_no_mask'] = (mean_reward, std_reward)
    
    # マスクありモデルの評価（マスクあり環境で）
    if models['masked']:
        mean_reward, std_reward = evaluate_model(
            models['masked'], masked_env, args.episodes, "Masked (with mask env)"
        )
        results['masked_with_mask'] = (mean_reward, std_reward)
    
    # --- 結果出力 ---
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS ({args.episodes} episodes each)")
    print(f"{'='*60}")
    
    if 'unmasked' in results:
        mean, std = results['unmasked']
        print(f"Unmasked Model:     {mean:6.2f} ± {std:5.2f}")
    
    if 'masked_no_mask' in results:
        mean, std = results['masked_no_mask']
        print(f"Masked Model (no mask env): {mean:6.2f} ± {std:5.2f}")
    
    if 'masked_with_mask' in results:
        mean, std = results['masked_with_mask']
        print(f"Masked Model (with mask env): {mean:6.2f} ± {std:5.2f}")
    
    print(f"{'='*60}")
    
    # 分析
    if 'unmasked' in results and 'masked_no_mask' in results:
        unmasked_mean = results['unmasked'][0]
        masked_mean = results['masked_no_mask'][0]
        improvement = masked_mean - unmasked_mean
        
        print(f"\nAnalysis:")
        print(f"Masked model improvement: {improvement:+.2f}")
        
        if improvement > 0.1:
            print("🎉 Masked model shows significant improvement!")
        elif improvement > 0:
            print("👍 Masked model shows slight improvement")
        elif improvement > -0.1:
            print("🤔 Models perform similarly")
        else:
            print("😞 Masked model performs worse")
    
    # 環境を閉じる
    base_env.close()


if __name__ == "__main__":
    main() 