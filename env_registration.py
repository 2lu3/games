"""
自作環境を Gymnasium に登録するだけのファイル。
置き場所は自由。train_rl.py の先頭で import しておく。
"""

import gymnasium as gym
import sys
import os

# プロジェクトのsrcディレクトリをPythonパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 1. あなたの環境クラスを import
from utttrlsim import UltimateTicTacToeEnv

# 2. 一度だけ登録（既登録ならスキップ）
_ENV_ID = "UTTTRLSim-v0"

if _ENV_ID not in gym.envs.registry:
    gym.register(
        id=_ENV_ID,
        entry_point="utttrlsim:UltimateTicTacToeEnv"
    )