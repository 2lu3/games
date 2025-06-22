"""
自作環境を Gymnasium に登録するだけのファイル。
"""

import gymnasium as gym

# 1. あなたの環境クラスを import（同じパッケージ内なので直接インポート可能）
from .env import UltimateTicTacToeEnv

# 2. 一度だけ登録（既登録ならスキップ）
_ENV_ID = "UTTTRLSim-v0"

if _ENV_ID not in gym.envs.registry:
    gym.register(
        id=_ENV_ID,
        entry_point="utttrlsim:UltimateTicTacToeEnv"
    )