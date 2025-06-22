# Ultimate Tic-Tac-Toe 強化学習パイプライン セットアップガイド

## 📋 概要

このリポジトリに **4つのファイルだけ追加** してStable Baselines 3による強化学習パイプラインを構築しました：

- `env_registration.py` - Gymnasium環境登録
- `config.yaml` - ハイパーパラメータ設定  
- `train_rl.py` - 学習スクリプト
- `evaluate_rl.py` - 評価スクリプト

## 🚀 セットアップ手順

### 1. 依存関係のインストール

```bash
# 仮想環境を作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 必要なライブラリをインストール
pip install "stable-baselines3[extra]>=2.3" "gymnasium[all]>=1.0" pyyaml torch
```

### 2. 動作確認

環境が正しく登録されるかテスト：

```bash
python3 -c "import env_registration; import gymnasium as gym; env = gym.make('UTTTRLSim-v0'); print('Environment registered successfully!'); print(f'Observation space: {env.observation_space}'); print(f'Action space: {env.action_space}')"
```

## 🎯 使用方法

### 学習の実行

```bash
# デフォルト設定で学習開始
python3 train_rl.py

# TensorBoardでログ監視（別ターミナル）
tensorboard --logdir logs/tb
```

### 学習済みモデルの評価

```bash
# 100エピソードで評価
python3 evaluate_rl.py

# エピソード数を指定
python3 evaluate_rl.py --episodes 50

# 描画モードで実行
python3 evaluate_rl.py --render --episodes 5
```

## ⚙️ 設定カスタマイズ

`config.yaml` を編集して学習パラメータを調整：

```yaml
env_id: UTTTRLSim-v0
n_envs: 16          # 並列環境数（CPUコア数に応じて調整）
total_steps: 2000000 # 学習ステップ数
seed: 42

ppo_params:
  learning_rate: 0.0003
  batch_size: 4096    # バッチサイズ
  n_steps: 2048
  # その他PPOパラメータ...
```

## 📊 期待される結果

### 学習進行例
```
Creating 8 environments...
Using Metal Performance Shaders (MPS) backend  # Apple Silicon
Creating PPO model...
Starting training for 1000000 steps...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 45.2     |
|    ep_rew_mean     | 0.125    |
| time/              |          |
|    fps             | 2847     |
|    iterations      | 100      |
|    time_elapsed    | 72       |
|    total_timesteps | 204800   |
...
```

### 評価結果例
```
=== Evaluation Results ===
Episodes: 100
Mean reward: 0.65 ± 0.32
🎉 Agent is performing well! (mostly winning)
```

## 🛠️ トラブルシューティング

### よくある問題と解決策

**Q. `ModuleNotFoundError: No module named 'utttrlsim'`**
```bash
# プロジェクトルートから実行していることを確認
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python3 train_rl.py
```

**Q. Apple Silicon Macで「MPS backend not available」**
```bash
# PyTorchを最新版にアップデート
pip install --upgrade torch
```

**Q. 学習が遅い**
- `config.yaml`の`n_envs`を増やす（CPUコア数まで）
- `batch_size`と`n_steps`を調整
- GPUが使える場合は自動で使用されます

**Q. 他のRLライブラリに移行したい**
- `env_registration.py`はそのまま使用可能
- TorchRL、RLlib等でも同じ`gym.make("UTTTRLSim-v0")`が使えます

## 📁 ファイル構成

```
your-project/
├── src/utttrlsim/          # 既存の環境実装
│   ├── env.py              # ← UltimateTicTacToeEnv
│   └── ...
├── env_registration.py     # ← 新規追加
├── config.yaml            # ← 新規追加  
├── train_rl.py            # ← 新規追加
├── evaluate_rl.py         # ← 新規追加
├── models/                # 学習済みモデル（自動生成）
└── logs/                  # TensorBoardログ（自動生成）
```

## 🔧 高度な使用法

### 分散学習
```bash
# 環境数を大幅に増やす
sed -i 's/n_envs: 8/n_envs: 32/' config.yaml
python3 train_rl.py
```

### カスタムコールバック
`train_rl.py`の`CheckpointCallback`部分を拡張して独自のコールバックを追加可能。

### 他のアルゴリズム
PPOをA2C、SAC等に変更も簡単：
```python
from stable_baselines3 import A2C
model = A2C("MlpPolicy", vec_env, ...)
```

---

これで**既存コードに一切手を加えることなく**、強化学習パイプラインが完成です！🎉