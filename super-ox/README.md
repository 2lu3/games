# Ultimate Tic-Tac-Toe 強化学習シミュレータ開発計画書

## 1. 概要

Ultimate Tic-Tac-Toe（以下 UTTT）の対戦環境を Python で実装し、各種強化学習（RL）アルゴリズムの研究・実験に利用できる Gym 互換 API を提供する。

## 2. 目的
- UTTT の状態遷移ロジックを正確かつ高速にシミュレートする。
- 強化学習エージェント（DQN・PPO・AlphaZero 風 MCTS+NN など）の学習と評価を容易にする。
- 学術用途でも再利用できるよう、テストとドキュメントを充実させる。

## 3. 要求仕様

### 3.1 機能要件
- 環境 API: OpenAI Gymnasium 準拠（reset(),step(),render()）
- 盤面表現: 81 マスの整数配列 (+ メタボード状況)
- 合法手自動生成: 現在のサブボード状態から合法手を高速抽出
- 勝敗判定: サブ／メタ双方を網羅
- ランダムプレイヤ: ベースライン用
- CLIレンダラー: ASCII／Unicode によるコンソール描画

### 3.2 非機能要件
- 速度: 単一 step() が ≤ 10 µs（M1 Mac 参考）を目標
- 再現性: 任意シードで完全再現可能
- 依存最小: numpy, gymnasium, torch のみ必須
- PEP 8 & typing: 型ヒント／Linter CI
- MIT License: 社外公開を想定

## 4. システム設計

utttrlsim/
 ├─ env.py        # Gym 環境クラス
 ├─ board.py      # 盤面データ構造 & ルール
 ├─ agents/
 │   ├─ random.py
 │   └─ mcts.py   # 任意実装
 ├─ training/
 │   └─ dqn.py    # サンプル学習スクリプト
 ├─ tests/
 └─ docs/

- board.py: ビットボード or numpy.int8 配列で実装。合法手キャッシュを検討。
- env.py: gymnasium.Env を継承し、観測空間を Box(0,2,(9,9),int8) とする。
- agents/: 参照実装として Random と簡易 MCTS を同梱。

## 5. 強化学習基盤
- 学習コード: stable-baselines3 / 自作スクリプト両対応
- 探索強化: AlphaZero 型 => MCTS + NN ポリシー・バリューネットのフックを用意
- 評価: Elo 推定／self-play win-rate

## 6. テスト & QA
- 単体テスト: Pytest で全関数網羅率 ≥ 90 %
- プロパティテスト: Hypothesis で盤面対称性などを検証
- CI: GitHub Actions → lint, mypy, pytest, ビルド

## 7. スケジュール (案)

| フェーズ                | 期間             | 成果物                       |
|-------------------------|------------------|------------------------------|
| 要件定義 & 設計         | 6/24 – 6/30      | 本計画書レビュー完了         |
| コア実装 (board/env)    | 7/1 – 7/7        | 動作する CLI シミュレータ    |
| RL API & baseline       | 7/8 – 7/14       | Gym 環境 + Random/MCTS Agent |
| テスト充実 & CI         | 7/15 – 7/21      | 90 % カバレッジ達成           |
| ドキュメント & 公開     | 7/22 – 7/25      | PyPI / GitHub 公開           |

## 8. リスク & 対応
- 盤面最適化が間に合わない → まず Python 実装、後日 Cython 最適化。
- RL 学習が収束しない → ハイパーパラメータ探索 & MCTS ハイブリッド戦略を検討。

## 9. 今後の拡張案
- ネットワーク対戦 (WebSocket)・GUI
- 教師あり学習データセット生成
- 他サイズ (4×4×4 メタ)

---

## 実装完了状況

### ✅ フェーズ1: コア実装完了 (2024年12月)

#### 実装内容

**1. ディレクトリ構造**
```
utttrlsim/
├── __init__.py          # パッケージ初期化
├── board.py             # 盤面データ構造・ゲームロジック
├── env.py               # Gymnasium準拠環境
├── agents/
│   ├── __init__.py
│   └── random.py        # ランダムエージェント
├── tests/
│   ├── __init__.py
│   ├── test_board.py    # 盤面テスト
│   └── test_env.py      # 環境テスト
├── training/            # 学習スクリプト用（空）
└── docs/               # ドキュメント用（空）
```

**2. 主要機能実装**

**`board.py`**:
- ✅ 9×9のnumpy配列で盤面表現
- ✅ 合法手自動生成（キャッシュ機能付き）
- ✅ サブボード・メタボードの勝敗判定
- ✅ 完全なゲームルール実装
- ✅ 盤面レンダリング（ASCII形式）

**`env.py`**:
- ✅ Gymnasium準拠API（reset, step, render, seed）
- ✅ 観測空間: Box(0,2,(9,9),int8)
- ✅ 行動空間: Discrete(81)
- ✅ 報酬設計（勝利:+1, 敗北:-1, 引き分け:0, 不正手:-100）
- ✅ RGB配列レンダリング機能

**`agents/random.py`**:
- ✅ ランダムエージェント（ベースライン用）
- ✅ 合法手からの一様ランダム選択
- ✅ 行動確率分布生成

**`main.py`**:
- ✅ CLIシミュレータ
- ✅ 人間 vs ランダムエージェント対戦
- ✅ ランダム vs ランダム対戦
- ✅ 環境テスト機能

**3. テスト実装**
- ✅ 盤面ロジックの単体テスト（test_board.py）
- ✅ 環境APIのテスト（test_env.py）
- ✅ 基本的な機能検証

**4. プロジェクト設定**
- ✅ `pyproject.toml`で依存関係管理
- ✅ 開発用ツール設定（black, mypy, pytest）
- ✅ パッケージメタデータ設定

#### 使用方法

**依存関係インストール:**
```bash
uv add numpy gymnasium
```

**テスト実行:**
```bash
python main.py --mode test
```

**人間 vs ランダム対戦:**
```bash
python main.py --mode human
```

**ランダム vs ランダム対戦:**
```bash
python main.py --mode random --games 10
```

#### 技術仕様

- **盤面表現**: numpy.int8配列（9×9）
- **行動空間**: 81次元離散空間（9サブボード × 9位置）
- **観測空間**: 9×9整数配列（0:空, 1:X, 2:O）
- **報酬設計**: 終局時のみ報酬（勝利:+1, 敗北:-1, 引き分け:0）
- **合法手生成**: キャッシュ機能付き高速生成
- **勝敗判定**: サブボード・メタボード両方の判定

#### 次のフェーズ予定

- **フェーズ2**: RL API & baseline（MCTSエージェント実装）
- **フェーズ3**: テスト充実 & CI設定
- **フェーズ4**: ドキュメント & 公開準備

---

以上
