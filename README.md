# Ultimate Tic-Tac-Toe 強化学習シミュレータ - 開発者ドキュメント

Ultimate Tic-Tac-Toe（UTTT）の強化学習環境を実装したPythonライブラリです。Gymnasium準拠のAPIを提供し、RLアルゴリズムの研究・実験に利用できます。

## Todo

- [x] テストデータ生成のヘルパー関数を抽出。長い board_state 構築ロジックが複数回登場。def fill_sub_board(board, grid_x, grid_y, pattern): ... のように共通化するとテスト本体が短くなる。
- [ ] 境界値・ランダム生成テストの追加→Hypothesis などで “ランダム合法手を生成 → 差分整合性チェック” を入れると手動で書ききれないエッジケースも網羅できる
- [ ] ネガティブテストをもう一歩深く。test_make_invalid_move は「同じマスを２回置く」だけ。他にも「ターゲット外のサブボードへ置く」「ゲーム終了後に置く」など異常系を網羅すると安心。


## アーキテクチャ概要

### コアコンポーネント

```
src/utttrlsim/
├── board.py             # ゲームロジック・盤面管理
├── env.py               # Gymnasium環境ラッパー
├── agents/random.py     # ベースラインエージェント
└── main.py              # CLIインターフェース
```

### データフロー

1. **環境初期化**: `UltimateTicTacToeEnv` → `UltimateTicTacToeBoard`
2. **行動実行**: `env.step(action)` → `board.make_move()` → 状態更新
3. **観測生成**: 盤面状態 → numpy配列（9×9）
4. **報酬計算**: 終局判定 → 報酬値（+1/-1/0）

## 実装詳細

### 盤面データ構造 (`board.py`)

#### クラス設計

```python
class UltimateTicTacToeBoard:
    def __init__(self):
        self.board: np.ndarray  # shape: (9, 9), dtype: int8
        self.meta_board: np.ndarray  # shape: (3, 3), dtype: int8
        self.current_player: int  # 1: X, 2: O
        self.last_move: Optional[int]  # 0-80の座標
        self.game_over: bool
        self.winner: Optional[int]
```

#### 座標系

- **全体座標**: 0-80（左上から右下へ連番）
- **サブボード座標**: 0-8（3×3の各サブボード内）
- **メタボード座標**: 0-8（3×3のメタボード）

#### 座標変換

```python
def sub_grid_to_global(sub_grid: int, local_pos: int) -> int:
    """サブボード座標 → 全体座標"""
    sub_row, sub_col = sub_grid // 3, sub_grid % 3
    local_row, local_col = local_pos // 3, local_pos % 3
    return (sub_row * 3 + local_row) * 9 + (sub_col * 3 + local_col)

def global_to_sub_grid(global_pos: int) -> Tuple[int, int]:
    """全体座標 → サブボード座標 + ローカル座標"""
    row, col = global_pos // 9, global_pos % 9
    sub_grid = (row // 3) * 3 + (col // 3)
    local_pos = (row % 3) * 3 + (col % 3)
    return sub_grid, local_pos
```

#### 合法手生成

```python
def get_legal_moves(self) -> List[int]:
    """現在の状態から合法手を生成"""
    if self.last_move is None:
        # 最初の手: 全位置が合法
        return [i for i in range(81) if self.board[i // 9, i % 9] == 0]
    
    # 前の手に対応するサブボードを取得
    target_sub_grid = self.last_move % 9
    
    # サブボードが埋まっている場合、任意の空いているサブボードに配置可能
    if self._is_sub_board_full(target_sub_grid):
        legal_moves = []
        for sub_grid in range(9):
            if not self._is_sub_board_full(sub_grid):
                legal_moves.extend(self._get_legal_moves_in_sub_board(sub_grid))
        return legal_moves
    
    # 指定されたサブボード内の空いている位置のみ
    return self._get_legal_moves_in_sub_board(target_sub_grid)
```

#### 勝敗判定

```python
def _check_meta_board_win(self) -> Optional[int]:
    """メタボードの勝敗を判定"""
    # 縦・横・斜めの3つ並びをチェック
    win_patterns = [
        # 横
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        # 縦
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        # 斜め
        [0, 4, 8], [2, 4, 6]
    ]
    
    for pattern in win_patterns:
        values = [self.meta_board[i // 3, i % 3] for i in pattern]
        if all(v == 1 for v in values):
            return 1
        if all(v == 2 for v in values):
            return 2
    return None
```

### 環境実装 (`env.py`)

#### Gymnasium準拠API

```python
class UltimateTicTacToeEnv(gym.Env):
    def __init__(self):
        self.board = UltimateTicTacToeBoard()
        
        # 観測空間: 9×9の整数配列 (0: 空, 1: X, 2: O)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(9, 9), dtype=np.int8
        )
        
        # 行動空間: 81次元離散空間
        self.action_space = gym.spaces.Discrete(81)
        
        # 環境状態
        self.current_player = 1
        self.terminated = False
        self.truncated = False
```

#### ステップ関数

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """環境を1ステップ進める"""
    # 不正手チェック
    if action not in self.board.get_legal_moves():
        return self._get_obs(), -100.0, True, False, {"invalid_move": True}
    
    # 行動実行
    self.board.make_move(action)
    
    # 状態更新
    self.current_player = 3 - self.current_player  # 1 ↔ 2
    self.terminated = self.board.game_over
    
    # 報酬計算
    reward = self._calculate_reward()
    
    return self._get_obs(), reward, self.terminated, self.truncated, {}
```

#### 観測生成

```python
def _get_obs(self) -> np.ndarray:
    """現在の盤面状態を観測配列として返す"""
    return self.board.board.copy()
```

#### 報酬設計

```python
def _calculate_reward(self) -> float:
    """終局時の報酬を計算"""
    if not self.terminated:
        return 0.0
    
    if self.board.winner == 1:
        return 1.0
    elif self.board.winner == 2:
        return -1.0
    else:
        return 0.0  # 引き分け
```

### エージェント実装 (`agents/random.py`)

#### ベースラインエージェント

```python
class RandomAgent:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def act(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """合法手からランダムに選択"""
        return self.rng.choice(legal_actions)
    
    def get_action_probs(self, obs: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """行動確率分布を生成"""
        probs = np.zeros(81)
        probs[legal_actions] = 1.0 / len(legal_actions)
        return probs
```

## テスト戦略

### テスト構造

```
tests/
├── test_board.py    # 盤面ロジックの単体テスト
└── test_env.py      # 環境APIの統合テスト
```

### テストカバレッジ

- **盤面ロジック**: 89%カバレッジ
- **環境API**: 93%カバレッジ
- **全体**: 70%カバレッジ

### 主要テストケース

#### 盤面テスト (`test_board.py`)

```python
class TestUltimateTicTacToeBoard:
    def test_legal_moves_first_move(self):
        """最初の手の合法手生成"""
        board = UltimateTicTacToeBoard()
        legal_moves = board.get_legal_moves()
        assert len(legal_moves) == 81
    
    def test_legal_moves_after_move(self):
        """手を打った後の合法手生成"""
        board = UltimateTicTacToeBoard()
        board.make_move(0)  # 左上に配置
        legal_moves = board.get_legal_moves()
        # 左上のサブボード内の空いている位置のみが合法
        assert all(move // 9 < 3 and move % 9 < 3 for move in legal_moves)
    
    def test_sub_board_win(self):
        """サブボードの勝敗判定"""
        board = UltimateTicTacToeBoard()
        # 左上のサブボードでXが勝利
        moves = [0, 9, 1, 10, 2]  # X: 0,1,2 / O: 9,10
        for move in moves:
            board.make_move(move)
        assert board.meta_board[0, 0] == 1
```

#### 環境テスト (`test_env.py`)

```python
class TestUltimateTicTacToeEnv:
    def test_step_valid_move(self):
        """有効な行動のステップ実行"""
        env = UltimateTicTacToeEnv()
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated
        assert reward == 0.0
    
    def test_step_invalid_move(self):
        """無効な行動の処理"""
        env = UltimateTicTacToeEnv()
        obs, info = env.reset()
        env.board.make_move(0)  # 位置0を埋める
        obs, reward, terminated, truncated, info = env.step(0)  # 同じ位置に配置
        assert terminated
        assert reward == -100.0
        assert info["invalid_move"]
```

## 開発ガイド

### 依存関係管理

```bash
# 新しい依存関係の追加
uv add package_name

# 開発用依存関係の追加
uv add --group dev package_name
```

### コード品質

```bash
# コードフォーマット
black src/ tests/

# 型チェック
mypy src/

# リント
flake8 src/ tests/

# テスト実行
python -m pytest tests/ -v --cov=utttrlsim
```

### 新しいエージェントの実装

```python
# agents/custom_agent.py
class CustomAgent:
    def __init__(self, **kwargs):
        # エージェントの初期化
        pass
    
    def act(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """観測と合法手から行動を選択"""
        # カスタムロジックを実装
        return legal_actions[0]
    
    def get_action_probs(self, obs: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """行動確率分布を生成（オプション）"""
        probs = np.zeros(81)
        probs[legal_actions] = 1.0 / len(legal_actions)
        return probs
```

### 環境の拡張

```python
# 新しい環境変数の追加
class UltimateTicTacToeEnv(gym.Env):
    def __init__(self, custom_param: str = "default"):
        super().__init__()
        self.custom_param = custom_param
        # 既存の初期化処理
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 既存のステップ処理
        info = {"custom_info": self.custom_param}
        return obs, reward, terminated, truncated, info
```

## パフォーマンス最適化

### 現在の実装

- **盤面表現**: numpy.int8配列（メモリ効率）
- **合法手生成**: キャッシュ機能なし（必要に応じて実装可能）
- **勝敗判定**: 線形探索（O(n)）

### 最適化候補

1. **ビットボード実装**: 64ビット整数での盤面表現
2. **合法手キャッシュ**: 状態ハッシュによるキャッシュ
3. **勝敗判定最適化**: 事前計算されたパターンテーブル
4. **Cython実装**: クリティカルパスの高速化

## 今後の拡張計画

### 短期目標

1. **MCTSエージェント**: UCB1選択とモンテカルロ評価
2. **強化学習統合**: stable-baselines3との互換性
3. **テスト充実**: プロパティテストと統合テスト

### 中期目標

1. **AlphaZero風実装**: MCTS + ニューラルネットワーク
2. **GUIインターフェース**: Tkinter/PyQt5ベース
3. **ネットワーク対戦**: WebSocket/HTTP API

### 長期目標

1. **分散学習**: マルチプロセス/マルチマシン学習
2. **教師あり学習**: 自己対戦データセット生成
3. **他ゲーム対応**: 汎用ボードゲームフレームワーク

## 技術的制約

### 現在の制限

- **Python 3.8+**: 型ヒントとf-stringの使用
- **numpy**: 数値計算の基盤
- **gymnasium**: RL環境の標準化

### 設計原則

1. **単一責任**: 各クラスは明確な責任を持つ
2. **型安全性**: 型ヒントによる静的解析
3. **テスト駆動**: 新機能はテストと共に実装
4. **ドキュメント**: 公開APIの完全なドキュメント化

## ライセンス

MIT License - 商用・学術用途での自由な利用が可能
