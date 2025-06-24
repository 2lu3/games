強化学習プロジェクト計画書

Gymnasium × Stable Baselines3 で対戦型ゲーム AI を実装する

⸻

1. 目的（Why）
	•	**二人対戦ボードゲーム（例：Ultimate Tic-Tac-Toe）**を題材に、
	•	自己対戦（Self-Play）で学習するエージェントを作成
	•	Stable Baselines3（以下 SB3）の最新アルゴリズムと gymnasium 環境を活用
	•	行動マスク（無効手の排除）や並列学習でサンプル効率を高める

⸻

2. 技術スタック & 前提
	•	Python 3.11 以上
	•	gymnasium ≥ 0.29 - 自作環境を実装
	•	stable-baselines3 ≥ 2.7（PyTorch backend）
	•	sb3-contrib - MaskablePPO を使用し，行動マスクに公式対応  ￼
	•	TensorBoard／Weights & Biases でロギング
	•	OS は macOS（Apple Silicon）を想定

⸻

3. タスク一覧（What / How）
	1.	ゲーム仕様決定 & データ構造設計
	•	盤面クラス（Board）と位置クラス（Position）をシンプルに分離
	•	numpy 配列で状態を持たせ，合法手一覧を返すメソッドを用意
	2.	Gymnasium 環境 (UltimateTicTacToeEnv) 実装
	•	observation_space: 盤面 + 次に置ける小区画などをエンコード
	•	action_space: spaces.Discrete(N) もしくは spaces.MultiDiscrete
	•	行動マスク: info["action_mask"] に np.bool_ 配列を返却
	•	SB3 Contrib の MaskablePPO では env.get_action_mask() を内部で呼び出す仕組みに沿う ￼
	•	Opponent Policy は最初は RandomAgent、後に自己対戦へ
	3.	自己対戦ラッパー SelfPlayWrapper
	•	Step 前に 50 % の確率で観測を鏡像変換し，対称性を活用
	•	Opponent 更新コールバック
	•	N 歩ごとに現在ポリシーをスナップショット → Opponent 側へコピー
	•	SB3 の Callback API で実装（on_rollout_end フック使用） ￼
	4.	学習スクリプト (train.py)
	•	MaskablePPO("MlpPolicy", env, ...)
	•	SubprocVecEnv で並列 8 環境
	•	ハイパーパラメータは
	•	γ = 0.99, λ = 0.95, lr = 2.5e-4, clip range = 0.2
	•	進捗は TensorBoard で可視化
	5.	評価スクリプト (evaluate.py)
	•	ランダム／過去世代／手動プレーヤーと 100 試合
	•	勝率・平均手数・Elo レーティング推定
	6.	CI / テスト
	•	pytest + pytest-cov
	•	ボードロジックと合法手マスクのユニットテストを 100 % カバー
	•	GitHub Actions で自動テスト
	7.	デプロイ & インタラクティブ対戦
	•	FastAPI + Vue で WebUI
	•	推論専用 Policy.predict(obs, deterministic=True) をエンドポイントに
	8.	拡張ロードマップ（任意）
	•	Curriculum Learning（盤面サイズ段階的拡大）
	•	AlphaZero-style Monte-Carlo Tree Search とのハイブリッド
	•	盤面画像入力（CNN ポリシー）への置き換え

⸻

4. 成功判定（Definition of Done）
	•	自己対戦により 勝率 ≥ 70 %（RandomAgent 比）を 3 日間の学習で達成
	•	行動マスク実装により 無効手選択率 0 %
	•	主要モジュールの テストカバレッジ 90 % 以上

⸻

5. 参考リンク
	•	Stable Baselines3 – Maskable PPO ドキュメント  ￼
	•	SB3 コールバックガイド  ￼

⸻

🔥 一歩ずつ組み上げれば，学習ログが伸びるたびにボードで強くなる AI が見えてきます。楽しみながら実装を進めていきましょう！