#!/usr/bin/env python3
"""
Ultimate Tic-Tac-Toe GUI using Pygame

A simple graphical interface for playing Ultimate Tic-Tac-Toe.
"""

import pathlib
import sys

import numpy as np
import pygame

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utttrlsim import env_registration
from utttrlsim.board import Player, Position
from utttrlsim.env import UltimateTicTacToeEnv


class UltimateTicTacToeGUI:
    """Ultimate Tic-Tac-ToeのGUIクラス"""

    def __init__(self, window_size=(800, 600)):
        """GUIを初期化"""
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Ultimate Tic-Tac-Toe")

        # 色の定義
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)

        # フォント
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # 環境の初期化
        self.env = UltimateTicTacToeEnv()
        self.observation, self.info = self.env.reset()

        # ボードの描画設定
        self.board_margin = 50
        self.board_size = min(window_size) - 2 * self.board_margin
        self.cell_size = self.board_size // 9
        self.sub_board_size = self.cell_size * 3

    def get_cell_from_pos(self, pos):
        """マウス位置からセル座標を取得"""
        x, y = pos
        if (
            self.board_margin <= x <= self.board_margin + self.board_size
            and self.board_margin <= y <= self.board_margin + self.board_size
        ):
            cell_x = (x - self.board_margin) // self.cell_size
            cell_y = (y - self.board_margin) // self.cell_size
            if 0 <= cell_x < 9 and 0 <= cell_y < 9:
                return cell_x, cell_y
        return None

    def draw_board(self):
        """ボードを描画"""
        # 背景
        self.screen.fill(self.WHITE)

        # メインボードの枠
        pygame.draw.rect(
            self.screen,
            self.BLACK,
            (self.board_margin, self.board_margin, self.board_size, self.board_size),
            3,
        )

        # サブボードの枠（太い線）
        for i in range(1, 3):
            # 縦線
            x = self.board_margin + i * self.sub_board_size
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (x, self.board_margin),
                (x, self.board_margin + self.board_size),
                3,
            )
            # 横線
            y = self.board_margin + i * self.sub_board_size
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (self.board_margin, y),
                (self.board_margin + self.board_size, y),
                3,
            )

        # セルの枠（細い線）
        for i in range(1, 9):
            if i % 3 != 0:  # サブボードの境界は描画しない
                # 縦線
                x = self.board_margin + i * self.cell_size
                pygame.draw.line(
                    self.screen,
                    self.GRAY,
                    (x, self.board_margin),
                    (x, self.board_margin + self.board_size),
                    1,
                )
                # 横線
                y = self.board_margin + i * self.cell_size
                pygame.draw.line(
                    self.screen,
                    self.GRAY,
                    (self.board_margin, y),
                    (self.board_margin + self.board_size, y),
                    1,
                )

        # セルの内容を描画
        board = self.observation["board"]
        action_mask = self.observation["action_mask"]

        for y in range(9):
            for x in range(9):
                cell_x = self.board_margin + x * self.cell_size
                cell_y = self.board_margin + y * self.cell_size

                # 合法手のハイライト
                action_id = y * 9 + x
                if action_mask[action_id] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.YELLOW,
                        (
                            cell_x + 2,
                            cell_y + 2,
                            self.cell_size - 4,
                            self.cell_size - 4,
                        ),
                    )

                # セルの内容
                cell_value = board[y, x]
                if cell_value == Player.X.value:
                    # Xを描画
                    text = self.font.render("X", True, self.RED)
                    text_rect = text.get_rect(
                        center=(
                            cell_x + self.cell_size // 2,
                            cell_y + self.cell_size // 2,
                        )
                    )
                    self.screen.blit(text, text_rect)
                elif cell_value == Player.O.value:
                    # Oを描画
                    text = self.font.render("O", True, self.BLUE)
                    text_rect = text.get_rect(
                        center=(
                            cell_x + self.cell_size // 2,
                            cell_y + self.cell_size // 2,
                        )
                    )
                    self.screen.blit(text, text_rect)

        # サブボードの勝者を表示
        meta_board = self.info["meta_board"]
        for sub_y in range(3):
            for sub_x in range(3):
                if meta_board[sub_y, sub_x] != Player.EMPTY.value:
                    # サブボード全体を塗りつぶし
                    start_x = self.board_margin + sub_x * self.sub_board_size
                    start_y = self.board_margin + sub_y * self.sub_board_size

                    color = (
                        self.RED
                        if meta_board[sub_y, sub_x] == Player.X.value
                        else self.BLUE
                    )
                    s = pygame.Surface((self.sub_board_size, self.sub_board_size))
                    s.set_alpha(100)
                    s.fill(color)
                    self.screen.blit(s, (start_x, start_y))

                    # 勝者の文字を表示
                    winner_text = (
                        "X" if meta_board[sub_y, sub_x] == Player.X.value else "O"
                    )
                    text = self.font.render(winner_text, True, self.BLACK)
                    text_rect = text.get_rect(
                        center=(
                            start_x + self.sub_board_size // 2,
                            start_y + self.sub_board_size // 2,
                        )
                    )
                    self.screen.blit(text, text_rect)

    def draw_info(self):
        """ゲーム情報を描画"""
        # 現在のプレイヤー
        current_player = "X" if self.info["current_player"] == Player.X.value else "O"
        player_text = f"現在のプレイヤー: {current_player}"
        text = self.font.render(player_text, True, self.BLACK)
        self.screen.blit(text, (20, 20))

        # ゲーム状態
        if self.info["game_over"]:
            if self.info["winner"] is not None:
                winner = "X" if self.info["winner"] == Player.X.value else "O"
                status_text = f"ゲーム終了! 勝者: {winner}"
                color = self.RED if winner == "X" else self.BLUE
            else:
                status_text = "引き分け!"
                color = self.GRAY
        else:
            status_text = "ゲーム進行中"
            color = self.BLACK

        status_surface = self.font.render(status_text, True, color)
        self.screen.blit(status_surface, (20, 60))

        # 操作説明
        help_text = "クリックして手を打つ | R: リセット | Q: 終了"
        help_surface = self.small_font.render(help_text, True, self.GRAY)
        self.screen.blit(help_surface, (20, self.window_size[1] - 30))

    def handle_click(self, pos):
        """マウスクリックを処理"""
        if self.info["game_over"]:
            return

        cell = self.get_cell_from_pos(pos)
        if cell is None:
            return

        x, y = cell
        action = y * 9 + x

        # 合法手かチェック
        if self.observation["action_mask"][action] == 1:
            # 手を実行
            self.observation, reward, terminated, truncated, self.info = self.env.step(
                action
            )

    def reset_game(self):
        """ゲームをリセット"""
        self.observation, self.info = self.env.reset()

    def run(self):
        """メインループ"""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左クリック
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Rキーでリセット
                        self.reset_game()
                    elif event.key == pygame.K_q:  # Qキーで終了
                        running = False

            # 描画
            self.draw_board()
            self.draw_info()
            pygame.display.flip()

            clock.tick(60)

        pygame.quit()


def main():
    """メイン関数"""
    gui = UltimateTicTacToeGUI()
    gui.run()


if __name__ == "__main__":
    main()
