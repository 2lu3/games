"""
Ultimate Tic-Tac-Toe Board Implementation

Core game logic including board representation, legal move generation,
and win condition checking.

Ultimate Tic-Tac-Toe Rules:
1. The game is played on a 9x9 board divided into 9 3x3 sub-boards
2. Players take turns placing X or O in empty cells
3. The first player can place their mark anywhere
4. Each subsequent move must be placed in the sub-board corresponding to the cell position of the opponent's last move
5. If the target sub-board is full or already won, the player can choose any available sub-board
6. To win a sub-board, a player must get 3 in a row (horizontally, vertically, or diagonally)
7. To win the game, a player must win 3 sub-boards in a row (horizontally, vertically, or diagonally)
8. If all sub-boards are full or won and no player has won 3 in a row, the game is a draw
"""

from enum import Enum
from typing import List, Optional, Set, Tuple, Union
import itertools
import numpy as np


class Player(Enum):
    """Player enumeration for Ultimate Tic-Tac-Toe"""

    EMPTY = 0
    X = 1
    O = 2


class Position:
    """
    Position class for Ultimate Tic-Tac-Toe board coordinates.

    Supports two initialization methods:
    - Position(global_id): 0-80 for global position
    - Position(grid_x, grid_y, cell_x, cell_y): Grid and cell coordinates
        - grid_x, grid_y: Sub-grid coordinates (0-2)
        - cell_x, cell_y: Cell coordinates within sub-grid (0-2)
    """

    def __init__(self, *args):
        """
        Initialize position.

        Args:
            Either:
            - board_id (int): Global position ID (0-80)
            - grid_x, grid_y, cell_x, cell_y (int, int, int, int): Grid and cell coordinates
                - grid_x, grid_y: Sub-grid coordinates (0-2)
                - cell_x, cell_y: Cell coordinates within sub-grid (0-2)
        """
        if len(args) == 1:
            # Initialize from global ID
            board_id = args[0]
            assert 0 <= board_id < 81, f"Position ID must be 0-80, got {board_id}"
            self._id = board_id
        elif len(args) == 4:
            # Initialize from grid and cell coordinates
            grid_x, grid_y, cell_x, cell_y = args
            assert 0 <= grid_x < 3, f"Grid X coordinate must be 0-2, got {grid_x}"
            assert 0 <= grid_y < 3, f"Grid Y coordinate must be 0-2, got {grid_y}"
            assert 0 <= cell_x < 3, f"Cell X coordinate must be 0-2, got {cell_x}"
            assert 0 <= cell_y < 3, f"Cell Y coordinate must be 0-2, got {cell_y}"
            self._id = (grid_x * 3 + cell_x) + (grid_y * 3 + cell_y) * 9
        else:
            raise ValueError(
                "Position requires either 1 argument (global_id) or 4 arguments (grid_x, grid_y, cell_x, cell_y)"
            )

    @property
    def board_id(self) -> int:
        """Board position ID (0-80)."""
        return self._id

    @property
    def board_x(self) -> int:
        """Board X coordinate (0-8) - position in the 9x9 main board."""
        return self._id % 9

    @property
    def board_y(self) -> int:
        """Board Y coordinate (0-8) - position in the 9x9 main board."""
        return self._id // 9

    @property
    def sub_grid_x(self) -> int:
        """Sub-grid X coordinate (0-2) - which 3x3 sub-grid horizontally."""
        return self.board_x // 3

    @property
    def sub_grid_y(self) -> int:
        """Sub-grid Y coordinate (0-2) - which 3x3 sub-grid vertically."""
        return self.board_y // 3

    @property
    def sub_grid_id(self) -> int:
        """Sub-grid ID (0-8) - unique identifier for the 3x3 sub-grid."""
        return self.sub_grid_x + self.sub_grid_y * 3

    @property
    def cell_x(self) -> int:
        """Cell X coordinate within sub-grid (0-2) - position within the 3x3 sub-grid."""
        return self.board_x % 3

    @property
    def cell_y(self) -> int:
        """Cell Y coordinate within sub-grid (0-2) - position within the 3x3 sub-grid."""
        return self.board_y % 3

    @property
    def cell_id(self) -> int:
        """Cell ID within sub-grid (0-8)."""
        return self.cell_x + self.cell_y * 3

    def __eq__(self, other):
        if isinstance(other, Position):
            return self._id == other._id
        return False

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        return f"Position(board_id={self._id}, board=({self.board_x}, {self.board_y}), sub_grid=({self.sub_grid_x}, {self.sub_grid_y}), cell=({self.cell_x}, {self.cell_y}))"


class UltimateTicTacToeBoard:
    """
    Ultimate Tic-Tac-Toe board implementation.

    Board representation: 9x9 numpy array where:
    - 0: Empty
    - 1: Player X
    - 2: Player O

    The board is divided into 9 3x3 sub-boards, each identified by grid coordinates (0-2, 0-2).
    """

    def __init__(
        self,
        board: Optional[np.ndarray] = None,
        current_player: Player = Player.X,
        last_move: Optional[Position] = None,
    ):
        """Initialize an empty Ultimate Tic-Tac-Toe board."""
        # Main board: 9x9 grid
        if board is None:
            board = np.full((9, 9), Player.EMPTY.value, dtype=np.int8)
        self._board: np.ndarray = board

        # Current player (1 for X, 2 for O)
        self._current_player: Player = current_player

        # Last move made (Position object)
        self._last_move: Optional[Position] = last_move

    @property
    def board(self) -> Tuple[Tuple[Position, Player], ...]:
        """
        Get board state as tuple of (Position, Player) tuples for non-empty cells.
        
        Returns:
            Tuple of (Position, Player) tuples representing occupied cells
        """
        occupied_cells = []
        for y in range(9):
            for x in range(9):
                cell_value = self._board[y, x]
                if cell_value != Player.EMPTY.value:
                    position = Position(x + y * 9)
                    player = Player(cell_value)
                    occupied_cells.append((position, player))
        return tuple(occupied_cells)

    @property
    def current_player(self) -> Player:
        """Get the current player."""
        return self._current_player

    @property
    def last_move(self) -> Optional[Position]:
        """Get the last move made."""
        return self._last_move

    def make_move(self, position: Position) -> None:
        """
        Make a move on the board.

        Args:
            position: Position object representing the move

        Raises:
            RuntimeError: If the game is already over
            ValueError: If the move is not legal
        """
        # Validate move
        if self.game_over:
            raise RuntimeError("Cannot make move: game is already over")

        legal_moves = self.get_legal_moves()
        if position not in legal_moves:
            raise ValueError(
                f"Invalid move: position {position} is not in legal moves. "
                f"Legal moves: {legal_moves}"
            )

        # Make the move
        self._board[position.board_y, position.board_x] = self._current_player.value

        # Update last move
        self._last_move = position

        # Switch players
        self._current_player = Player.O if self._current_player == Player.X else Player.X

    def get_legal_moves(self) -> List[Position]:
        """
        Get all legal moves for the current player.

        Ultimate Tic-Tac-Toe rules for legal moves:
        1. First move: can play anywhere
        2. Subsequent moves: must play in the sub-board corresponding to the cell position of the last move
        3. If the target sub-board is full or already won, can play in any available sub-board

        Returns:
            List of Position objects representing legal moves.
        """
        legal_moves = set()

        if self._last_move is None:
            # First move: can play anywhere
            # Assert that board is completely empty for first move
            assert np.all(self._board == 0), "Board must be empty for first move"

            # All 81 positions are legal for first move
            for pos_id in range(81):
                pos = Position(pos_id)
                legal_moves.add(pos)
        else:
            # Subsequent moves: must play in the sub-board corresponding to last move's cell position
            target_sub_grid_x = self._last_move.cell_x
            target_sub_grid_y = self._last_move.cell_y

            # Check if target sub-board is available (not won and not full)
            target_sub_board_available = self.subboard_winner[
                target_sub_grid_y, target_sub_grid_x
            ] == Player.EMPTY.value and not self._is_sub_board_full(
                target_sub_grid_x, target_sub_grid_y
            )

            if target_sub_board_available:
                # Must play in target sub-board
                for cell_id in range(9):
                    cell = Position(
                        target_sub_grid_x, target_sub_grid_y, cell_id % 3, cell_id // 3
                    )
                    if self._board[cell.board_y, cell.board_x] == Player.EMPTY.value:
                        legal_moves.add(cell)
            else:
                # Target sub-board is won or full, can play in any available sub-board
                for sub_grid_x, sub_grid_y in itertools.product(range(3), range(3)):
                    # Skip sub-boards that are won or full
                    if self.subboard_winner[
                        sub_grid_y, sub_grid_x
                    ] != Player.EMPTY.value or self._is_sub_board_full(
                        sub_grid_x, sub_grid_y
                    ):
                        continue

                    # Add all empty cells in this available sub-board
                    for cell_id in range(9):
                        cell = Position(
                            sub_grid_x, sub_grid_y, cell_id % 3, cell_id // 3
                        )
                        if self._board[cell.board_y, cell.board_x] == Player.EMPTY.value:
                            legal_moves.add(cell)

        return list(legal_moves)

    def reset(self, current_player: Player = Player.X) -> None:
        """Reset the board to initial state."""
        self._board.fill(Player.EMPTY.value)
        self._current_player = Player.X
        self._last_move = None

    def render(self) -> str:
        """
        Render the board as a string for display.

        Returns:
            String representation of the board with sub-boards separated by lines
        """
        result = []

        for meta_row in range(3):
            for sub_row in range(3):
                line = ""
                for meta_col in range(3):
                    for sub_col in range(3):
                        pos = Position(meta_col, meta_row, sub_col, sub_row)
                        cell = self._board[pos.board_y, pos.board_x]

                        if cell == Player.EMPTY.value:
                            line += "."
                        elif cell == Player.X.value:
                            line += "X"
                        else:
                            line += "O"

                        if sub_col < 2:
                            line += " "

                    if meta_col < 2:
                        line += " | "

                result.append(line)

            if meta_row < 2:
                result.append("-" * 23)

        return "\n".join(result)

    def copy(self) -> "UltimateTicTacToeBoard":
        """Create a deep copy of the board."""
        new_board = UltimateTicTacToeBoard()
        new_board._board = self._board.copy()
        new_board._current_player = self._current_player
        new_board._last_move = self._last_move  # Position objects are immutable
        return new_board

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current board state.

        Returns:
            Tuple of (main_board, meta_board) where:
            - main_board: 9x9 array representing the full board
            - meta_board: 3x3 array representing which sub-boards are won
        """
        return self._board.copy(), self.subboard_winner.copy()

    @property
    def game_over(self) -> bool:
        """
        Check if the game is over.

        Game ends when:
        1. A player wins 3 sub-boards in a row (horizontally, vertically, or diagonally)
        2. All sub-boards are either won or full (draw)
        """
        # Check if either player has a winning line on the meta-board
        for player in (Player.X, Player.O):
            if self._check_win_pattern_for_player(self.subboard_winner, player):
                return True

        # Check for draw: all sub-boards must be either won or full
        for i in range(3):
            for j in range(3):
                # If any sub-board is not won and not full, game is not over
                if self.subboard_winner[
                    i, j
                ] == Player.EMPTY.value and not self._is_sub_board_full(j, i):
                    return False

        # All sub-boards are either won or full → draw
        return True

    @property
    def winner(self) -> Player:
        """
        Return the winner, or Player.EMPTY if draw or game not finished.

        Returns:
            Player.X or Player.O if that player has won, Player.EMPTY otherwise
        """
        # If the game is not yet over there is no winner
        if not self.game_over:
            return Player.EMPTY

        # Check which player (if any) owns a winning line on the meta-board
        for player in (Player.X, Player.O):
            if self._check_win_pattern_for_player(self.subboard_winner, player):
                return player

        # No player has a winning line → draw
        return Player.EMPTY

    @property
    def subboard_winner(self) -> np.ndarray:
        """
        Get the current sub-board winners.

        Returns:
            3x3 array where:
            - Player.EMPTY.value: Sub-board not yet won
            - Player.X.value: Player X won this sub-board
            - Player.O.value: Player O won this sub-board
        """
        result = np.full((3, 3), Player.EMPTY.value, dtype=np.int8)

        for grid_x, grid_y in itertools.product(range(3), range(3)):
            # Extract sub-board data using numpy slicing
            start_y = grid_y * 3
            start_x = grid_x * 3
            sub_board_data = self._board[start_y : start_y + 3, start_x : start_x + 3]

            # Check if X won this sub-board
            if self._check_win_pattern_for_player(sub_board_data, Player.X):
                result[grid_y, grid_x] = Player.X.value
            # Check if O won this sub-board
            elif self._check_win_pattern_for_player(sub_board_data, Player.O):
                result[grid_y, grid_x] = Player.O.value
            # Otherwise, sub-board is not yet won (result remains Player.EMPTY.value)

        return result

    def _is_sub_board_full(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if a sub-board is full (no empty cells).

        Args:
            grid_x, grid_y: Sub-grid coordinates (0-2)

        Returns:
            True if the sub-board has no empty cells, False otherwise
        """
        start_y = grid_y * 3
        start_x = grid_x * 3
        sub_board_data = self._board[start_y : start_y + 3, start_x : start_x + 3]
        return not np.any(sub_board_data == Player.EMPTY.value)

    def _check_win_pattern_for_player(
        self, board_slice: np.ndarray, player: Player
    ) -> bool:
        """
        Check if a specific player has a winning pattern on a 3x3 board slice.

        Args:
            board_slice: 3x3 numpy array to check
            player: Player to check for winning pattern

        Returns:
            True if the player has 3 in a row (horizontally, vertically, or diagonally)
        """
        player_value = player.value

        # Check rows
        for row in range(3):
            if np.all(board_slice[row, :] == player_value):
                return True

        # Check columns
        for col in range(3):
            if np.all(board_slice[:, col] == player_value):
                return True

        # Check diagonals
        if np.all(np.diag(board_slice) == player_value):
            return True
        if np.all(np.diag(np.fliplr(board_slice)) == player_value):
            return True

        return False
