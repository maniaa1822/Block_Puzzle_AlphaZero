from __future__ import annotations

"""
Complete Block Puzzle Game Logic
Modern block puzzle game where players place 3 pieces at a time on a 9x9 grid
to clear complete rows and columns for points.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import IntEnum


@dataclass
class GameConfig:
    """Configuration for block puzzle game"""
    grid_size: int = 5
    max_episode_steps: int = 10000
    pieces_per_set: int = 3
    base_placement_points: int = 1
    line_clear_points: int = 10
    combo_multiplier: float = 2.0
    # Value normalization for AlphaZero terminal target
    value_norm_decay: float = 0.99  # EMA decay for running stats
    value_norm_min_std: float = 1.0  # floor on std to avoid divide-by-small
    value_norm_warmup: int = 10  # use fallback std until this many updates
    value_norm_fallback_std: float = 50.0  # std used during warmup
    value_tanh_scale: float = 1.0  # multiplier inside tanh for slope control


class PieceType(IntEnum):
    """Simplified piece set for 5x5: I(2), O(2x2), L(2x2-L)."""
    I = 0  # Line of length 2
    O = 1  # Square 2x2
    L = 2  # L-shape within 2x2 (3 cells)


class PieceShapes:
    """Static piece shape definitions (simplified)."""

    # Base shapes (rotation 0)
    SHAPES = {
        # Domino (1x2)
        PieceType.I: np.array([[1, 1]], dtype=int),
        # Square 2x2
        PieceType.O: np.array([[1, 1], [1, 1]], dtype=int),
        # L-shape in a 2x2 bounding box (3 cells)
        PieceType.L: np.array([[1, 0], [1, 1]], dtype=int),
    }

    @classmethod
    def get_shape(cls, piece_type: PieceType, rotation: int = 0) -> np.ndarray:
        """Get piece shape with specified rotation"""
        shape = cls.SHAPES[piece_type].copy()
        for _ in range(rotation % 4):
            shape = np.rot90(shape)
        return shape

    @classmethod
    def get_all_rotations(cls, piece_type: PieceType) -> List[np.ndarray]:
        """Get all unique rotations for a piece type"""
        rotations: List[np.ndarray] = []
        for r in range(4):
            shape = cls.get_shape(piece_type, r)
            # Check if this rotation is unique
            if not any(np.array_equal(shape, existing) for existing in rotations):
                rotations.append(shape)
        return rotations


class Piece:
    """Individual piece instance"""

    def __init__(self, piece_type: PieceType):
        self.piece_type = piece_type
        self.rotation = 0

    def get_shape(self, rotation: int | None = None) -> np.ndarray:
        """Get current piece shape at specified rotation"""
        if rotation is None:
            rotation = self.rotation
        return PieceShapes.get_shape(self.piece_type, rotation)

    def get_all_rotations(self) -> List[np.ndarray]:
        """Get all unique rotations of this piece"""
        return PieceShapes.get_all_rotations(self.piece_type)

    def get_bounding_box(self, rotation: int = 0) -> Tuple[int, int]:
        """Get (height, width) of piece at rotation"""
        shape = self.get_shape(rotation)
        return shape.shape


class GameGrid:
    """Game grid with placement and line clearing logic"""

    def __init__(self, size: int = 5):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

    def is_valid_placement(self, piece_shape: np.ndarray, x: int, y: int) -> bool:
        """Check if piece can be placed at position (x,y)"""
        piece_h, piece_w = piece_shape.shape

        # Check bounds
        if x < 0 or y < 0:
            return False
        if x + piece_w > self.size or y + piece_h > self.size:
            return False

        # Check for collision with existing blocks
        for py in range(piece_h):
            for px in range(piece_w):
                if piece_shape[py, px] and self.grid[y + py, x + px]:
                    return False

        return True

    def place_piece(self, piece_shape: np.ndarray, x: int, y: int) -> int:
        """
        Place piece on grid and return number of cells placed
        Assumes position is already validated
        """
        cells_placed = 0
        piece_h, piece_w = piece_shape.shape

        for py in range(piece_h):
            for px in range(piece_w):
                if piece_shape[py, px]:
                    self.grid[y + py, x + px] = 1
                    cells_placed += 1

        return cells_placed

    def clear_complete_lines(self) -> Tuple[int, int]:
        """
        Clear complete rows and columns
        Returns: (lines_cleared, cells_cleared)
        """
        lines_cleared = 0
        cells_cleared = 0

        # Clear complete rows
        rows_to_clear = [row for row in range(self.size) if np.all(self.grid[row, :])]
        for row in rows_to_clear:
            self.grid[row, :] = 0
            lines_cleared += 1
            cells_cleared += self.size

        # Clear complete columns
        cols_to_clear = [col for col in range(self.size) if np.all(self.grid[:, col])]
        for col in cols_to_clear:
            self.grid[:, col] = 0
            lines_cleared += 1
            cells_cleared += self.size

        return lines_cleared, cells_cleared

    def get_valid_placements(self, piece_shape: np.ndarray) -> List[Tuple[int, int]]:
        """Get all valid (x,y) positions for a piece shape"""
        valid_positions: List[Tuple[int, int]] = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_placement(piece_shape, x, y):
                    valid_positions.append((x, y))
        return valid_positions

    def get_filled_ratio(self) -> float:
        """Get percentage of grid that is filled"""
        return float(np.sum(self.grid)) / float(self.size * self.size)

    def copy(self) -> "GameGrid":
        """Create a copy of the current grid"""
        new_grid = GameGrid(self.size)
        new_grid.grid = self.grid.copy()
        return new_grid


class ScoreCalculator:
    """Handles scoring logic and reward calculation"""

    def __init__(self, config: GameConfig):
        self.base_placement_points = getattr(config, "base_placement_points", 1)
        self.line_clear_points = getattr(config, "line_clear_points", 10)
        self.combo_multiplier = getattr(config, "combo_multiplier", 2.0)

    def calculate_score(self, cells_placed: int, lines_cleared: int) -> int:
        """Calculate score for a single placement"""
        placement_score = int(cells_placed * self.base_placement_points)
        if lines_cleared == 0:
            return placement_score
        line_score = int(lines_cleared * self.line_clear_points)
        if lines_cleared > 1:
            combo_bonus = int(line_score * (self.combo_multiplier ** (lines_cleared - 1)))
            return placement_score + line_score + combo_bonus
        return placement_score + line_score


class ValueNormalizer:
    """EMA-based running score normalizer to produce bounded value targets.

    Computes z = (x - mean) / std, where mean and std are maintained via
    exponential moving averages. The final target is tanh(scale * z).
    """

    def __init__(
        self,
        decay: float = 0.99,
        min_std: float = 1.0,
        warmup: int = 10,
        fallback_std: float = 50.0,
        tanh_scale: float = 1.0,
    ):
        self.decay = float(decay)
        self.min_std = float(min_std)
        self.warmup = int(warmup)
        self.fallback_std = float(fallback_std)
        self.tanh_scale = float(tanh_scale)

        self.mean = 0.0
        self.mean_sq = 0.0
        self.count = 0

    def update(self, x: float) -> None:
        """Update running mean and variance using EMA on value x."""
        d = self.decay
        if self.count == 0:
            # Initialize directly to avoid long burn-in
            self.mean = float(x)
            self.mean_sq = float(x) * float(x)
            self.count = 1
            return
        prev_mean = self.mean
        self.mean = d * self.mean + (1.0 - d) * float(x)
        self.mean_sq = d * self.mean_sq + (1.0 - d) * float(x) * float(x)
        # Ensure numerical sanity if mean drifts
        if not np.isfinite(self.mean) or not np.isfinite(self.mean_sq):
            self.mean = prev_mean
            # Do not update mean_sq if instability occurs
        self.count += 1

    def _std(self) -> float:
        if self.count < self.warmup:
            return max(self.fallback_std, self.min_std)
        var = max(self.mean_sq - self.mean * self.mean, 0.0)
        std = float(np.sqrt(var))
        return max(std, self.min_std)

    def normalize(self, x: float) -> float:
        """Return z-score of x under running stats with floors/warmup."""
        std = self._std()
        return (float(x) - self.mean) / std

    def value_target(self, x: float) -> float:
        """Bounded target in [-1, 1] via tanh(scale * z)."""
        z = self.normalize(x)
        return float(np.tanh(self.tanh_scale * z))

    def get_stats(self) -> dict:
        std = self._std()
        return {"mean": float(self.mean), "std": float(std), "count": int(self.count)}


class BlockPuzzleGame:
    """Main game engine for block puzzle"""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.grid = GameGrid(self.config.grid_size)
        self.score_calculator = ScoreCalculator(self.config)
        self.value_normalizer = ValueNormalizer(
            decay=self.config.value_norm_decay,
            min_std=self.config.value_norm_min_std,
            warmup=self.config.value_norm_warmup,
            fallback_std=self.config.value_norm_fallback_std,
            tanh_scale=self.config.value_tanh_scale,
        )

        # Game state
        self.current_pieces: List[Piece] = []
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.step_count = 0
        self.game_over = False
        self._pending_final_score: Optional[float] = None

        self.generate_new_piece_set()

    def generate_new_piece_set(self) -> None:
        """Generate a new set: one each of I(2), O(2x2), L(2x2-L), shuffled."""
        base_set = [Piece(PieceType.I), Piece(PieceType.O), Piece(PieceType.L)]
        # Shuffle order
        np.random.shuffle(base_set)
        # Respect pieces_per_set if changed (<=3)
        self.current_pieces = base_set[: max(0, min(self.config.pieces_per_set, len(base_set)))]

    def get_current_piece_types(self) -> List[int]:
        return [piece.piece_type.value for piece in self.current_pieces]

    def can_place_any_piece(self) -> bool:
        for piece in self.current_pieces:
            for rotation in range(4):
                shape = piece.get_shape(rotation)
                if len(self.grid.get_valid_placements(shape)) > 0:
                    return True
        return False

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """List of (piece_idx, x, y, rotation) valid actions"""
        actions: List[Tuple[int, int, int, int]] = []
        for piece_idx, piece in enumerate(self.current_pieces):
            for rotation in range(4):
                shape = piece.get_shape(rotation)
                for x, y in self.grid.get_valid_placements(shape):
                    actions.append((piece_idx, x, y, rotation))
        return actions

    def place_piece(self, piece_idx: int, x: int, y: int, rotation: int = 0) -> Tuple[bool, int, int]:
        if piece_idx < 0 or piece_idx >= len(self.current_pieces):
            return False, 0, 0
        piece = self.current_pieces[piece_idx]
        shape = piece.get_shape(rotation)
        if not self.grid.is_valid_placement(shape, x, y):
            return False, 0, 0
        cells_placed = self.grid.place_piece(shape, x, y)
        lines_cleared, _ = self.grid.clear_complete_lines()
        gained = self.score_calculator.calculate_score(cells_placed, lines_cleared)
        self.score += gained
        self.total_pieces_placed += 1
        self.total_lines_cleared += lines_cleared
        self.step_count += 1
        self.current_pieces.pop(piece_idx)
        if len(self.current_pieces) == 0:
            self.generate_new_piece_set()
        if not self.can_place_any_piece():
            self.game_over = True
            # Defer baseline update until reset so targets can be read first
            self._pending_final_score = float(self.score)
        return True, gained, lines_cleared

    def simulate_placement(self, piece_idx: int, x: int, y: int, rotation: int = 0) -> Tuple[bool, int, int]:
        if piece_idx < 0 or piece_idx >= len(self.current_pieces):
            return False, 0, 0
        piece = self.current_pieces[piece_idx]
        shape = piece.get_shape(rotation)
        if not self.grid.is_valid_placement(shape, x, y):
            return False, 0, 0
        temp_grid = self.grid.copy()
        cells_placed = temp_grid.place_piece(shape, x, y)
        lines_cleared, _ = temp_grid.clear_complete_lines()
        potential = self.score_calculator.calculate_score(cells_placed, lines_cleared)
        return True, potential, lines_cleared

    def get_state(self) -> dict:
        return {
            "grid": self.grid.grid.copy(),
            "current_pieces": self.get_current_piece_types(),
            "pieces_remaining": len(self.current_pieces),
            "score": self.score,
            "total_lines_cleared": self.total_lines_cleared,
            "total_pieces_placed": self.total_pieces_placed,
            "step_count": self.step_count,
            "game_over": self.game_over,
            "filled_ratio": self.grid.get_filled_ratio(),
            "value_baseline": self.value_normalizer.get_stats(),
        }

    def get_value_target(self) -> float:
        """Return AlphaZero-style value target for the current state.

        - If the game is ongoing, returns 0.0 so training uses network-predicted value.
        - If terminal, returns tanh-normalized final score in [-1, 1].
        """
        if not self.game_over:
            return 0.0
        return self.value_normalizer.value_target(self.score)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        # If previous episode ended and we haven't updated stats yet, do it now
        if self._pending_final_score is not None:
            self.value_normalizer.update(self._pending_final_score)
            self._pending_final_score = None
        self.grid = GameGrid(self.config.grid_size)
        self.current_pieces = []
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.step_count = 0
        self.game_over = False
        self.generate_new_piece_set()

    def get_game_stats(self) -> dict:
        return {
            "final_score": self.score,
            "pieces_placed": self.total_pieces_placed,
            "lines_cleared": self.total_lines_cleared,
            "steps_taken": self.step_count,
            "final_fill_ratio": self.grid.get_filled_ratio(),
            "avg_score_per_piece": self.score / max(1, self.total_pieces_placed),
            "avg_lines_per_piece": self.total_lines_cleared / max(1, self.total_pieces_placed),
        }


class GameAnalytics:
    """Helper class for analyzing game states and positions"""

    @staticmethod
    def get_board_features(grid: np.ndarray) -> dict:
        size = grid.shape[0]
        height_map: List[int] = []
        for col in range(size):
            height = 0
            for row in range(size):
                if grid[row, col] == 1:
                    height = size - row
                    break
            height_map.append(height)
        holes = 0
        for col in range(size):
            found_block = False
            for row in range(size):
                if grid[row, col] == 1:
                    found_block = True
                elif found_block and grid[row, col] == 0:
                    holes += 1
        bumpiness = sum(abs(height_map[i] - height_map[i + 1]) for i in range(size - 1))
        almost_complete_rows = sum(1 for row in range(size) if np.sum(grid[row, :]) >= size - 1)
        almost_complete_cols = sum(1 for col in range(size) if np.sum(grid[:, col]) >= size - 1)
        return {
            "max_height": max(height_map) if height_map else 0,
            "avg_height": float(np.mean(height_map)) if height_map else 0.0,
            "holes": holes,
            "bumpiness": bumpiness,
            "filled_cells": int(np.sum(grid)),
            "fill_ratio": float(np.sum(grid)) / float(size * size),
            "almost_complete_lines": almost_complete_rows + almost_complete_cols,
            "empty_rows": sum(1 for row in range(size) if np.sum(grid[row, :]) == 0),
            "empty_cols": sum(1 for col in range(size) if np.sum(grid[:, col]) == 0),
        }

    @staticmethod
    def evaluate_placement_quality(grid: GameGrid, piece_shape: np.ndarray, x: int, y: int) -> dict:
        if not grid.is_valid_placement(piece_shape, x, y):
            return {"valid": False}
        temp_grid = grid.copy()
        cells_placed = temp_grid.place_piece(piece_shape, x, y)
        lines_cleared, _ = temp_grid.clear_complete_lines()
        features_before = GameAnalytics.get_board_features(grid.grid)
        features_after = GameAnalytics.get_board_features(temp_grid.grid)
        # Mobility after: count legal placements for all simplified piece types (unique rotations)
        mobility = 0
        for pt in PieceType:
            for rot_shape in PieceShapes.get_all_rotations(pt):
                mobility += len(temp_grid.get_valid_placements(rot_shape))
        # Count 2x2 empty windows after
        windows2x2 = 0
        size = temp_grid.size
        for yy in range(size - 1):
            for xx in range(size - 1):
                block = temp_grid.grid[yy : yy + 2, xx : xx + 2]
                if np.sum(block) == 0:
                    windows2x2 += 1
        # Count interior single-cell zeros that are 4-neighbor surrounded by filled cells
        singletons = 0
        for yy in range(1, size - 1):
            for xx in range(1, size - 1):
                if temp_grid.grid[yy, xx] == 0:
                    if (
                        temp_grid.grid[yy - 1, xx] == 1
                        and temp_grid.grid[yy + 1, xx] == 1
                        and temp_grid.grid[yy, xx - 1] == 1
                        and temp_grid.grid[yy, xx + 1] == 1
                    ):
                        singletons += 1
        return {
            "valid": True,
            "cells_placed": cells_placed,
            "lines_cleared": lines_cleared,
            "height_increase": features_after["max_height"] - features_before["max_height"],
            "holes_created": features_after["holes"] - features_before["holes"],
            "bumpiness_change": features_after["bumpiness"] - features_before["bumpiness"],
            "almost_complete_lines_after": features_after["almost_complete_lines"],
            "mobility_after": mobility,
            "windows2x2_after": windows2x2,
            "singletons_after": singletons,
        }


def print_grid(grid: np.ndarray) -> None:
    for row in grid:
        print("".join(["█" if cell else "·" for cell in row]))


def print_piece(piece_shape: np.ndarray) -> None:
    for row in piece_shape:
        print("".join(["█" if cell else "·" for cell in row]))


def run_game_demo() -> None:  # pragma: no cover
    game = BlockPuzzleGame()
    print("=== Block Puzzle Game Demo ===")
    print(f"Initial pieces: {game.get_current_piece_types()}")
    print("\nInitial grid:")
    print_grid(game.grid.grid)
    if len(game.current_pieces) > 0:
        piece = game.current_pieces[0]
        shape = piece.get_shape(0)
        print(f"\nPiece 0 shape:")
        print_piece(shape)
        success, score_gained, lines_cleared = game.place_piece(0, 0, 0, 0)
        if success:
            print(f"\nPlaced piece! Score gained: {score_gained}, Lines cleared: {lines_cleared}")
            print("Grid after placement:")
            print_grid(game.grid.grid)
            print(f"Remaining pieces: {game.get_current_piece_types()}")
            print(f"Total score: {game.score}")
        else:
            print("Could not place piece at (0,0)")
    valid_actions = game.get_valid_actions()
    print(f"\nTotal valid actions available: {len(valid_actions)}")
    if valid_actions:
        print(f"Example valid action: {valid_actions[0]} (piece_idx, x, y, rotation)")


if __name__ == "__main__":  # pragma: no cover
    run_game_demo()
    print("\n=== Piece Rotation Test ===")
    for piece_type in PieceType:
        print(f"\n{piece_type.name} piece rotations:")
        rotations = PieceShapes.get_all_rotations(piece_type)
        for i, rotation in enumerate(rotations):
            print(f"Rotation {i}:")
            print_piece(rotation)



