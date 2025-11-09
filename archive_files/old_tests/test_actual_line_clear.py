#!/usr/bin/env python3
"""Deterministically verify that a scripted policy can clear a line."""

import pytest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import (
    make_env,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_HARD_DROP,
    ACTION_ROTATE_CW,
    ACTION_ROTATE_CCW,
)


BASE_SHAPES = {
    "I": {(0, 0), (0, 1), (0, 2), (0, 3)},
    "O": {(0, 0), (0, 1), (1, 0), (1, 1)},
    "T": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "S": {(0, 1), (0, 2), (1, 0), (1, 1)},
    "Z": {(0, 0), (0, 1), (1, 1), (1, 2)},
    "J": {(0, 0), (1, 0), (1, 1), (1, 2)},
    "L": {(0, 2), (1, 0), (1, 1), (1, 2)},
}


def _normalize(coords):
    """Normalize coords by shifting to origin (y, x)."""
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    normalized = {(r - min_r, c - min_c) for r, c in coords}
    if normalized in (
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 2), (2, 0), (2, 2)},
        {(0, 0), (0, 3), (3, 0), (3, 3)},
        {(0, 0), (0, 1), (0, 2), (0, 3),
         (1, 0), (1, 1), (1, 2), (1, 3),
         (2, 0), (2, 1), (2, 2), (2, 3),
         (3, 0), (3, 1), (3, 2), (3, 3)},
    ):
        return {(0, 0), (0, 1), (1, 0), (1, 1)}
    return normalized


def _rotate(coords):
    """Rotate 90 degrees clockwise about origin and renormalize."""
    rotated = {(c, -r) for r, c in coords}
    return _normalize(rotated)


def _generate_orientations(base):
    """Return unique orientations for a base shape."""
    orientations = []
    current = _normalize(base)
    for _ in range(4):
        if current not in orientations:
            orientations.append(current)
        current = _rotate(current)
    return orientations


ORIENTATIONS = {name: _generate_orientations(coords) for name, coords in BASE_SHAPES.items()}


def _extract_board(obs):
    """Return a binary 20x10 board regardless of wrapper format."""
    if isinstance(obs, dict):
        board = obs["board"][0:20, 4:14]
    else:
        board = obs[..., 0]
    return (np.asarray(board) > 0).astype(np.uint8)


def _extract_active_mask(obs):
    """Return the active tetromino mask aligned with the board."""
    if isinstance(obs, dict):
        mask = obs.get("active_tetromino_mask")
        if mask is None:
            return None
        region = mask[0:20, 4:14]
    else:
        if obs.shape[-1] < 2:
            return None
        region = obs[..., 1]
    result = (np.asarray(region) > 0).astype(np.uint8)
    return result if result.any() else None


def _identify_piece(mask):
    """Identify piece type and orientation index from mask."""
    coords = [(r, c) for r, c in zip(*np.where(mask > 0))]
    normalized = _normalize(coords)
    for name, orientations in ORIENTATIONS.items():
        for idx, orient in enumerate(orientations):
            if normalized == orient:
                return name, idx, orient
    raise ValueError(f"Unknown piece shape: {sorted(normalized)}")


def _column_heights(board):
    """Compute heights (number of filled cells) for each column."""
    heights = []
    rows = board.shape[0]
    for col in range(board.shape[1]):
        filled = np.where(board[:, col] > 0)[0]
        heights.append(rows - filled[0] if len(filled) else 0)
    return heights


def _count_holes(board):
    heights = _column_heights(board)
    holes = 0
    for col in range(board.shape[1]):
        filled = np.where(board[:, col] > 0)[0]
        if len(filled):
            top = filled[0]
            holes += int((board[top:, col] == 0).sum())
    return holes


def _bumpiness(board):
    heights = _column_heights(board)
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def _collides(board, shape, row, col):
    """Check if placing shape at (row, col) collides with walls or filled cells."""
    for dr, dc in shape:
        rr = row + dr
        cc = col + dc
        if cc < 0 or cc >= board.shape[1]:
            return True
        if rr >= board.shape[0]:
            return True
        if rr >= 0 and board[rr, cc]:
            return True
    return False


def _drop_row(board, shape, col):
    """Compute landing row for shape placed at column; return None if impossible."""
    row = -4  # allow spawn above board
    while True:
        if _collides(board, shape, row + 1, col):
            break
        row += 1
        if row > board.shape[0]:
            break
    if row < -3:
        return None
    return row


def _place_shape(board, shape, row, col):
    new_board = board.copy()
    for dr, dc in shape:
        rr = row + dr
        cc = col + dc
        if rr >= 0:
            new_board[rr, cc] = 1
    return new_board


def _clear_lines(board):
    full_rows = [r for r in range(board.shape[0]) if board[r].all()]
    if not full_rows:
        return board, 0
    remaining = np.delete(board, full_rows, axis=0)
    cleared = len(full_rows)
    new_board = np.vstack([np.zeros((cleared, board.shape[1]), dtype=board.dtype), remaining])
    return new_board, cleared


def _plan_move(board, piece_name, current_idx):
    orientations = ORIENTATIONS[piece_name]
    best_plan = None
    best_score = None
    for idx, shape in enumerate(orientations):
        width = max(c for _, c in shape) + 1
        for col in range(0, board.shape[1] - width + 1):
            landing_row = _drop_row(board, shape, col)
            if landing_row is None:
                continue
            placed = _place_shape(board, shape, landing_row, col)
            new_board, lines = _clear_lines(placed)
            holes = _count_holes(new_board)
            agg_height = sum(_column_heights(new_board))
            bump = _bumpiness(new_board)
            score = (-lines, holes, agg_height, bump)
            if best_score is None or score < best_score:
                best_score = score
                best_plan = (idx, col, lines)
    return best_plan


def _select_target_column(board, mask):
    """Choose a placement column that keeps the surface as flat as possible."""
    heights = _column_heights(board)
    cols = np.where(mask.any(axis=0))[0]
    width = cols[-1] - cols[0] + 1
    best_col = 0
    best_score = None

    for start in range(0, board.shape[1] - width + 1):
        segment = heights[start : start + width]
        score = (max(segment), sum(segment), start)
        if best_score is None or score < best_score:
            best_score = score
            best_col = start
    return best_col


def _align_and_drop(env, obs, target_col, target_shape, rotations):
    """Apply rotations and horizontal moves before hard drop."""
    total_lines = 0
    term = trunc = False
    info = {}

    # Rotate to desired orientation
    for action in rotations:
        obs, _, term, trunc, info = env.step(action)
        total_lines += info.get("lines_cleared", 0)
        if term or trunc:
            return obs, total_lines, term, trunc, info

    while True:
        mask = _extract_active_mask(obs)
        if mask is None:
            break
        if _normalize([(r, c) for r, c in zip(*np.where(mask > 0))]) != target_shape:
            # Safety: orientation drifted; stop to prevent infinite loop.
            break
        cols = np.where(mask.any(axis=0))[0]
        current_left = cols.min()
        if current_left > target_col:
            obs, _, term, trunc, info = env.step(ACTION_LEFT)
        elif current_left < target_col:
            obs, _, term, trunc, info = env.step(ACTION_RIGHT)
        else:
            break
        total_lines += info.get("lines_cleared", 0)
        if term or trunc:
            return obs, total_lines, term, trunc, info

    obs, _, term, trunc, info = env.step(ACTION_HARD_DROP)
    total_lines += info.get("lines_cleared", 0)
    return obs, total_lines, term, trunc, info


def _play_until_line(env, obs, max_pieces=120):
    """Drive the environment with a simple heuristic until a line clears."""
    total_lines = 0

    for _ in range(max_pieces):
        board = _extract_board(obs)
        mask = _extract_active_mask(obs)

        if mask is None:
            # No active piece (e.g., game just ended); advance once.
            obs, _, term, trunc, info = env.step(ACTION_HARD_DROP)
            total_lines += info.get("lines_cleared", 0)
            if term or trunc:
                return obs, total_lines, term, trunc
            continue

        piece_name, current_idx, current_shape = _identify_piece(mask)
        plan = _plan_move(board, piece_name, current_idx)
        if plan is None:
            target_col = _select_target_column(board, mask)
            rotations = []
            target_shape = current_shape
        else:
            target_idx, target_col, _ = plan
            orientations = ORIENTATIONS[piece_name]
            target_shape = orientations[target_idx]
            # Determine rotation sequence (prefer fewer moves)
            num_orient = len(orientations)
            cw = (target_idx - current_idx) % num_orient
            ccw = (current_idx - target_idx) % num_orient
            rotations = []
            if cw == 0 or cw <= ccw:
                rotations = [ACTION_ROTATE_CW] * cw
            else:
                rotations = [ACTION_ROTATE_CCW] * ccw

        obs, gained, term, trunc, info = _align_and_drop(
            env, obs, target_col, target_shape, rotations
        )
        total_lines += gained
        if total_lines > 0 or term or trunc:
            return obs, total_lines, term, trunc

    return obs, total_lines, False, False


def test_actual_line_clear():
    """Ensure the environment reports a cleared line using a scripted policy."""
    pytest.importorskip("tetris_gymnasium")

    env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
    try:
        obs, _ = env.reset(seed=7)
        total_lines = 0

        # Try a handful of deterministic seeds before failing.
        for attempt in range(5):
            obs, gained, term, trunc = _play_until_line(env, obs, max_pieces=120)
            total_lines += gained
            if total_lines > 0:
                break
            if term or trunc:
                obs, _ = env.reset(seed=7 + attempt + 1)

        assert (
            total_lines > 0
        ), "Scripted heuristic failed to clear a line in expected conditions."
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
