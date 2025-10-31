# src/reward_shaping.py
"""
Tetris Reward Shaping (Overnight version, single policy)

Goals:
- Stable overnight learning with clear signals
- Minimal moving parts (one shaper), predictable ranges
- Compatibility: exports the same names train.py expects

This module exposes ONE shaping function:
    overnight_reward_shaping(obs, action, reward, done, info)

…plus aliases:
    balanced_reward_shaping   = overnight_reward_shaping
    aggressive_reward_shaping = overnight_reward_shaping
    positive_reward_shaping   = overnight_reward_shaping
"""

import numpy as np

# =============================================================================
# BOARD ANALYSIS HELPERS (kept small, fast, and deterministic)
# =============================================================================

def extract_board_from_obs(obs):
    """
    Extract playable 20x10 board and binarize to {0,1}.
    Accepts:
      - dict with 'board' (24x18 or 20x10)
      - 2D array (20x10)
      - 3D array (20x10x1)
    """
    if isinstance(obs, dict):
        full = obs.get("board", np.zeros((20, 10)))
        if full.shape == (24, 18):
            board = full[2:22, 4:14]
        elif full.shape[:2] == (20, 10):
            board = full[:20, :10]
        else:
            h, w = full.shape[:2]
            h0 = max(0, (h - 20) // 2)
            w0 = max(0, (w - 10) // 2)
            board = full[h0:h0+20, w0:w0+10]
    elif hasattr(obs, "shape"):
        if len(obs.shape) == 3 and obs.shape[2] == 1:
            board = obs[:, :, 0]
        elif len(obs.shape) == 2:
            board = obs
        else:
            board = np.zeros((20, 10))
    else:
        board = np.zeros((20, 10))

    if board.shape != (20, 10):
        out = np.zeros((20, 10))
        h, w = min(20, board.shape[0]), min(10, board.shape[1])
        out[:h, :w] = board[:h, :w]
        board = out

    # Binary normalize for robust metrics
    return (board > 0).astype(np.float32)


def get_column_heights(board):
    """Height of each column (distance from highest filled cell to bottom)."""
    rows, cols = board.shape
    heights = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r, c] != 0:
                h = rows - r
                break
        heights.append(h)
    return heights


def calculate_aggregate_height(board):
    """Sum of column heights."""
    return sum(get_column_heights(board))


def count_holes(board):
    """Empty cells with at least one filled cell above in the same column."""
    rows, cols = board.shape
    holes = 0
    for c in range(cols):
        seen = False
        for r in range(rows):
            if board[r, c] != 0:
                seen = True
            elif seen and board[r, c] == 0:
                holes += 1
    return holes


def calculate_bumpiness(board):
    """Sum of absolute diffs of adjacent column heights (scaled lightly)."""
    h = np.array(get_column_heights(board), dtype=np.int32)
    if len(h) <= 1:
        return 0.0
    return float(np.abs(np.diff(h)).sum())


def calculate_wells(board):
    """
    Sum of well depths: how much lower a column is vs min(neighbor heights).
    """
    h = get_column_heights(board)
    n = len(h)
    wells = 0
    for i in range(n):
        left = h[i-1] if i > 0 else h[i]
        right = h[i+1] if i < n-1 else h[i]
        wells += max(0, min(left, right) - h[i])
    return float(wells)


def get_max_height(board):
    h = get_column_heights(board)
    return max(h) if h else 0


def calculate_horizontal_distribution(board):
    """
    0..1 proxy measuring how spread pieces are across columns.
    Higher is better (discourages center-stacking).
    """
    if board.size == 0 or not board.any():
        return 0.5
    rows, cols = board.shape
    xs = np.arange(cols, dtype=np.float32)
    vals = []
    for r in range(rows):
        row = board[r]
        if row.any():
            w = row
            center = (xs * w).sum() / max(1e-6, w.sum())
            var = ((xs - center) ** 2 * w).sum() / max(1e-6, w.sum())
            max_var = ((cols - 1) ** 2) / 4.0
            vals.append(var / max_var if max_var > 0 else 0.5)
    return float(np.mean(vals)) if vals else 0.5


# =============================================================================
# SINGLE REWARD SHAPER (overnight)
# =============================================================================

def overnight_reward_shaping(obs, action, reward, done, info):
    """
    Single shaping policy tuned for stable overnight training.

    Structure:
      shaped = base(line clear) + survival
               - (height + holes + bumpiness + wells)
               + (spread bonus)
               - death penalty

    Ranges:
      Clamp to [-100, 600] to keep gradients well-behaved.
    """
    board = extract_board_from_obs(obs)

    # Base: amplify env reward a bit (env reward is small/sparse)
    shaped = float(reward) * 100.0

    # Metrics (fast)
    agg_h   = calculate_aggregate_height(board)       # 0..200
    holes   = count_holes(board)                      # 0..~200
    bump    = calculate_bumpiness(board)              # 0..~100
    wells   = calculate_wells(board)                  # 0..~100
    spread  = calculate_horizontal_distribution(board)  # 0..1

    # Penalties (light but consistent)
    shaped -= 0.12 * agg_h
    shaped -= 1.30 * holes
    shaped -= 0.06 * bump
    shaped -= 0.10 * wells

    # Bonus: distribution
    shaped += 4.0 * spread

    # Survival bonus (helps early exploration)
    steps = int(info.get("steps", 0))
    shaped += min(steps * 0.02, 3.0)

    # Line clear bonus (simple, monotonic)
    lines = int(info.get("lines_cleared", 0))
    if lines > 0:
        shaped += lines * 80.0
        if lines == 4:  # modest tetris kicker
            shaped += 120.0

    # Episode end penalty
    if done:
        shaped -= 30.0

    # Clamp
    return float(np.clip(shaped, -100.0, 600.0))


# Backward-compat aliases so train.py keeps working without edits
balanced_reward_shaping   = overnight_reward_shaping
aggressive_reward_shaping = overnight_reward_shaping
positive_reward_shaping   = overnight_reward_shaping


# =============================================================================
# Quick self-test (optional)
# =============================================================================
if __name__ == "__main__":
    b = np.zeros((20, 10), dtype=np.float32)
    b[-4:, :] = 1
    demo = overnight_reward_shaping(b, 0, reward=0.0, done=False,
                                    info={"steps": 50, "lines_cleared": 1})
    print("Demo shaped reward:", demo)
