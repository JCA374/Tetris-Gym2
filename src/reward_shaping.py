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
      - 3D array (20x10x1) - single channel
      - 3D array (20x10x4) - 4-channel (extracts channel 0 = board)

    Note: Tetris Gymnasium raw board (24x18) has:
      - Rows 0-19:  Spawn + playable area
      - Rows 20-23: Bottom walls (NOT playable)
      - Cols 4-13:  Playable width
    """
    if isinstance(obs, dict):
        full = obs.get("board", np.zeros((20, 10)))
        if full.shape == (24, 18):
            board = full[0:20, 4:14]  # FIXED: Extract rows 0-19, not 2-21
        elif full.shape[:2] == (20, 10):
            board = full[:20, :10]
        else:
            h, w = full.shape[:2]
            h0 = max(0, (h - 20) // 2)
            w0 = max(0, (w - 10) // 2)
            board = full[h0:h0+20, w0:w0+10]
    elif hasattr(obs, "shape"):
        if len(obs.shape) == 3:
            # 3D observation - extract channel 0 (board channel)
            # Works for both (20,10,1) and (20,10,4)
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
    One-pass reward shaper focused on:
      - Killing center stacking
      - Encouraging side-well play
      - Rewarding line clears more when the surface is clean
      - (Optional) rewarding immediate improvements if prev_* metrics are provided in `info`

    Returns a scalar 'shaped' reward to add to the env reward.
    """
    import numpy as np

    # --------- Robust board extraction (works for (20,10), (20,10,1), Dict, etc.) ----------
    board = None
    if isinstance(obs, dict):
        # Try common keys
        for k in ("board", "obs", "observation"):
            if k in obs:
                full = obs[k]
                break
        else:
            # fall back to first array-like value
            full = next((v for v in obs.values() if hasattr(v, "shape")), None)
        if full is None:
            # last resort
            board = np.zeros((20, 10), dtype=np.int32)
        else:
            if len(full.shape) == 3:
                # (H, W, C) → channel 0
                full = full[:, :, 0]
            # Prefer center crop to 20x10 (avoid bottom "wall" rows from wrappers)
            h, w = full.shape[:2]
            h0 = max(0, (h - 20) // 2)
            w0 = max(0, (w - 10) // 2)
            board = full[h0:h0+20, w0:w0+10]
    elif hasattr(obs, "shape"):
        if len(obs.shape) == 3:
            board = obs[:, :, 0]
        elif len(obs.shape) == 2:
            board = obs
        else:
            board = np.zeros((20, 10), dtype=np.int32)
    else:
        board = np.zeros((20, 10), dtype=np.int32)

    board = np.asarray(board)
    board = (board > 0).astype(np.uint8)  # binary for geometry metrics

    # ----------------------------- Geometry helpers -----------------------------
    def get_column_heights(b):
        h, w = b.shape
        heights = []
        for c in range(w):
            col = b[:, c]
            nz = np.where(col > 0)[0]
            heights.append(h - nz[0] if nz.size else 0)
        return heights

    def count_holes(b):
        h, w = b.shape
        holes = 0
        for c in range(w):
            col = b[:, c]
            top = np.argmax(col > 0) if np.any(col) else h
            if top < h:
                holes += int((col[top+1:] == 0).sum())
        return int(holes)

    def surface_bumpiness(heights):
        return float(np.sum(np.abs(np.diff(heights)))) if len(heights) > 1 else 0.0

    def aggregate_height(heights):
        return float(np.sum(heights))

    # ----------------------------- Current metrics ------------------------------
    heights   = get_column_heights(board)
    holes     = count_holes(board)
    bump      = surface_bumpiness(heights)
    agg_h     = aggregate_height(heights)
    height_std = float(np.std(np.array(heights, dtype=np.float32)))

    # Outer lanes (encourage using sides; discourage empty sides)
    outer_idx     = [0, 1, 2, 7, 8, 9]
    outer_empty   = sum(1 for i in outer_idx if heights[i] == 0)
    columns_used  = sum(1 for h_ in heights if h_ > 0)

    # Optional deltas (only applied if training loop supplies prev_* in info)
    prev_holes = info.get("prev_holes")
    prev_bump  = info.get("prev_bump")
    prev_agg_h = info.get("prev_agg_h")

    # Lines cleared (env-dependent key)
    lines = info.get("lines_cleared", info.get("lines", 0)) or 0

    # ----------------------------- Shaping terms --------------------------------
    shaped = 0.0

    # Survival encouragement (small, steady)
    shaped += 2.0  # per step stay-alive bonus

    # Kill center-stacking: punish empty outer lanes & height concentration
    shaped -= 60.0 * outer_empty          # strong—forces side usage
    shaped -= 8.0  * height_std           # penalize "single mountain"

    # Basic geometry costs (rebalance toward hole removal)
    shaped -= 1.25 * holes
    shaped -= 0.60 * bump
    shaped -= 0.03 * agg_h

    # Side-well pattern bonus (attractor away from center)
    left_well  = (heights[0] <= 1) and all(h >= 8 for h in heights[1:4])
    right_well = (heights[9] <= 1) and all(h >= 8 for h in heights[6:9])
    if left_well or right_well:
        shaped += 120.0

    # Light spread encouragement (reward using more columns)
    shaped += 4.0 * columns_used

    # Optional immediate improvements (only when prev_* present)
    if prev_holes is not None:
        shaped += 2.0  * max(0.0, (prev_holes - holes))
    if prev_bump is not None:
        shaped += 0.5  * max(0.0, (prev_bump  - bump))
    if prev_agg_h is not None:
        shaped += 0.05 * max(0.0, (prev_agg_h - agg_h))

    # Line clears scale with surface cleanliness
    if lines > 0:
        surface_clean = 1.0 / (1.0 + holes * 0.05 + bump * 0.02)
        shaped += (250.0 * lines) * surface_clean

    # Terminal penalty (keep small; big negatives can destabilize)
    if done:
        shaped -= 20.0

    # Return: env reward + shaped signal
    return float(reward + shaped)


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
