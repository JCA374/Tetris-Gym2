#!/usr/bin/env python3
"""Deep analysis of CompleteVisionWrapper extraction issue"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env
import gymnasium as gym

print("="*80)
print("DEEP ANALYSIS: CompleteVisionWrapper Board Extraction")
print("="*80)

# Create raw environment (unwrapped to access dict observation)
wrapped_env = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)

# Get the raw tetris environment before wrappers
raw_env = wrapped_env
while hasattr(raw_env, 'env'):
    raw_env = raw_env.env

obs, _ = wrapped_env.reset(seed=42)

board = obs['board']
mask = obs['active_tetromino_mask']

print(f"\n1️⃣ RAW ENVIRONMENT STRUCTURE")
print(f"   Board shape: {board.shape}")
print(f"   Active piece mask shape: {mask.shape}")
print(f"   Unique values in board: {np.unique(board)}")

# Analyze the structure row by row
print(f"\n2️⃣ ROW-BY-ROW ANALYSIS (showing all 24 rows)")
print(f"   Legend: █ = filled, . = empty, X = active piece")
print()

for r in range(board.shape[0]):
    row_board = board[r, :]
    row_mask = mask[r, :]

    # Build display string
    row_str = ""
    for c in range(board.shape[1]):
        if row_mask[c] > 0:
            row_str += "X"
        elif row_board[c] > 0:
            row_str += "█"
        else:
            row_str += "."

    filled_count = np.count_nonzero(row_board)
    mask_count = np.count_nonzero(row_mask)

    # Identify row type
    row_type = ""
    if filled_count == 18:  # All columns filled
        row_type = " ← COMPLETELY FILLED (WALL?)"
    elif filled_count == 8:  # Left + right walls (4+4)
        row_type = " ← SIDE WALLS ONLY"
    elif filled_count > 0:
        row_type = f" ← {filled_count} cells filled"

    if mask_count > 0:
        row_type += f" [ACTIVE PIECE: {mask_count}]"

    print(f"   Row {r:2d}: {row_str} {row_type}")

# Analyze the extraction zones
print(f"\n3️⃣ EXTRACTION ZONES")
print(f"   Current wrapper: board[2:22, 4:14]")
print(f"   - Takes rows 2-21 (20 rows)")
print(f"   - Takes cols 4-13 (10 cols)")

# Check what each extraction would give us
print(f"\n4️⃣ TESTING DIFFERENT EXTRACTION STRATEGIES")

strategies = [
    ("Current [2:22, 4:14]", 2, 22, 4, 14),
    ("Option A [0:20, 4:14]", 0, 20, 4, 14),
    ("Option B [4:24, 4:14]", 4, 24, 4, 14),
    ("Option C [2:20, 4:14]", 2, 20, 4, 14),  # 18 rows
]

for name, r_start, r_end, c_start, c_end in strategies:
    extracted = board[r_start:r_end, c_start:c_end]
    shape = extracted.shape

    # Check bottom rows
    bottom_1 = extracted[-1, :]
    bottom_2 = extracted[-2, :] if shape[0] >= 2 else None

    bottom_1_full = np.all(bottom_1 > 0)
    bottom_2_full = np.all(bottom_2 > 0) if bottom_2 is not None else False

    print(f"\n   {name}:")
    print(f"     Shape: {shape}")
    print(f"     Bottom row (idx {shape[0]-1}): {'█'*10 if bottom_1_full else 'Mixed'}")
    print(f"     2nd from bottom: {'█'*10 if bottom_2_full else 'Mixed'}")

    # Count how many bottom rows are completely filled
    fully_filled_bottom = 0
    for r_idx in range(shape[0]-1, -1, -1):
        if np.all(extracted[r_idx, :] > 0):
            fully_filled_bottom += 1
        else:
            break

    print(f"     Consecutive filled rows from bottom: {fully_filled_bottom}")

    if fully_filled_bottom > 0:
        print(f"     ⚠️  WARNING: Bottom {fully_filled_bottom} row(s) are walls!")

# Check the playable area more carefully
print(f"\n5️⃣ IDENTIFYING TRUE PLAYABLE AREA")

# Find where the walls actually are
print(f"\n   Checking column 4-13 (playable width):")
for r in range(24):
    playable_cols = board[r, 4:14]
    filled = np.count_nonzero(playable_cols)
    all_filled = np.all(playable_cols > 0)

    if all_filled:
        print(f"     Row {r}: ALL 10 playable columns filled (WALL)")
    elif filled > 0:
        print(f"     Row {r}: {filled}/10 playable columns filled")

# Test with actual gameplay
print(f"\n6️⃣ GAMEPLAY TEST: Do placed pieces appear in walls?")
print(f"   Resetting and placing 5 pieces...")

obs, _ = wrapped_env.reset(seed=999)
for i in range(5):
    # Hard drop
    obs, reward, term, trunc, info = wrapped_env.step(6)
    if term or trunc:
        print(f"   Game ended after {i+1} pieces")
        break

board_after = obs['board']

print(f"\n   Bottom 4 rows after 5 placements (cols 4-13):")
for r in range(20, 24):
    playable_cols = board_after[r, 4:14]
    row_str = "".join("█" if c > 0 else "." for c in playable_cols)
    filled = np.count_nonzero(playable_cols)
    print(f"     Row {r}: {row_str} ({filled}/10 filled)")

print(f"\n7️⃣ CONCLUSION & RECOMMENDATION")
print("="*80)

wrapped_env.close()
