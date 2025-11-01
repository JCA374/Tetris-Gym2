#!/usr/bin/env python3
"""Debug: Why do pieces move left but land in center?"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("PIECE MOVEMENT DEBUG")
print("="*80)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=42)

print("\nTest 1: Single piece - move LEFT and observe each step")
print("="*60)

# Show initial spawn
mask = obs['active_tetromino_mask'][0:20, 4:14]
cols = np.where(mask.sum(axis=0) > 0)[0]
print(f"Step 0 - Spawn: columns {cols.min()}-{cols.max()}")

# Move LEFT step by step
for i in range(1, 11):
    obs, reward, term, trunc, info = env.step(0)  # Action 0 = LEFT

    mask = obs['active_tetromino_mask'][0:20, 4:14]
    cols = np.where(mask.sum(axis=0) > 0)[0]
    rows = np.where(mask.sum(axis=1) > 0)[0]

    if len(cols) > 0 and len(rows) > 0:
        print(f"Step {i} - LEFT: columns {cols.min()}-{cols.max()}, rows {rows.min()}-{rows.max()}")

    if term or trunc:
        print(f"  LOCKED at step {i}")
        break

# Show where it landed
board = obs['board'][0:20, 4:14]
print("\nFinal board (top 5 rows):")
print("  " + "0123456789")
for r in range(5):
    row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(10))
    cols_filled = [c for c in range(10) if board[r, c] > 0]
    if cols_filled:
        print(f"{r}: {row_str}  (cols {min(cols_filled)}-{max(cols_filled)})")
    else:
        print(f"{r}: {row_str}")

env.close()

# Test 2: Check if HARD_DROP behaves differently
print("\n" + "="*60)
print("Test 2: Move LEFT, then HARD_DROP")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=42)

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols = np.where(mask.sum(axis=0) > 0)[0]
print(f"Initial spawn: columns {cols.min()}-{cols.max()}")

# Move LEFT 8 times WITHOUT dropping
for i in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols = np.where(mask.sum(axis=0) > 0)[0]
print(f"After 8 LEFT moves: columns {cols.min()}-{cols.max()}")

# Now HARD_DROP
obs, _, term, trunc, _ = env.step(5)  # HARD_DROP

board = obs['board'][0:20, 4:14]
print("\nBoard after HARD_DROP:")
print("  " + "0123456789")
for r in range(5):
    row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(10))
    if np.any(board[r, :] > 0):
        cols_filled = [c for c in range(10) if board[r, c] > 0]
        print(f"{r}: {row_str}  (cols {min(cols_filled)}-{max(cols_filled)})")

env.close()

# Test 3: Use DOWN instead of HARD_DROP
print("\n" + "="*60)
print("Test 3: Move LEFT, then use DOWN repeatedly (not HARD_DROP)")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=42)

# Move LEFT 8 times
for i in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols = np.where(mask.sum(axis=0) > 0)[0]
print(f"After LEFT moves: columns {cols.min()}-{cols.max()}")

# Use DOWN repeatedly instead of HARD_DROP
for i in range(20):
    obs, _, term, trunc, _ = env.step(2)  # DOWN (action 2)
    if term or trunc:
        break

board = obs['board'][0:20, 4:14]
print("\nBoard after DOWN drops:")
print("  " + "0123456789")
for r in range(5):
    row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(10))
    if np.any(board[r, :] > 0):
        cols_filled = [c for c in range(10) if board[r, c] > 0]
        print(f"{r}: {row_str}  (cols {min(cols_filled)}-{max(cols_filled)})")

env.close()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If pieces land in different columns than where they moved to,")
print("there might be a gravity/collision issue during the drop phase.")
print("="*80)
