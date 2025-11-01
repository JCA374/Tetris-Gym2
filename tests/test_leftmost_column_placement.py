#!/usr/bin/env python3
"""Test if pieces can actually LAND in leftmost columns (not just move there)"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("LEFTMOST COLUMN PLACEMENT TEST")
print("="*80)

env = Tetris(render_mode=None)

def print_board(board, title="Board"):
    print(f"\n{title}:")
    print("  " + "0123456789")
    for r in range(min(15, board.shape[0])):
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        fullness = np.count_nonzero(board[r, :])
        print(f"{r:2d} {row_str}  ({fullness}/10)")

# Test: Try to place pieces specifically in columns 0, 1, 2
print("\nAttempting to place pieces in leftmost columns (0, 1, 2)")
print("Strategy: Move LEFT many times, then HARD DROP\n")

obs, _ = env.reset(seed=100)

for piece_num in range(20):
    print(f"Piece {piece_num}:")

    # Show where piece starts
    mask = obs['active_tetromino_mask'][0:20, 4:14]
    cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        print(f"  Spawn: columns {cols.min()}-{cols.max()}")

    # Move FAR LEFT
    for i in range(15):
        obs, _, term, trunc, _ = env.step(0)  # LEFT (action 0)
        if term or trunc:
            break

    if not (term or trunc):
        # Show where piece is now
        mask = obs['active_tetromino_mask'][0:20, 4:14]
        cols = np.where(mask.sum(axis=0) > 0)[0]
        if len(cols) > 0:
            print(f"  After LEFT: columns {cols.min()}-{cols.max()}")

        # HARD DROP
        obs, reward, term, trunc, info = env.step(5)  # HARD_DROP (action 5)

        # Check where it landed
        board = obs['board'][0:20, 4:14]

        # Find the piece that just landed (highest non-zero cells)
        for r in range(20):
            if np.any(board[r, :] > 0):
                landed_cols = np.where(board[r, :] > 0)[0]
                print(f"  Landed: row {r}, columns {landed_cols.min()}-{landed_cols.max()}")
                break

    if term or trunc:
        print(f"\nGame over at piece {piece_num}")
        break

# Show final board
board = obs['board'][0:20, 4:14]
print_board(board, "FINAL BOARD")

# Column heights
heights = []
for c in range(10):
    h = 0
    for r in range(20):
        if board[r, c] > 0:
            h = 20 - r
            break
    heights.append(h)

print(f"\nColumn heights: {heights}")

# Check which columns have any pieces
used_columns = [c for c in range(10) if heights[c] > 0]
unused_columns = [c for c in range(10) if heights[c] == 0]

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Used columns: {used_columns}")
print(f"Unused columns: {unused_columns}")
print()

if 0 in used_columns and 1 in used_columns and 2 in used_columns:
    print("✅ Leftmost columns (0, 1, 2) CAN be used!")
    print("   The issue is with agent exploration/learning")
else:
    print(f"❌ Leftmost columns NOT used!")
    print(f"   Columns {unused_columns} remain empty")
    if 0 not in used_columns:
        print("\n⚠️  CRITICAL: Column 0 is unreachable or has placement issues!")

print("="*80)
