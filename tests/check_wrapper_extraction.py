#!/usr/bin/env python3
"""Check what the wrapper is actually extracting"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import tetris_gymnasium  # Register the environment

print("="*70)
print("CHECKING WRAPPER EXTRACTION LOGIC")
print("="*70)

# Create raw environment
raw_env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
obs, _ = raw_env.reset(seed=42)

board = obs['board']
print(f"\nRaw board shape: {board.shape}")
print(f"Expected: 24 rows x 18 cols")
print(f"  Rows 0-3: Top padding/spawn area")
print(f"  Rows 4-23: Main game area")
print(f"  Cols 0-3: Left wall")
print(f"  Cols 4-13: Playable area (10 cols)")
print(f"  Cols 14-17: Right wall")

# Check bottom rows
print("\nBottom 4 rows of raw board:")
for r in range(20, 24):
    row_str = "".join("█" if board[r, c] > 0 else "." for c in range(18))
    print(f"  Row {r}: {row_str}")

print("\nCurrent wrapper extraction: board[2:22, 4:14]")
extracted_current = board[2:22, 4:14]
print(f"  Shape: {extracted_current.shape}")
print(f"  This gets rows 2-21 from raw board")
print(f"  Row 21 from raw board:")
row_21_str = "".join("█" if board[21, c] > 0 else "." for c in range(4, 14))
print(f"    {row_21_str}")

print("\nCorrect wrapper extraction should be: board[0:20, 4:14] or board[4:24, 4:14]?")
print("Let's check both...")

# Option 1: rows 0-19
print("\nOption 1: board[0:20, 4:14] (rows 0-19)")
extracted_opt1 = board[0:20, 4:14]
print(f"  Shape: {extracted_opt1.shape}")
print(f"  Bottom row (row 19 from raw):")
row_19_str = "".join("█" if board[19, c] > 0 else "." for c in range(4, 14))
print(f"    {row_19_str} ← Should be playable, not wall")

# Option 2: rows 4-23
print("\nOption 2: board[4:24, 4:14] (rows 4-23)")
extracted_opt2 = board[4:24, 4:14]
print(f"  Shape: {extracted_opt2.shape}")
print(f"  Bottom row (row 23 from raw):")
row_23_str = "".join("█" if board[23, c] > 0 else "." for c in range(4, 14))
print(f"    {row_23_str} ← This IS the wall!")

# Let's check where the actual walls start
print("\nFinding where walls start from bottom:")
for r in range(23, -1, -1):
    middle_cols = board[r, 4:14]  # playable columns
    if np.all(middle_cols > 0):
        print(f"  Row {r}: COMPLETELY FILLED (likely wall)")
    elif np.any(middle_cols > 0):
        print(f"  Row {r}: Partially filled (has pieces)")
        break
    else:
        print(f"  Row {r}: Empty")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The tetris-gymnasium environment (24x18) layout appears to be:
- Rows 0-3: Top spawn area (hidden)
- Rows 4-19: Visible playable area (16 rows)
- Rows 20-23: Bottom wall (4 rows)
- Cols 0-3: Left wall
- Cols 4-13: Playable area (10 cols)
- Cols 14-17: Right wall

But standard Tetris has 20 visible rows, not 16!

The wrapper is extracting rows [2:22] which includes:
- Some spawn area (rows 2-3)
- Playable area (rows 4-19)
- Bottom wall (rows 20-21)

That's why the bottom 2 rows are always filled!

CORRECT extraction should be: board[4:24, 4:14] to get all 20 rows
including the bottom wall area, OR board[0:20, 4:14] to get spawn + playable
area without the wall.

Actually, we need to check the tetris-gymnasium docs to understand the layout.
""")

raw_env.close()
