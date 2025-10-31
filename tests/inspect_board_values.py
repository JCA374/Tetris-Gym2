#!/usr/bin/env python3
"""Inspect actual board values to understand line clearing"""

from tetris_gymnasium.envs import Tetris
import numpy as np

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=42)

print("="*80)
print("BOARD VALUE INSPECTION")
print("="*80)

# Take a few steps to place some pieces
for i in range(5):
    obs, reward, term, trunc, info = env.step(5)  # HARD_DROP

board = obs['board']

print(f"\nFull board shape: {board.shape}")
print(f"Unique values on board: {np.unique(board)}")
print()

# Show a few rows to see the pattern
print("Sample rows (showing all 18 columns):")
print(" Col: " + "".join(f"{i%10}" for i in range(18)))
for r in [0, 1, 18, 19, 20, 21, 22, 23]:
    row_values = "".join(str(board[r, c]) for c in range(18))
    print(f"Row {r:2d}: {row_values}")

print("\nPlayable area (rows 0-19, cols 4-13):")
playable = board[0:20, 4:14]
print(f"Shape: {playable.shape}")
print(f"Unique values: {np.unique(playable)}")

print("\nBottom rows of playable area:")
print("  " + "0123456789")
for r in range(15, 20):
    row = playable[r, :]
    row_str = "".join(str(int(v)) for v in row)
    fullness = np.count_nonzero(row)
    print(f"{r}: {row_str}  ({fullness}/10 filled)")

# Check what the line clearing function would detect
print("\n" + "="*40)
print("Testing line clearing detection:")
print("="*40)

# Manually check each row of the FULL board
filled_rows = (~(board == 0).any(axis=1)) & (~(board == 1).all(axis=1))
print(f"\nRows detected as filled: {np.where(filled_rows)[0]}")

# Check just the playable area
playable_filled = ~(playable == 0).any(axis=1)
print(f"Playable rows that are completely filled (no zeros): {np.where(playable_filled)[0]}")

# Show which rows have zeros
rows_with_zeros = []
for r in range(20):
    if (playable[r, :] == 0).any():
        num_zeros = (playable[r, :] == 0).sum()
        rows_with_zeros.append(f"Row {r}: {num_zeros} zeros")

print(f"\nPlayable rows with empty cells:")
for row_info in rows_with_zeros[-10:]:  # Show last 10
    print(f"  {row_info}")

env.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The line clearing function checks the FULL board (24x18),")
print("not just the playable area (20x10).")
print("A row is cleared only if it has:")
print("  1. No zeros (all cells filled)")
print("  2. Not all ones (not a wall row)")
print()
print("Since walls (value=1) are in columns 0-3 and 14-17,")
print("the playable columns 4-13 need to be completely filled")
print("with piece values (2-9) for the row to clear.")
print("="*80)
