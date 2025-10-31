#!/usr/bin/env python3
"""Smarter stacking strategy - rotate through positions more gradually"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("SMARTER STACKING TEST")
print("="*80)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=500)

def print_board(board, num_rows=12):
    """Print board visualization"""
    print("  " + "0123456789")
    for r in range(min(num_rows, board.shape[0])):
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        fullness = np.count_nonzero(board[r, :])
        marker = " <-- FULL!" if fullness == 10 else ""
        print(f"{r:2d} {row_str}  ({fullness}/10){marker}")

# Smarter strategy: Position pieces at specific columns
# Cycle through target columns: 0, 2, 4, 6, 8, 7, 5, 3, 1, 9
# This gradually fills all columns

target_columns = [0, 2, 4, 6, 8, 7, 5, 3, 1, 9]
total_lines = 0
step = 0
max_row_fullness = 0

print("\nStrategy: Cycle through all column positions systematically\n")

for piece_num in range(500):
    target_col = target_columns[piece_num % len(target_columns)]

    # Calculate how to reach target column from center (~column 5)
    # Pieces spawn around column 4-5
    moves_needed = target_col - 5

    if moves_needed < 0:
        # Move LEFT
        for _ in range(abs(moves_needed) * 2):  # *2 to ensure we get there
            obs, _, term, trunc, _ = env.step(0)  # LEFT
            if term or trunc:
                break
    elif moves_needed > 0:
        # Move RIGHT
        for _ in range(moves_needed * 2):
            obs, _, term, trunc, _ = env.step(1)  # RIGHT
            if term or trunc:
                break

    if not (term or trunc):
        # Hard drop
        obs, reward, term, trunc, info = env.step(5)  # HARD_DROP
        step += 1

        # Check for lines
        lines = info.get('lines_cleared', 0)
        if lines > 0:
            total_lines += lines
            board = obs['board'][0:20, 4:14]
            print(f"\n✅ ✅ ✅ STEP {step}: {lines} LINE(S) CLEARED! ✅ ✅ ✅")
            print(f"   Total cleared: {total_lines}")
            print(f"   Reward: {reward}\n")
            print_board(board, 15)

    if term or trunc:
        print(f"\nGame over at piece {piece_num + 1}")
        break

    # Progress update
    if (piece_num + 1) % 100 == 0:
        board = obs['board'][0:20, 4:14]
        for r in range(20):
            fullness = np.count_nonzero(board[r, :])
            max_row_fullness = max(max_row_fullness, fullness)

        print(f"\n--- Piece {piece_num + 1} ---")
        print(f"Max row fullness: {max_row_fullness}/10")
        print(f"Total lines cleared: {total_lines}")
        print_board(board, 12)

# Final board
board = obs['board'][0:20, 4:14]
print(f"\n" + "="*80)
print("FINAL BOARD")
print("="*80)
print_board(board, 20)

# Column usage stats
print("\nColumn heights:")
heights = []
for c in range(10):
    h = 0
    for r in range(20):
        if board[r, c] > 0:
            h = 20 - r
            break
    heights.append(h)
print(f"  {heights}")

# Max row fullness
for r in range(20):
    fullness = np.count_nonzero(board[r, :])
    max_row_fullness = max(max_row_fullness, fullness)

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Pieces placed: {piece_num + 1}")
print(f"Steps taken: {step}")
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness: {max_row_fullness}/10")
print()

if total_lines > 0:
    print(f"✅ ✅ ✅ LINE CLEARING CONFIRMED! ✅ ✅ ✅")
    print(f"   The environment WORKS!")
    print(f"   Problem was the WRONG ACTION MAPPING!")
elif max_row_fullness >= 9:
    print(f"⚠️  Almost cleared lines ({max_row_fullness}/10)")
    print(f"   Need better piece placement strategy")
else:
    print(f"❌ Max fullness: {max_row_fullness}/10")

print("="*80)
