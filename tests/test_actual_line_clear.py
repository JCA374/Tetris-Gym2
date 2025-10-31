#!/usr/bin/env python3
"""Test if we can ACTUALLY clear a line now that we know all columns are reachable"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("ACTUAL LINE CLEARING TEST")
print("="*80)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=42)

def print_board(board):
    """Print board with row fullness"""
    print("  " + "0123456789")
    for r in range(min(12, board.shape[0])):
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        fullness = np.count_nonzero(board[r, :])
        marker = " <-- FULL!" if fullness == 10 else ""
        print(f"{r:2d} {row_str}  ({fullness}/10){marker}")

# Strategy: Aggressively spread pieces across ALL columns
# Pattern: FAR LEFT (10x), drop, FAR RIGHT (10x), drop, CENTER, drop...
patterns = [
    [0]*10 + [5],  # Far left
    [1]*10 + [5],  # Far right
    [0]*5 + [5],   # Mid left
    [1]*5 + [5],   # Mid right
    [5],           # Center
]

total_lines = 0
step = 0
pattern_idx = 0
max_row_fullness = 0

print("\nStrategy: Spread pieces aggressively across all columns\n")

for piece_num in range(200):  # Try 200 pieces
    pattern = patterns[pattern_idx % len(patterns)]

    for action in pattern:
        obs, reward, term, trunc, info = env.step(action)
        step += 1

        lines = info.get('lines_cleared', 0)
        if lines > 0:
            total_lines += lines
            board = obs['board'][0:20, 4:14]
            print(f"\n✅ ✅ ✅ STEP {step}: {lines} LINE(S) CLEARED! ✅ ✅ ✅")
            print(f"   Total lines cleared: {total_lines}")
            print(f"   Reward: {reward}")
            print_board(board)

        if term or trunc:
            break

    if term or trunc:
        break

    pattern_idx += 1

    # Show progress every 50 pieces
    if piece_num % 50 == 0:
        board = obs['board'][0:20, 4:14]
        for r in range(20):
            fullness = np.count_nonzero(board[r, :])
            max_row_fullness = max(max_row_fullness, fullness)

        print(f"\nPiece {piece_num}: Max row fullness so far: {max_row_fullness}/10")
        if piece_num % 100 == 0:
            print_board(board)

# Final stats
board = obs['board'][0:20, 4:14]
for r in range(20):
    fullness = np.count_nonzero(board[r, :])
    max_row_fullness = max(max_row_fullness, fullness)

env.close()

print(f"\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Pieces placed: {piece_num + 1}")
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness achieved: {max_row_fullness}/10 cells")
print()

if total_lines > 0:
    print(f"✅ ✅ ✅ LINE CLEARING WORKS! ✅ ✅ ✅")
    print(f"   Cleared {total_lines} lines!")
    print(f"   The environment is FULLY FUNCTIONAL!")
    print()
    print(f"   The problem was:")
    print(f"   1. WRONG action mapping (now fixed)")
    print(f"   2. Agent needs to learn aggressive movements")
elif max_row_fullness == 10:
    print(f"❌ Row became full but wasn't cleared - BUG in environment")
else:
    print(f"❌ Couldn't fill a complete row")
    print(f"   Max: {max_row_fullness}/10 cells")

print("="*80)
