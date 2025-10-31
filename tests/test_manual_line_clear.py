#!/usr/bin/env python3
"""Try to manually clear a line by moving pieces left/right before dropping"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("MANUAL LINE CLEARING ATTEMPT")
print("="*80)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=42)

def print_board(board, title="Board"):
    """Print board visualization"""
    print(f"\n{title}:")
    print("  " + "0123456789")
    for r in range(min(15, board.shape[0])):
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        fullness = np.count_nonzero(board[r, :])
        print(f"{r:2d} {row_str}  ({fullness}/10)")

# Strategy: Alternate between moving LEFT, then drop, then RIGHT, then drop
# This should spread pieces across the board
print("\nStrategy: Alternate LEFT/RIGHT movements before dropping")
print("Pattern: LEFT x3, HARD_DROP, RIGHT x3, HARD_DROP, repeat...\n")

# Action mapping: LEFT=0, RIGHT=1, DOWN=2, HARD_DROP=5
pattern = [
    [0, 0, 0, 5],  # Move left 3x, hard drop
    [1, 1, 1, 5],  # Move right 3x, hard drop
]

total_lines = 0
max_row_fullness = 0
step = 0
pattern_idx = 0

for i in range(50):  # Try 50 pieces
    for action in pattern[pattern_idx]:
        obs, reward, term, trunc, info = env.step(action)
        step += 1

        # Check for lines
        lines = info.get('lines_cleared', 0)
        if lines > 0:
            total_lines += lines
            print(f"✅ Step {step}: {lines} LINE(S) CLEARED!")
            board = obs['board'][0:20, 4:14]
            print_board(board, f"After clearing {lines} lines")

        if term or trunc:
            break

    if term or trunc:
        print(f"\n❌ Game over at step {step}")
        break

    pattern_idx = (pattern_idx + 1) % 2

    # Show progress every 10 pieces
    if i % 10 == 0:
        board = obs['board'][0:20, 4:14]
        max_row_fullness = max(max_row_fullness, max([np.count_nonzero(board[r, :]) for r in range(20)]))
        print(f"\nPiece {i}:")
        print_board(board, "Current board")

# Final board
board = obs['board'][0:20, 4:14]
print_board(board, "FINAL BOARD")

# Calculate final stats
for r in range(20):
    fullness = np.count_nonzero(board[r, :])
    max_row_fullness = max(max_row_fullness, fullness)

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness: {max_row_fullness}/10 cells")
print()

if total_lines > 0:
    print(f"✅ LINE CLEARING WORKS!")
    print(f"   We CAN clear lines by spreading pieces!")
elif max_row_fullness == 10:
    print(f"❌ Full row achieved but NOT cleared - env bug!")
else:
    print(f"❌ Max fullness: {max_row_fullness}/10")
    print(f"   Pieces still cannot reach all columns")

print("="*80)
