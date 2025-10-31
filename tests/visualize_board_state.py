#!/usr/bin/env python3
"""Visualize what's actually happening on the board"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("BOARD STATE VISUALIZATION")
print("="*80)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=42)

def print_board(board, title="Board"):
    """Print a visual representation of the board"""
    print(f"\n{title}:")
    print("  " + "0123456789")
    for r in range(min(10, board.shape[0])):  # Show top 10 rows
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        print(f"{r:2d} {row_str}")

# Show initial board
board = obs['board']
playable = board[0:20, 4:14]
print_board(playable, "Initial playable area (20x10)")

# Show piece spawn location
mask = obs['active_tetromino_mask']
piece_playable = mask[0:20, 4:14]
print_board(piece_playable, "Active piece location")

print("\n" + "="*40)
print("Taking 20 HARD_DROP actions...")
print("="*40)

for i in range(20):
    obs, reward, term, trunc, info = env.step(5)  # HARD_DROP

    board = obs['board']
    playable = board[0:20, 4:14]

    # Check for new piece
    mask = obs['active_tetromino_mask']
    piece_playable = mask[0:20, 4:14]

    lines = info.get('lines_cleared', 0)

    if lines > 0 or i < 5 or i % 5 == 0:  # Show first 5 and every 5th
        print(f"\nStep {i+1}:")
        if lines > 0:
            print(f"✅ {lines} LINE(S) CLEARED!")
        print_board(playable)

        # Show column heights
        heights = []
        for c in range(10):
            h = 0
            for r in range(20):
                if playable[r, c] > 0:
                    h = 20 - r
                    break
            heights.append(h)
        print(f"Heights: {heights}")

    if term or trunc:
        print(f"\n❌ Game over at step {i+1}")
        print_board(playable, "Final board")
        break

env.close()

print("\n" + "="*80)
