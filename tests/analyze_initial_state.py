#!/usr/bin/env python3
"""Analyze what the initial board state looks like"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env

print("="*70)
print("ANALYZING INITIAL BOARD STATE")
print("="*70)

env = make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False)
obs, info = env.reset(seed=1)

print(f"\nObservation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")
print(f"Observation range: [{obs.min()}, {obs.max()}]")

# Extract board
board = obs[:, :, 0] if obs.ndim == 3 else obs
board_binary = (board > 0).astype(np.uint8)

print(f"\nBoard shape: {board.shape}")
print(f"Non-zero cells: {np.count_nonzero(board_binary)}")
print(f"Unique values: {np.unique(board)}")

# Print the board
print("\nInitial Board (first spawn):")
print("   " + "".join(str(i) for i in range(board.shape[1])))
print("   " + "-" * board.shape[1])
for r in range(board.shape[0]):
    row_str = "".join("█" if board[r, c] > 0 else "." for c in range(board.shape[1]))
    nz = np.count_nonzero(board[r, :])
    print(f"{r:2d}|{row_str}  ({nz} filled)")

# Check which rows are completely filled
print("\nCompletely filled rows:")
for r in range(board.shape[0]):
    if np.all(board[r, :] > 0):
        print(f"  Row {r}: ALL {board.shape[1]} cells filled!")

# The active piece is probably included in the board!
# Let's check the raw environment output without wrapper
print("\n" + "="*70)
print("RAW ENVIRONMENT (without CompleteVisionWrapper)")
print("="*70)

import gymnasium as gym
raw_env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
raw_obs, raw_info = raw_env.reset(seed=1)

print(f"\nRaw observation type: {type(raw_obs)}")
if isinstance(raw_obs, dict):
    print("Raw observation keys:", list(raw_obs.keys()))
    if 'board' in raw_obs:
        raw_board = raw_obs['board']
        print(f"\nRaw board shape: {raw_board.shape}")
        print(f"Raw board non-zero: {np.count_nonzero(raw_board)}")
        print(f"Raw board unique values: {np.unique(raw_board)}")

        print("\nRaw Board:")
        print("   " + "".join(str(i % 10) for i in range(raw_board.shape[1])))
        print("   " + "-" * raw_board.shape[1])
        for r in range(raw_board.shape[0]):
            row_str = "".join("█" if raw_board[r, c] > 0 else "." for c in range(raw_board.shape[1]))
            print(f"{r:2d}|{row_str}")

    if 'active_tetromino_mask' in raw_obs:
        mask = raw_obs['active_tetromino_mask']
        print(f"\nActive piece mask shape: {mask.shape}")
        print(f"Active piece mask non-zero: {np.count_nonzero(mask)}")

raw_env.close()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The CompleteVisionWrapper likely combines the board + active piece into one view.
This means the 'board' in the observation includes the falling piece!

That's why ALL columns are filled from the start - the active piece is spawning
in the middle and touching multiple columns.

The test is checking filled_columns() which counts ANY cell > 0, including
the active falling piece. This is NOT the right way to test piece placement.

The test should check the LOCKED pieces only, not the active falling piece.
""")

env.close()
