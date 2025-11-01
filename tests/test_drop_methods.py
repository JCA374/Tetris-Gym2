#!/usr/bin/env python3
"""Test different ways to drop pieces - find one that preserves position"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("DROP METHOD COMPARISON")
print("="*80)

# Method 1: HARD_DROP (action 5)
print("\nMethod 1: Move LEFT + HARD_DROP (action 5)")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=100)

for _ in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_before = np.where(mask.sum(axis=0) > 0)[0]
print(f"Before drop: columns {cols_before.min()}-{cols_before.max()}")

obs, _, _, _, _ = env.step(5)  # HARD_DROP

board = obs['board'][0:20, 4:14]
for r in range(20):
    if np.any(board[r, :] > 0):
        cols_landed = np.where(board[r, :] > 0)[0]
        print(f"After HARD_DROP: columns {cols_landed.min()}-{cols_landed.max()}")
        break

env.close()

# Method 2: Just gravity (NOOP or other action)
print("\nMethod 2: Move LEFT + let gravity drop (NOOP action 7)")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=100)

for _ in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_before = np.where(mask.sum(axis=0) > 0)[0]
print(f"Before drop: columns {cols_before.min()}-{cols_before.max()}")

# Let gravity drop it using NOOP
for i in range(30):
    obs, _, term, trunc, _ = env.step(7)  # NOOP
    if term or trunc:
        break

board = obs['board'][0:20, 4:14]
for r in range(20):
    if np.any(board[r, :] > 0):
        cols_landed = np.where(board[r, :] > 0)[0]
        print(f"After gravity NOOP: columns {cols_landed.min()}-{cols_landed.max()}")
        break

env.close()

# Method 3: DOWN action repeatedly
print("\nMethod 3: Move LEFT + DOWN action (action 2)")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=100)

for _ in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_before = np.where(mask.sum(axis=0) > 0)[0]
print(f"Before drop: columns {cols_before.min()}-{cols_before.max()}")

for i in range(30):
    obs, _, term, trunc, _ = env.step(2)  # DOWN
    if term or trunc:
        break

board = obs['board'][0:20, 4:14]
for r in range(20):
    if np.any(board[r, :] > 0):
        cols_landed = np.where(board[r, :] > 0)[0]
        print(f"After DOWN: columns {cols_landed.min()}-{cols_landed.max()}")
        break

env.close()

# Method 4: Check the raw environment's internal state
print("\nMethod 4: Check internal piece position (self.x)")
print("="*60)

env = Tetris(render_mode=None)
obs, _ = env.reset(seed=100)

print(f"Initial self.x (piece X position): {env.x}")

for i in range(8):
    obs, _, _, _, _ = env.step(0)  # LEFT
    print(f"After LEFT {i+1}: self.x = {env.x}")

mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_mask = np.where(mask.sum(axis=0) > 0)[0]
print(f"\nMask shows columns: {cols_mask.min()}-{cols_mask.max()}")
print(f"But self.x = {env.x}")
print(f"Padding offset: self.padding = {env.padding}")

# Now HARD_DROP and check what happens
print(f"\nBefore HARD_DROP: self.x = {env.x}")
obs, _, _, _, _ = env.step(5)  # HARD_DROP
print(f"After HARD_DROP: self.x = {env.x}")

board = obs['board'][0:20, 4:14]
for r in range(20):
    if np.any(board[r, :] > 0):
        cols_landed = np.where(board[r, :] > 0)[0]
        print(f"Piece landed in columns: {cols_landed.min()}-{cols_landed.max()}")
        break

env.close()

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("If all methods land in CENTER despite moving LEFT,")
print("there's a fundamental issue with the environment.")
print()
print("If some methods preserve position, we need to use those.")
print("="*80)
