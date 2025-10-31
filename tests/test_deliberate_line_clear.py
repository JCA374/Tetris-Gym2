#!/usr/bin/env python3
"""Deliberately try to clear a line - drop pieces straight down"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("DELIBERATE LINE CLEARING ATTEMPT")
print("="*80)

env = Tetris(render_mode=None)

# Strategy: Just use HARD_DROP (action 5) repeatedly to stack pieces
# This should fill up the bottom rows quickly
print("\nStrategy: Spam HARD_DROP to stack pieces vertically\n")

total_lines = 0
max_row_fullness = 0

for episode in range(10):
    obs, info = env.reset(seed=episode)
    done = False
    step = 0

    while not done and step < 200:
        # Action 5 = HARD_DROP
        obs, reward, term, trunc, info = env.step(5)
        done = term or trunc
        step += 1

        # Check board
        board = obs['board']
        playable = board[0:20, 4:14]

        # Check row fullness
        for r in range(playable.shape[0]):
            row_full = np.count_nonzero(playable[r, :])
            max_row_fullness = max(max_row_fullness, row_full)

        # Check for lines cleared
        lines = info.get('lines_cleared', 0)
        if lines > 0:
            total_lines += lines
            print(f"✅ Episode {episode}, Step {step}: {lines} LINE(S) CLEARED!")
            print(f"   Total so far: {total_lines}")
            print(f"   Reward: {reward}")

    if episode % 2 == 0:
        print(f"Episode {episode} done: {step} steps, max fullness: {max_row_fullness}/10")

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Episodes: 10")
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness ever: {max_row_fullness}/10 cells")
print()

if total_lines > 0:
    print(f"✅ LINE CLEARING WORKS!")
else:
    print(f"❌ No lines cleared even with deliberate stacking")
    print(f"   Max fullness: {max_row_fullness}/10")
    if max_row_fullness < 10:
        print(f"   Rows never became completely full")

print("="*80)
