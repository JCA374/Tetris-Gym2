#!/usr/bin/env python3
"""Test line clearing with RAW environment (no wrapper)"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import tetris_gymnasium  # Register the environment

print("="*80)
print("TESTING RAW TETRIS-GYMNASIUM ENVIRONMENT")
print("="*80)

# Create RAW environment without our wrapper
# Use the direct class to avoid registration issues
from tetris_gymnasium.envs import Tetris
env = Tetris(render_mode=None)

total_lines = 0
episodes_tested = 50

print(f"\nRunning {episodes_tested} episodes with random actions...")
print("Testing if tetris-gymnasium itself can clear lines\n")

for ep in range(episodes_tested):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 500:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1

        # Check raw board
        raw_board = obs['board']
        playable = raw_board[0:20, 4:14]  # Our extraction

        # Check for full rows
        for r in range(playable.shape[0]):
            row_fullness = np.count_nonzero(playable[r, :])
            if row_fullness == 10:
                print(f"✅ FULL ROW at episode {ep}, row {r}!")

        # Check environment's line clear reporting
        lines_cleared = info.get('number_of_lines', 0)
        if lines_cleared > 0:
            total_lines += lines_cleared
            print(f"✅ Environment reported {lines_cleared} lines cleared at episode {ep}")

    if (ep + 1) % 10 == 0:
        print(f"Progress: {ep+1}/{episodes_tested} episodes, {total_lines} lines cleared so far")

env.close()

print(f"\n" + "="*80)
print(f"RAW ENVIRONMENT RESULTS")
print(f"="*80)
print(f"Episodes tested: {episodes_tested}")
print(f"Total lines cleared: {total_lines}")
print(f"Average: {total_lines/episodes_tested:.2f} lines per episode")
print()

if total_lines == 0:
    print("❌ RAW tetris-gymnasium CANNOT clear lines either!")
    print("   This is a fundamental limitation of tetris-gymnasium v0.3.0")
    print()
    print("   RECOMMENDATION: Use a different Tetris environment")
elif total_lines > 0:
    print("✅ RAW environment CAN clear lines!")
    print("   The problem is with OUR wrapper extraction or observation")
    print()
    print("   Need to investigate wrapper code")

print("="*80)
