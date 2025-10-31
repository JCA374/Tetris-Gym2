#!/usr/bin/env python3
"""Test with CORRECT action mapping - can we clear lines now?"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, ACTION_HARD_DROP
from src.reward_shaping import extract_board_from_obs
import numpy as np

print("="*80)
print("TEST WITH CORRECT ACTION MAPPING")
print("="*80)

print(f"\nAction mapping:")
print(f"  LEFT=0, RIGHT=1, DOWN=2, ROTATE_CW=3")
print(f"  ROTATE_CCW=4, HARD_DROP=5, SWAP=6, NOOP=7")
print(f"  Using HARD_DROP = {ACTION_HARD_DROP}\n")

env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)

total_lines = 0
episodes_tested = 100
max_row_fullness_ever = 0

print(f"Running {episodes_tested} episodes with random actions...\n")

for ep in range(episodes_tested):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 500:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1

        # Check board state
        board = extract_board_from_obs(obs)

        # Check each row
        for r in range(board.shape[0]):
            row_fullness = np.count_nonzero(board[r, :])
            max_row_fullness_ever = max(max_row_fullness_ever, row_fullness)

            if row_fullness == 10:
                print(f"✅ FULL ROW at episode {ep}, row {r}!")

        # Check if line was cleared (CORRECT KEY: 'lines_cleared')
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            total_lines += lines_cleared
            print(f"✅ LINE CLEARED! Episode {ep}, Step {step}, Lines: {lines_cleared}")
            print(f"   Total lines so far: {total_lines}")

    if (ep + 1) % 20 == 0:
        print(f"Progress: {ep+1}/{episodes_tested} episodes, {total_lines} lines cleared")

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Episodes tested: {episodes_tested}")
print(f"Total lines cleared: {total_lines}")
print(f"Average: {total_lines/episodes_tested:.2f} lines per episode")
print(f"Max row fullness ever: {max_row_fullness_ever}/10 cells")
print()

if total_lines > 0:
    print(f"✅ SUCCESS! Lines CAN be cleared with correct action mapping!")
    print(f"   The problem was the WRONG action mapping in config.py")
    print(f"   Your agent was training with inverted controls!")
else:
    print(f"❌ Still no lines cleared")
    print(f"   Max row fullness: {max_row_fullness_ever}/10")
    if max_row_fullness_ever < 10:
        print(f"   No row ever became full (movement still limited?)")

print("="*80)
