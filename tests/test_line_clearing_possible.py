#!/usr/bin/env python3
"""Test if lines can actually be cleared given movement limitations"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, ACTION_HARD_DROP
from src.reward_shaping import extract_board_from_obs

print("="*80)
print("TESTING IF LINE CLEARING IS POSSIBLE")
print("="*80)

env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)

# Run 100 episodes and check if ANY line is ever cleared
total_lines = 0
episodes_tested = 100
max_row_fullness_ever = 0

print(f"\nRunning {episodes_tested} episodes with random actions...")
print("Checking if any row ever becomes completely filled (10/10 cells)\n")

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
                print(f"✅ FULL ROW FOUND! Episode {ep}, Row {r}")

        # Check if line was cleared
        lines_cleared = info.get('number_of_lines', 0)
        if lines_cleared > 0:
            total_lines += lines_cleared
            print(f"✅ LINE CLEARED! Episode {ep}, Step {step}, Lines: {lines_cleared}")
            print(f"   Total lines so far: {total_lines}")

env.close()

print(f"\n" + "="*80)
print(f"RESULTS")
print(f"="*80)
print(f"Episodes tested: {episodes_tested}")
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness ever seen: {max_row_fullness_ever}/10 cells")
print()

if total_lines == 0:
    print("❌ CRITICAL PROBLEM: NO LINES WERE CLEARED!")
    print("   This confirms lines CANNOT be cleared with movement limitations.")
    print()
    print("   The agent is training on an IMPOSSIBLE task!")
    print("   Reward shaping expects line clears but they never happen.")
elif max_row_fullness_ever < 10:
    print(f"⚠️  PROBLEM: No row ever became completely filled!")
    print(f"   Maximum: {max_row_fullness_ever}/10 cells")
    print(f"   Yet {total_lines} lines were reported as cleared.")
    print(f"   This suggests a mismatch in how we're observing the board.")
else:
    print(f"✅ Lines CAN be cleared! {total_lines} lines in {episodes_tested} episodes")
    print(f"   Average: {total_lines/episodes_tested:.2f} lines per episode")

print("="*80)
