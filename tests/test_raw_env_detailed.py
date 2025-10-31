#!/usr/bin/env python3
"""Detailed test of raw environment - check info dict and board state"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("DETAILED RAW ENVIRONMENT ANALYSIS")
print("="*80)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=42)

print(f"\nInitial info dict keys: {info.keys()}")
print(f"Initial info dict: {info}")

# Run one episode and track everything
total_lines = 0
max_row_fullness = 0
step_count = 0

print(f"\nRunning one episode with random actions...\n")

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    step_count += 1

    # Extract board and check fullness
    board = obs['board']  # Raw 24x18
    playable = board[0:20, 4:14]  # Extract playable area (20x10)

    # Check each row for fullness
    for r in range(playable.shape[0]):
        row_fullness = np.count_nonzero(playable[r, :])
        max_row_fullness = max(max_row_fullness, row_fullness)

        if row_fullness == 10:
            print(f"✅ FULL ROW at step {step_count}, row {r}!")
            print(f"   Board state:")
            for rr in range(max(0, r-2), min(20, r+3)):
                row_str = "".join("█" if playable[rr, c] > 0 else "." for c in range(10))
                marker = " <-- FULL" if rr == r else ""
                print(f"      Row {rr:2d}: {row_str}{marker}")

    # Check all possible keys for line clear info
    if i < 5:  # Print first 5 steps to see what info contains
        print(f"Step {step_count} info: {info}")

    # Try different possible keys
    lines_cleared = info.get('number_of_lines', 0)
    if lines_cleared > 0:
        total_lines += lines_cleared
        print(f"✅ LINES CLEARED at step {step_count}: {lines_cleared}")
        print(f"   Info: {info}")
        print(f"   Reward: {reward}")

    if term or trunc:
        print(f"\n Episode ended at step {step_count}")
        break

env.close()

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Steps: {step_count}")
print(f"Total lines cleared: {total_lines}")
print(f"Max row fullness ever: {max_row_fullness}/10 cells")
print()

if total_lines == 0 and max_row_fullness < 10:
    print(f"⚠️  No full rows achieved (max {max_row_fullness}/10)")
    print(f"   Random actions might not be enough to test line clearing")
elif total_lines == 0 and max_row_fullness == 10:
    print(f"❌ Full rows achieved but NOT cleared!")
    print(f"   This would indicate a library bug")
else:
    print(f"✅ Line clearing works!")

print("="*80)
