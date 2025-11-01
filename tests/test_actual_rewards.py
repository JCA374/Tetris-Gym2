#!/usr/bin/env python3
"""Test what rewards the agent is actually seeing"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reward_shaping import overnight_reward_shaping

print("="*80)
print("TESTING ACTUAL REWARD VALUES")
print("="*80)

# Simulate typical center-stacking board from the logs
# Column heights: [0, 0, 0, 14, 20, 19, 19, 0, 0, 0]
board = np.zeros((20, 10), dtype=np.float32)
heights = [0, 0, 0, 14, 20, 19, 19, 0, 0, 0]

for c, h in enumerate(heights):
    if h > 0:
        board[-h:, c] = 1

# Add holes (typical is ~25)
for i in range(25):
    r = np.random.randint(10, 19)
    c = np.random.choice([3, 4, 5, 6])
    if r < 20 and board[r, c] > 0:
        board[r, c] = 0

print("\nBoard configuration:")
print("  " + "0123456789")
for r in range(5):
    row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(10))
    print(f"{r:2d} {row_str}")
print("...")
for r in range(15, 20):
    row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(10))
    print(f"{r:2d} {row_str}")

print(f"\nColumn heights: {heights}")

# Calculate reward for each step
print("\n" + "="*60)
print("REWARD CALCULATION PER STEP")
print("="*60)

step_num = 14
reward_not_done = overnight_reward_shaping(
    board, action=0, reward=0.0, done=False,
    info={"steps": step_num, "lines_cleared": 0}
)

reward_done = overnight_reward_shaping(
    board, action=0, reward=0.0, done=True,
    info={"steps": step_num, "lines_cleared": 0}
)

print(f"\nPer-step reward (alive):  {reward_not_done:.2f}")
print(f"Final step reward (death): {reward_done:.2f}")

# Simulate episode
print("\n" + "="*60)
print("SIMULATED EPISODE (14 steps, center-stacking)")
print("="*60)

total_reward = 0
for step in range(1, 15):
    if step < 14:
        r = overnight_reward_shaping(
            board, action=0, reward=0.0, done=False,
            info={"steps": step, "lines_cleared": 0}
        )
    else:
        # Last step - death
        r = overnight_reward_shaping(
            board, action=0, reward=0.0, done=True,
            info={"steps": step, "lines_cleared": 0}
        )

    # Apply train.py clamp
    r_clamped = np.clip(r, -100.0, 600.0)

    print(f"Step {step:2d}: reward={r:.2f}, clamped={r_clamped:.2f}")
    total_reward += r_clamped

print(f"\nTotal episode reward: {total_reward:.2f}")

# Compare to what we see in logs
print("\n" + "="*60)
print("COMPARISON TO ACTUAL LOGS")
print("="*60)
print(f"Expected total (14 steps): {total_reward:.2f}")
print(f"Actual from log (Episode 1271): -1397.6")
print(f"Difference: {abs(total_reward - (-1397.6)):.2f}")

if abs(total_reward - (-1397.6)) < 50:
    print("\n✅ Rewards match! The shaping IS being applied correctly.")
else:
    print("\n❌ Rewards don't match! Something else is happening.")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\nThe agent IS seeing the penalties!")
print("But it's getting -100 per step (hitting the clamp).")
print()
print("The problem is NOT that penalties are too weak.")
print("The problem is that ALL strategies result in -100/step!")
print()
print("Because:")
print("  - Center-stacking gets -100/step (clamped)")
print("  - ANY other strategy ALSO gets heavily penalized early on")
print("  - Agent has no gradient to learn from!")
print()
print("Solution: The agent dies too fast (14 steps)")
print("         It never experiences longer episodes where")
print("         balanced play would give better rewards.")
