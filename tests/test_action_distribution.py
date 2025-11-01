#!/usr/bin/env python3
"""Test if agent/random policy uses LEFT actions enough to reach outer columns"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env
import numpy as np

print("="*80)
print("ACTION DISTRIBUTION TEST")
print("="*80)

env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)

# Test 1: Pure random policy
print("\nTest 1: Random action policy")
print("="*60)

action_counts = {i: 0 for i in range(8)}
column_usage = [0] * 10

obs, _ = env.reset()

for ep in range(50):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 500:
        action = env.action_space.sample()  # Random
        action_counts[action] += 1

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1

    # Check column usage
    # Extract board from observation
    if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
        board = obs[:, :, 0]  # Channel 0 = board
    else:
        board = obs

    for c in range(10):
        if np.any(board[:, c] > 0):
            column_usage[c] += 1

print("\nRandom action distribution:")
action_names = ["LEFT", "RIGHT", "DOWN", "ROT_CW", "ROT_CCW", "HARD_DROP", "SWAP", "NOOP"]
total_actions = sum(action_counts.values())
for i, name in enumerate(action_names):
    pct = 100 * action_counts[i] / total_actions if total_actions > 0 else 0
    print(f"  {i} {name:10s}: {action_counts[i]:5d} ({pct:5.1f}%)")

print(f"\nColumn usage (out of 50 episodes):")
for c in range(10):
    bar = "█" * (column_usage[c] // 2)
    print(f"  Column {c}: {column_usage[c]:2d} {bar}")

unused = [c for c in range(10) if column_usage[c] == 0]
if unused:
    print(f"\n❌ Unused columns: {unused}")
    print(f"   Random policy should use all columns - this indicates a problem!")
else:
    print(f"\n✅ All columns used by random policy")

# Test 2: Biased policy (more LEFT actions)
print("\n" + "="*60)
print("Test 2: Biased policy (50% LEFT, 20% RIGHT, 30% HARD_DROP)")
print("="*60)

column_usage_biased = [0] * 10

for ep in range(50):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 500:
        # Biased policy: favor LEFT
        rand = np.random.random()
        if rand < 0.5:
            action = 0  # LEFT
        elif rand < 0.7:
            action = 1  # RIGHT
        else:
            action = 5  # HARD_DROP

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1

    # Check column usage
    if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
        board = obs[:, :, 0]
    else:
        board = obs

    for c in range(10):
        if np.any(board[:, c] > 0):
            column_usage_biased[c] += 1

print(f"\nColumn usage with LEFT-biased policy:")
for c in range(10):
    bar = "█" * (column_usage_biased[c] // 2)
    print(f"  Column {c}: {column_usage_biased[c]:2d} {bar}")

left_cols_used = sum(1 for c in [0, 1, 2, 3] if column_usage_biased[c] > 0)
right_cols_used = sum(1 for c in [6, 7, 8, 9] if column_usage_biased[c] > 0)

print()
if left_cols_used >= 3:
    print(f"✅ LEFT-biased policy uses leftmost columns ({left_cols_used}/4)")
else:
    print(f"❌ LEFT-biased policy STILL doesn't use left columns ({left_cols_used}/4)")
    print(f"   This indicates a fundamental environment issue!")

env.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If random policy doesn't use all columns → environment bug")
print("If LEFT-biased policy uses left columns → agent needs better exploration")
print("="*80)
