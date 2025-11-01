#!/usr/bin/env python3
"""Test the new anti-center-stacking reward shaping"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reward_shaping import overnight_reward_shaping

print("="*80)
print("ANTI-CENTER-STACKING REWARD TEST")
print("="*80)

# Create test boards with different column height patterns

# Scenario 1: Severe center-stacking (user's problem)
print("\nScenario 1: SEVERE CENTER-STACKING")
print("Column heights: [0, 0, 0, 19, 19, 19, 19, 0, 0, 0]")
print("-" * 60)

board1 = np.zeros((20, 10), dtype=np.float32)
# Fill columns 3-6 to height 19
for c in [3, 4, 5, 6]:
    board1[-19:, c] = 1

obs1 = board1  # Simple 2D observation
reward1 = overnight_reward_shaping(obs1, action=0, reward=0.0, done=False,
                                   info={"steps": 100, "lines_cleared": 0})

print(f"Shaped reward: {reward1:.2f}")

# Scenario 2: Moderate center-bias
print("\nScenario 2: MODERATE CENTER-BIAS")
print("Column heights: [0, 2, 5, 10, 15, 14, 9, 6, 3, 0]")
print("-" * 60)

board2 = np.zeros((20, 10), dtype=np.float32)
heights2 = [0, 2, 5, 10, 15, 14, 9, 6, 3, 0]
for c, h in enumerate(heights2):
    if h > 0:
        board2[-h:, c] = 1

reward2 = overnight_reward_shaping(board2, action=0, reward=0.0, done=False,
                                   info={"steps": 100, "lines_cleared": 0})

print(f"Shaped reward: {reward2:.2f}")

# Scenario 3: Good balanced distribution
print("\nScenario 3: BALANCED DISTRIBUTION (GOOD!)")
print("Column heights: [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]")
print("-" * 60)

board3 = np.zeros((20, 10), dtype=np.float32)
heights3 = [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
for c, h in enumerate(heights3):
    if h > 0:
        board3[-h:, c] = 1

reward3 = overnight_reward_shaping(board3, action=0, reward=0.0, done=False,
                                   info={"steps": 100, "lines_cleared": 0})

print(f"Shaped reward: {reward3:.2f}")

# Scenario 4: Perfect even distribution
print("\nScenario 4: PERFECT EVEN DISTRIBUTION")
print("Column heights: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]")
print("-" * 60)

board4 = np.zeros((20, 10), dtype=np.float32)
for c in range(10):
    board4[-8:, c] = 1

reward4 = overnight_reward_shaping(board4, action=0, reward=0.0, done=False,
                                   info={"steps": 100, "lines_cleared": 0})

print(f"Shaped reward: {reward4:.2f}")

# Scenario 5: Using all columns but some low
print("\nScenario 5: ALL COLUMNS USED (even if some low)")
print("Column heights: [2, 3, 5, 12, 15, 14, 10, 6, 4, 2]")
print("-" * 60)

board5 = np.zeros((20, 10), dtype=np.float32)
heights5 = [2, 3, 5, 12, 15, 14, 10, 6, 4, 2]
for c, h in enumerate(heights5):
    if h > 0:
        board5[-h:, c] = 1

reward5 = overnight_reward_shaping(board5, action=0, reward=0.0, done=False,
                                   info={"steps": 100, "lines_cleared": 0})

print(f"Shaped reward: {reward5:.2f}")

print("\n" + "="*80)
print("REWARD COMPARISON")
print("="*80)

scenarios = [
    ("Severe center-stacking [0,0,0,19,19,19,19,0,0,0]", reward1),
    ("Moderate center-bias [0,2,5,10,15,14,9,6,3,0]", reward2),
    ("Balanced [5,6,8,10,12,11,9,7,5,4]", reward3),
    ("Perfect even [8,8,8,8,8,8,8,8,8,8]", reward4),
    ("All columns used [2,3,5,12,15,14,10,6,4,2]", reward5),
]

# Sort by reward
scenarios_sorted = sorted(scenarios, key=lambda x: x[1], reverse=True)

print("\nRanked by reward (best to worst):")
for i, (desc, reward) in enumerate(scenarios_sorted):
    print(f"{i+1}. {desc:50s} → {reward:+7.2f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if reward1 < reward3 and reward1 < reward5:
    print("✅ Center-stacking is PENALIZED (lower reward)")
else:
    print("❌ Center-stacking is NOT penalized enough")

if reward3 > reward1 or reward5 > reward1:
    print("✅ Balanced/wide distribution is REWARDED")
else:
    print("❌ Balanced distribution is not rewarded enough")

reward_diff = reward3 - reward1
print(f"\nReward difference (Balanced - Center-stacked): {reward_diff:+.2f}")

if abs(reward_diff) < 10:
    print("⚠️  Difference too small - agent might not learn")
elif abs(reward_diff) > 20:
    print("✅ Strong signal - agent should learn to avoid center-stacking")
else:
    print("✅ Moderate signal - should help but may need tuning")

print("="*80)
