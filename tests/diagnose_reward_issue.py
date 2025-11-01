#!/usr/bin/env python3
"""Diagnose why agent isn't learning to avoid center-stacking"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reward_shaping import overnight_reward_shaping, calculate_bumpiness, get_column_heights

print("="*80)
print("REWARD SHAPING DIAGNOSIS")
print("="*80)

# Simulate what the agent sees with center-stacking
print("\n1. CENTER-STACKING (what your agent does)")
print("-" * 60)

board_center = np.zeros((20, 10), dtype=np.float32)
heights_center = [0, 0, 0, 13, 19, 18, 18, 0, 0, 0]
for c, h in enumerate(heights_center):
    if h > 0:
        board_center[-h:, c] = 1

# Add holes (agent creates these)
for i in range(28):
    r = np.random.randint(10, 19)
    c = np.random.choice([3, 4, 5, 6])
    board_center[r, c] = 0

bumpiness_center = calculate_bumpiness(board_center)
print(f"Heights: {heights_center}")
print(f"Bumpiness: {bumpiness_center}")

# Calculate reward per step (13 steps total)
reward_step = overnight_reward_shaping(
    board_center, action=0, reward=0.0, done=False,
    info={"steps": 13, "lines_cleared": 0}
)
reward_death = overnight_reward_shaping(
    board_center, action=0, reward=0.0, done=True,
    info={"steps": 13, "lines_cleared": 0}
)

print(f"Reward per step (alive): {reward_step:.2f}")
print(f"Reward at death: {reward_death:.2f}")
print(f"Total episode reward (13 steps): {12 * reward_step + reward_death:.2f}")

# Simulate what GOOD play would look like
print("\n2. BALANCED SPREADING (good play)")
print("-" * 60)

board_good = np.zeros((20, 10), dtype=np.float32)
heights_good = [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
for c, h in enumerate(heights_good):
    if h > 0:
        board_good[-h:, c] = 1

# Few holes
for i in range(2):
    r = np.random.randint(15, 19)
    c = np.random.randint(0, 10)
    board_good[r, c] = 0

bumpiness_good = calculate_bumpiness(board_good)
print(f"Heights: {heights_good}")
print(f"Bumpiness: {bumpiness_good}")

# This agent survives much longer - let's say 180 steps
reward_step_good = overnight_reward_shaping(
    board_good, action=0, reward=0.0, done=False,
    info={"steps": 100, "lines_cleared": 0}
)
reward_with_lines = overnight_reward_shaping(
    board_good, action=0, reward=1.0, done=False,
    info={"steps": 100, "lines_cleared": 2}  # Cleared 2 lines!
)

print(f"Reward per step (no lines): {reward_step_good:.2f}")
print(f"Reward when clearing 2 lines: {reward_with_lines:.2f}")
print(f"Total episode reward (180 steps, 8 lines): ~{180 * reward_step_good + 4 * reward_with_lines:.2f}")

print("\n" + "="*80)
print("ANALYSIS: WHY ISN'T THE AGENT LEARNING?")
print("="*80)

print("\n🔴 PROBLEM #1: Agent Dies Too Fast (Exploration)")
print("-" * 60)
print("Your agent survives only 13-15 steps per episode.")
print("It NEVER experiences what happens with balanced play!")
print()
print("With center-stacking:")
print("  - Dies in ~13 steps")
print("  - Never clears lines")
print("  - Never sees the reward for balanced play")
print()
print("The agent CAN'T learn 'spreading is good' because it never tries it!")

print("\n🔴 PROBLEM #2: Penalties Too Weak")
print("-" * 60)

print(f"Bumpiness penalty: -0.06 * {bumpiness_center:.0f} = {-0.06 * bumpiness_center:.2f}")
print(f"  ↑ TOO WEAK! Should be at least -1.0 per bumpiness point")
print()

heights_center_arr = np.array(heights_center)
std_center = np.std(heights_center_arr)
print(f"Height std dev penalty: -0.5 * {std_center:.1f} = {-0.5 * std_center:.2f}")
print(f"  ↑ TOO WEAK! Center-stacking has huge variance but tiny penalty")

print("\n🔴 PROBLEM #3: Agent Exploration Settings")
print("-" * 60)
print("Need to check:")
print("  1. Is epsilon high enough? (should be >0.5 early)")
print("  2. Is agent actually using LEFT actions? (25% in exploration)")
print("  3. Is Q-network learning? (check loss values)")

print("\n" + "="*80)
print("RECOMMENDED FIXES")
print("="*80)

print("\n1. INCREASE BUMPINESS PENALTY (10x stronger)")
print("   Change: shaped -= 0.06 * bump")
print("   To:     shaped -= 1.0 * bump")
print()

print("2. INCREASE HEIGHT STD DEV PENALTY (20x stronger)")
print("   Change: shaped -= 0.5 * height_std")
print("   To:     shaped -= 10.0 * height_std")
print()

print("3. INCREASE OUTER COLUMN PENALTY (2x stronger)")
print("   Change: shaped -= outer_unused * 5.0")
print("   To:     shaped -= outer_unused * 10.0")
print()

print("4. ADD SURVIVAL INCENTIVE (encourage longer episodes)")
print("   Change: shaped += min(steps * 0.02, 3.0)")
print("   To:     shaped += min(steps * 0.1, 10.0)")
print()

print("5. REDUCE DEATH PENALTY (currently too harsh)")
print("   Change: shaped -= 30.0")
print("   To:     shaped -= 10.0")
print("   Reason: -30 discourages exploration")

print("\n" + "="*80)
