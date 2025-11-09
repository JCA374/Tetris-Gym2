#!/usr/bin/env python3
"""Complete test of the reward system - verify it actually works"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reward_shaping import (
    overnight_reward_shaping,
    get_column_heights,
    calculate_bumpiness,
    count_holes,
    calculate_horizontal_distribution
)

print("="*80)
print("COMPLETE REWARD SYSTEM TEST")
print("="*80)

def create_board(heights):
    """Create a board with specified column heights"""
    board = np.zeros((20, 10), dtype=np.float32)
    for c, h in enumerate(heights):
        if h > 0:
            board[-h:, c] = 1
    return board

def print_board(board, title="Board"):
    """Print board visualization"""
    print(f"\n{title}:")
    print("  " + "0123456789")
    for r in range(min(10, board.shape[0])):
        row_str = "".join("█" if board[r, c] > 0 else "·" for c in range(board.shape[1]))
        print(f"{r:2d} {row_str}")

def analyze_board(board, name, step=10):
    """Analyze a board configuration and calculate reward"""
    heights = get_column_heights(board)
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(board)
    spread = calculate_horizontal_distribution(board)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    print_board(board[:10, :])

    print(f"\nMetrics:")
    print(f"  Column heights: {heights}")
    print(f"  Holes: {holes}")
    print(f"  Bumpiness: {bumpiness:.1f}")
    print(f"  Spread (0-1): {spread:.3f}")

    # Calculate reward for this state
    reward_alive = overnight_reward_shaping(
        board, action=0, reward=0.0, done=False,
        info={"steps": step, "lines_cleared": 0}
    )

    reward_death = overnight_reward_shaping(
        board, action=0, reward=0.0, done=True,
        info={"steps": step, "lines_cleared": 0}
    )

    print(f"\nRewards:")
    print(f"  Per step (alive): {reward_alive:+.2f}")
    print(f"  At death: {reward_death:+.2f}")

    return reward_alive, heights, holes, bumpiness

# Test 1: Center-stacking (what the agent does)
print("\n" + "="*80)
print("TEST 1: CENTER-STACKING (Current Agent Behavior)")
print("="*80)

board1 = create_board([0, 0, 0, 15, 19, 20, 19, 0, 0, 0])
r1, h1, holes1, bump1 = analyze_board(board1, "Center-Stacking", step=10)

# Test 2: Slight spreading
print("\n" + "="*80)
print("TEST 2: SLIGHT SPREADING (Some Learning)")
print("="*80)

board2 = create_board([0, 0, 2, 10, 15, 16, 12, 8, 0, 0])
r2, h2, holes2, bump2 = analyze_board(board2, "Slight Spreading", step=10)

# Test 3: Good spreading
print("\n" + "="*80)
print("TEST 3: GOOD SPREADING (Target Behavior)")
print("="*80)

board3 = create_board([2, 4, 7, 10, 12, 11, 9, 6, 3, 1])
r3, h3, holes3, bump3 = analyze_board(board3, "Good Spreading", step=10)

# Test 4: Perfect balance
print("\n" + "="*80)
print("TEST 4: PERFECT BALANCE (Ideal)")
print("="*80)

board4 = create_board([8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
r4, h4, holes4, bump4 = analyze_board(board4, "Perfect Balance", step=10)

# Test 5: Empty board (start)
print("\n" + "="*80)
print("TEST 5: EMPTY BOARD (Game Start)")
print("="*80)

board5 = create_board([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
r5, h5, holes5, bump5 = analyze_board(board5, "Empty Board", step=1)

# SUMMARY
print("\n" + "="*80)
print("SUMMARY: REWARD COMPARISON")
print("="*80)

results = [
    ("Empty board (start)", r5),
    ("Center-stacking", r1),
    ("Slight spreading", r2),
    ("Good spreading", r3),
    ("Perfect balance", r4),
]

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\nRanked from BEST to WORST reward:")
for i, (name, reward) in enumerate(results_sorted, 1):
    print(f"{i}. {name:30s} → {reward:+7.2f}")

print("\n" + "="*80)
print("GRADIENT ANALYSIS")
print("="*80)

gradient = r3 - r1  # Good spreading minus center-stacking
print(f"\nReward difference (Good - Center): {gradient:+.2f}")
print(f"Percentage improvement: {abs(gradient/r1)*100:.1f}%")

if gradient > 30:
    print("\n✅ STRONG GRADIENT - Agent should be able to learn!")
    print(f"   The {gradient:.0f} point difference is enough for Q-learning.")
elif gradient > 10:
    print("\n⚠️  MODERATE GRADIENT - Agent might learn slowly")
    print(f"   The {gradient:.0f} point difference is small but might work.")
else:
    print("\n❌ WEAK GRADIENT - Agent will struggle to learn!")
    print(f"   The {gradient:.0f} point difference is too small.")

print("\n" + "="*80)
print("DIAGNOSTICS")
print("="*80)

# Check if rewards make sense
print("\n1. Are better strategies rewarded more?")
if r4 > r3 > r2 > r1:
    print("   ✅ YES - Perfect > Good > Slight > Center")
else:
    print("   ❌ NO - Reward ordering is wrong!")
    print(f"      Perfect: {r4:.2f}")
    print(f"      Good:    {r3:.2f}")
    print(f"      Slight:  {r2:.2f}")
    print(f"      Center:  {r1:.2f}")

print("\n2. Is center-stacking penalized?")
if r1 < 0:
    print(f"   ✅ YES - Center-stacking gets {r1:.2f}")
else:
    print(f"   ❌ NO - Center-stacking gets positive reward: {r1:.2f}")

print("\n3. Is the gradient large enough?")
reward_range = r4 - r1
if reward_range > 50:
    print(f"   ✅ YES - Range is {reward_range:.2f} points")
else:
    print(f"   ❌ NO - Range is only {reward_range:.2f} points")

print("\n4. Are all rewards negative?")
if r1 < 0 and r2 < 0 and r3 < 0 and r4 < 0:
    print("   ⚠️  WARNING - All strategies give negative rewards!")
    print("      Agent might not survive long enough to learn.")
    print("      Consider increasing survival bonus or reducing penalties.")
elif r4 > 0:
    print(f"   ✅ GOOD - Best strategy gives positive reward: {r4:.2f}")
else:
    print(f"   ⚠️  Mixed - Some positive, some negative")

print("\n" + "="*80)
print("EPISODE SIMULATION")
print("="*80)

# Simulate a full episode for each strategy
print("\nSimulating 50-step episodes:")

for name, board in [("Center-stack", board1), ("Good spread", board3)]:
    total = 0
    for step in range(1, 51):
        if step < 50:
            r = overnight_reward_shaping(
                board, action=0, reward=0.0, done=False,
                info={"steps": step, "lines_cleared": 0}
            )
        else:
            r = overnight_reward_shaping(
                board, action=0, reward=0.0, done=True,
                info={"steps": step, "lines_cleared": 0}
            )
        total += r

    print(f"\n{name}:")
    print(f"  Total reward (50 steps): {total:.2f}")
    print(f"  Average per step: {total/50:.2f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if gradient < 10:
    print("\n❌ PROBLEM: Gradient too weak!")
    print("\nSuggested fixes:")
    print("  1. INCREASE spread bonus (currently 15.0)")
    print("  2. INCREASE column usage bonus (currently 4.0)")
    print("  3. DECREASE penalties slightly")
    print("  4. INCREASE survival bonus")
elif abs(r1) > 80:
    print("\n⚠️  WARNING: Penalties might be too harsh!")
    print(f"\nCenter-stacking gives {r1:.2f} per step.")
    print("If agent dies in 10 steps, total reward is ~" + str(r1*10))
    print("\nThis might prevent exploration.")
    print("Consider reducing penalties by 20-30%.")
else:
    print("\n✅ REWARD SYSTEM LOOKS GOOD!")
    print(f"\nGradient: {gradient:.2f} points")
    print(f"Range: {r4:.2f} to {r1:.2f}")
    print("\nIf agent is still center-stacking, the problem is likely:")
    print("  1. Epsilon too low (not enough exploration)")
    print("  2. Agent dying too fast (needs to survive longer)")
    print("  3. Q-network not learning (check loss values)")

print("\n" + "="*80)
