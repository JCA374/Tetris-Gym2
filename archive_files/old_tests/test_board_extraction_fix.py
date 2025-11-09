#!/usr/bin/env python3
"""Test that board extraction works with 4-channel observations"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env
from src.reward_shaping import (
    extract_board_from_obs,
    get_column_heights,
    count_holes,
    calculate_bumpiness
)

print("="*80)
print("TESTING BOARD EXTRACTION WITH 4-CHANNEL OBSERVATIONS")
print("="*80)

# Create 4-channel environment
env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
obs, _ = env.reset(seed=42)

print(f"\n1️⃣ OBSERVATION INFO")
print(f"   Shape: {obs.shape}")
print(f"   Type: {type(obs)}")

# Extract board using the reward shaping function
board = extract_board_from_obs(obs)

print(f"\n2️⃣ EXTRACTED BOARD")
print(f"   Shape: {board.shape}")
print(f"   Expected: (20, 10)")
print(f"   Match: {'✅' if board.shape == (20, 10) else '❌'}")

print(f"\n3️⃣ BOARD CONTENT")
print(f"   Non-zero cells: {np.count_nonzero(board)}")
print(f"   Min value: {board.min()}")
print(f"   Max value: {board.max()}")

# Calculate metrics
heights = get_column_heights(board)
holes = count_holes(board)
bumpiness = calculate_bumpiness(board)

print(f"\n4️⃣ METRICS (should NOT be all zeros)")
print(f"   Column heights: {heights}")
print(f"   Max height: {max(heights) if heights else 0}")
print(f"   Holes: {holes}")
print(f"   Bumpiness: {bumpiness}")

# Check if all zeros (the bug)
all_zeros = all(h == 0 for h in heights)
if all_zeros:
    print(f"\n   ❌ BUG STILL PRESENT: All heights are zero!")
else:
    print(f"\n   ✅ FIX WORKING: Heights detected!")

# Place some pieces and test again
print(f"\n5️⃣ GAMEPLAY TEST")
print(f"   Placing 10 pieces...")

for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        print(f"   Game ended after {i+1} steps")
        break

# Extract board from final observation
final_board = extract_board_from_obs(obs)
final_heights = get_column_heights(final_board)
final_holes = count_holes(final_board)
final_bumpiness = calculate_bumpiness(final_board)

print(f"\n   After gameplay:")
print(f"     Board cells: {np.count_nonzero(final_board)}")
print(f"     Column heights: {final_heights}")
print(f"     Max height: {max(final_heights) if final_heights else 0}")
print(f"     Holes: {final_holes}")
print(f"     Bumpiness: {final_bumpiness:.1f}")

# Visual check
max_row_fullness = 0
for r in range(final_board.shape[0]):
    filled = int((final_board[r, :] > 0).sum())
    max_row_fullness = max(max_row_fullness, filled)

print(f"     Max row fullness: {max_row_fullness}/10 cells")

env.close()

print(f"\n" + "="*80)
print(f"RESULT")
print(f"="*80)

if all_zeros:
    print(f"❌ Board extraction still broken - metrics all zero")
else:
    print(f"✅ Board extraction FIXED - metrics working correctly!")

print(f"="*80)
