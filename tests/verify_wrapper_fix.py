#!/usr/bin/env python3
"""Test the proposed CompleteVisionWrapper fix"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env

print("="*80)
print("VERIFICATION TEST: Proposed CompleteVisionWrapper Fix")
print("="*80)

# Create environment WITHOUT wrapper to get raw dict
env_raw = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
obs_raw, _ = env_raw.reset(seed=42)

raw_board = obs_raw['board']
print(f"\n1️⃣ RAW BOARD STRUCTURE")
print(f"   Shape: {raw_board.shape}")
print(f"   Rows 20-23 (expected walls):")
for r in range(20, 24):
    playable = raw_board[r, 4:14]
    all_filled = np.all(playable > 0)
    print(f"     Row {r}: {'WALL (all filled)' if all_filled else 'Not a wall'}")

# Simulate CURRENT (buggy) extraction
current_extraction = raw_board[2:22, 4:14]
current_extraction = (current_extraction > 0).astype(np.uint8)

print(f"\n2️⃣ CURRENT EXTRACTION [2:22, 4:14] - BUGGY")
print(f"   Shape: {current_extraction.shape}")
print(f"   Bottom 2 rows:")
for r in [-2, -1]:
    row = current_extraction[r, :]
    all_filled = np.all(row > 0)
    row_str = "".join("█" if c else "." for c in row)
    print(f"     Row {r}: {row_str} {'← WALL!' if all_filled else ''}")

# Simulate PROPOSED (fixed) extraction
fixed_extraction = raw_board[0:20, 4:14]
fixed_extraction = (fixed_extraction > 0).astype(np.uint8)

print(f"\n3️⃣ PROPOSED EXTRACTION [0:20, 4:14] - FIXED")
print(f"   Shape: {fixed_extraction.shape}")
print(f"   Bottom 2 rows:")
for r in [-2, -1]:
    row = fixed_extraction[r, :]
    all_filled = np.all(row > 0)
    row_str = "".join("█" if c else "." for c in row)
    print(f"     Row {r}: {row_str} {'← WALL!' if all_filled else ''}")

# Test with gameplay - place 10 pieces
print(f"\n4️⃣ GAMEPLAY TEST: Placing 10 pieces")

obs_raw, _ = env_raw.reset(seed=999)
for i in range(10):
    obs_raw, _, term, trunc, _ = env_raw.step(6)  # Hard drop
    if term or trunc:
        break

board_after = obs_raw['board']

# Compare extractions after gameplay
current_after = board_after[2:22, 4:14]
current_after = (current_after > 0).astype(np.uint8)

fixed_after = board_after[0:20, 4:14]
fixed_after = (fixed_after > 0).astype(np.uint8)

print(f"\n   After {i+1} pieces placed:")
print(f"\n   CURRENT (buggy) - bottom 4 rows:")
for r in range(16, 20):
    row = current_after[r, :]
    row_str = "".join("█" if c else "." for c in row)
    filled = np.count_nonzero(row)
    print(f"     Row {r}: {row_str} ({filled}/10)")

print(f"\n   PROPOSED (fixed) - bottom 4 rows:")
for r in range(16, 20):
    row = fixed_after[r, :]
    row_str = "".join("█" if c else "." for c in row)
    filled = np.count_nonzero(row)
    print(f"     Row {r}: {row_str} ({filled}/10)")

# Count completely filled bottom rows
current_wall_rows = sum(1 for r in range(-4, 0) if np.all(current_after[r, :] > 0))
fixed_wall_rows = sum(1 for r in range(-4, 0) if np.all(fixed_after[r, :] > 0))

print(f"\n5️⃣ WALL ROW COUNT (completely filled bottom rows)")
print(f"   CURRENT extraction: {current_wall_rows} wall rows ❌")
print(f"   PROPOSED extraction: {fixed_wall_rows} wall rows {'✅' if fixed_wall_rows == 0 else '❌'}")

# Test the fix would work with CompleteVisionWrapper
print(f"\n6️⃣ WRAPPER COMPATIBILITY TEST")
print(f"   CURRENT extraction shape: {current_extraction.shape}")
print(f"   PROPOSED extraction shape: {fixed_extraction.shape}")
print(f"   Expected shape: (20, 10)")

if fixed_extraction.shape == (20, 10):
    print(f"   ✅ Shape matches observation_space")
else:
    print(f"   ❌ Shape mismatch!")

# Summary
print(f"\n" + "="*80)
print(f"VERIFICATION SUMMARY")
print(f"="*80)

issues_current = []
issues_fixed = []

if current_wall_rows > 0:
    issues_current.append(f"Contains {current_wall_rows} wall rows")

if fixed_wall_rows > 0:
    issues_fixed.append(f"Contains {fixed_wall_rows} wall rows")

if fixed_extraction.shape != (20, 10):
    issues_fixed.append(f"Wrong shape: {fixed_extraction.shape}")

print(f"\nCURRENT [2:22, 4:14]:")
if issues_current:
    for issue in issues_current:
        print(f"   ❌ {issue}")
else:
    print(f"   ✅ No issues")

print(f"\nPROPOSED [0:20, 4:14]:")
if issues_fixed:
    for issue in issues_fixed:
        print(f"   ❌ {issue}")
else:
    print(f"   ✅ No issues - FIX IS CORRECT!")

if not issues_fixed:
    print(f"\n🎯 RECOMMENDATION: Apply the fix!")
    print(f"   Change src/env_wrapper.py line 32:")
    print(f"   FROM: board = full_board[2:22, 4:14]")
    print(f"   TO:   board = full_board[0:20, 4:14]")
else:
    print(f"\n⚠️  WARNING: Fix has issues, further investigation needed")

env_raw.close()
print("="*80)
