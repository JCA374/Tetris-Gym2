#!/usr/bin/env python3
"""Verify board is actually 10 columns wide and pieces can reach all columns"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP
from src.reward_shaping import extract_board_from_obs, get_column_heights

print("="*80)
print("VERIFYING BOARD WIDTH AND COLUMN ACCESSIBILITY")
print("="*80)

env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)

# Test 1: Check wrapped observation width
obs, _ = env.reset(seed=42)
print(f"\n1️⃣ WRAPPED OBSERVATION")
print(f"   Shape: {obs.shape}")
print(f"   Width (columns): {obs.shape[1]}")
print(f"   Expected: 10 columns")

# Test 2: Extract and verify board
board = extract_board_from_obs(obs)
print(f"\n2️⃣ EXTRACTED BOARD")
print(f"   Shape: {board.shape}")
print(f"   Width (columns): {board.shape[1]}")

# Test 3: Try to reach column 0 (leftmost)
print(f"\n3️⃣ TESTING LEFT COLUMN ACCESS (Column 0)")
obs, _ = env.reset(seed=100)

print(f"   Moving LEFT 50 times, then HARD DROP...")
for i in range(50):
    obs, _, term, trunc, _ = env.step(ACTION_LEFT)
    if term or trunc:
        print(f"   Game ended during movement at step {i}")
        break

if not (term or trunc):
    obs, _, term, trunc, _ = env.step(ACTION_HARD_DROP)

board_left = extract_board_from_obs(obs)
heights_left = get_column_heights(board_left)

print(f"   Column heights: {heights_left}")
print(f"   Column 0 height: {heights_left[0]}")
print(f"   Column 0 reached: {'✅ YES' if heights_left[0] > 0 else '❌ NO'}")

# Show board
print(f"\n   Board state (first 8 rows):")
for r in range(8):
    row_str = "".join("█" if board_left[r, c] > 0 else "." for c in range(10))
    print(f"     {row_str}")

# Test 4: Try to reach column 9 (rightmost)
print(f"\n4️⃣ TESTING RIGHT COLUMN ACCESS (Column 9)")
obs, _ = env.reset(seed=101)

print(f"   Moving RIGHT 50 times, then HARD DROP...")
for i in range(50):
    obs, _, term, trunc, _ = env.step(ACTION_RIGHT)
    if term or trunc:
        print(f"   Game ended during movement at step {i}")
        break

if not (term or trunc):
    obs, _, term, trunc, _ = env.step(ACTION_HARD_DROP)

board_right = extract_board_from_obs(obs)
heights_right = get_column_heights(board_right)

print(f"   Column heights: {heights_right}")
print(f"   Column 9 height: {heights_right[9]}")
print(f"   Column 9 reached: {'✅ YES' if heights_right[9] > 0 else '❌ NO'}")

# Show board
print(f"\n   Board state (first 8 rows):")
for r in range(8):
    row_str = "".join("█" if board_right[r, c] > 0 else "." for c in range(10))
    print(f"     {row_str}")

# Test 5: Check if middle columns are easier to reach
print(f"\n5️⃣ TESTING MIDDLE COLUMN (Column 5)")
obs, _ = env.reset(seed=102)

print(f"   Placing piece with minimal movement...")
for i in range(5):
    obs, _, term, trunc, _ = env.step(ACTION_DOWN)
    if term or trunc:
        break

if not (term or trunc):
    obs, _, _, _, _ = env.step(ACTION_HARD_DROP)

board_middle = extract_board_from_obs(obs)
heights_middle = get_column_heights(board_middle)

print(f"   Column heights: {heights_middle}")
print(f"   Non-zero columns: {[i for i, h in enumerate(heights_middle) if h > 0]}")

# Test 6: Check raw observation to see if wrapper is cutting off columns
print(f"\n6️⃣ RAW OBSERVATION CHECK")
env_raw = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
obs_raw, _ = env_raw.reset(seed=42)

raw_board = obs_raw['board']
print(f"   Raw board shape: {raw_board.shape}")
print(f"   Expected: (24, 18)")
print(f"   Playable columns in raw: 4-13 (10 columns)")

# Extract manually
manual_extract = raw_board[0:20, 4:14]
print(f"   Manual extraction shape: {manual_extract.shape}")
print(f"   Manual extraction width: {manual_extract.shape[1]} columns")

env_raw.close()

print(f"\n" + "="*80)
print(f"ANALYSIS")
print(f"="*80)

issues = []
if obs.shape[1] != 10:
    issues.append(f"❌ Wrapped observation width is {obs.shape[1]}, not 10!")
if board.shape[1] != 10:
    issues.append(f"❌ Extracted board width is {board.shape[1]}, not 10!")
if heights_left[0] == 0:
    issues.append(f"⚠️  Cannot reach column 0 (leftmost)")
if heights_right[9] == 0:
    issues.append(f"⚠️  Cannot reach column 9 (rightmost)")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ All checks passed - board is correct 10 columns wide")

# Final diagnostic
print(f"\n📊 SUMMARY:")
print(f"   Observation width: {obs.shape[1]} columns")
print(f"   Board width: {board.shape[1]} columns")
print(f"   Column 0 accessible: {'✅' if heights_left[0] > 0 else '❌'}")
print(f"   Column 9 accessible: {'✅' if heights_right[9] > 0 else '❌'}")

env.close()
print("="*80)
