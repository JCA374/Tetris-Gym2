#!/usr/bin/env python3
"""
Diagnostic Test: Line Clear Bug
================================

Tests whether the training script can properly detect line clears.

BUG: Training script looks for info['number_of_lines']
     Environment returns info['lines_cleared']
"""

import sys
import gymnasium as gym
import tetris_gymnasium.envs
import numpy as np

print("=" * 80)
print("DIAGNOSTIC TEST: Line Clear Detection Bug")
print("=" * 80)

# Test 1: What does the environment actually return?
print("\n[TEST 1] Environment Info Dict Keys")
print("-" * 80)

env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)
obs, info = env.reset()

print(f"✓ Environment created")
print(f"✓ Info dict keys: {sorted(info.keys())}")
print(f"✓ Lines cleared field: info['lines_cleared'] = {info['lines_cleared']}")

if 'number_of_lines' in info:
    print(f"✓ ALSO found: info['number_of_lines'] = {info['number_of_lines']}")
else:
    print(f"✗ NOT FOUND: info['number_of_lines'] (training script looks for this!)")

# Test 2: Can we actually clear lines?
print("\n[TEST 2] Attempting to Clear Lines")
print("-" * 80)

env.reset()
lines_cleared_total = 0
max_attempts = 1000

print("Taking random actions to try to clear lines...")
for i in range(max_attempts):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if info['lines_cleared'] > 0:
        lines_cleared_total += info['lines_cleared']
        print(f"✓ Lines cleared at step {i}: {info['lines_cleared']}")
        print(f"  Total lines cleared: {lines_cleared_total}")
        break

    if terminated or truncated:
        env.reset()

if lines_cleared_total > 0:
    print(f"\n✓ SUCCESS: Cleared {lines_cleared_total} lines")
else:
    print(f"\n✗ FAILURE: No lines cleared in {max_attempts} steps")

# Test 3: What does the training script expect?
print("\n[TEST 3] Training Script Field Name Bug")
print("-" * 80)

# Simulate what training script does
lines_from_correct_field = info.get('lines_cleared', 0)
lines_from_wrong_field = info.get('number_of_lines', 0)

print(f"Environment provides: info['lines_cleared'] = {lines_from_correct_field}")
print(f"Training script uses:  info.get('number_of_lines', 0) = {lines_from_wrong_field}")

if lines_from_wrong_field == 0 and lines_from_correct_field >= 0:
    print(f"\n✗ BUG CONFIRMED: Training script will ALWAYS see 0 lines!")
    print(f"   Even though lines_cleared = {lines_from_correct_field}")
else:
    print(f"\n✓ No bug detected")

# Test 4: Force a line clear by filling a row manually
print("\n[TEST 4] Force Line Clear Test")
print("-" * 80)

env.reset()
print("Using hard drops to stack pieces...")

for _ in range(100):
    action = 5  # HARD_DROP
    obs, reward, terminated, truncated, info = env.step(action)

    if info['lines_cleared'] > 0:
        print(f"✓ HARD DROP cleared {info['lines_cleared']} line(s)!")
        print(f"  Reward: {reward}")
        print(f"  Info: {info}")
        break

    if terminated or truncated:
        print(f"✗ Game ended before clearing lines")
        break

env.close()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Environment field name: 'lines_cleared'")
print(f"Training script expects: 'number_of_lines'")
print(f"\n🐛 BUG: Field name mismatch!")
print(f"📝 FIX: Change train_feature_vector.py line 109, 254:")
print(f"   FROM: info.get('number_of_lines', 0)")
print(f"   TO:   info.get('lines_cleared', 0)")
print("=" * 80)
