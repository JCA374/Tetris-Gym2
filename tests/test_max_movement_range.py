#!/usr/bin/env python3
"""Test maximum movement range - can we reach columns 0 and 9?"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("MAXIMUM MOVEMENT RANGE TEST")
print("="*80)

env = Tetris(render_mode=None)

# Test 1: How far LEFT can we go?
print("\nTest 1: Maximum LEFT movement")
print("=" * 40)

obs, info = env.reset(seed=100)
mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_before = np.where(mask.sum(axis=0) > 0)[0]
print(f"Initial piece columns: {cols_before.min()}-{cols_before.max()}")

# Move LEFT 10 times
for i in range(10):
    obs, _, term, trunc, _ = env.step(0)  # LEFT
    if term or trunc:
        break

mask_after = obs['active_tetromino_mask'][0:20, 4:14]
cols_after = np.where(mask_after.sum(axis=0) > 0)[0]
if len(cols_after) > 0:
    print(f"After 10 LEFT moves: columns {cols_after.min()}-{cols_after.max()}")
    print(f"→ Leftmost column reached: {cols_after.min()}")

env.close()

# Test 2: How far RIGHT can we go?
print("\nTest 2: Maximum RIGHT movement")
print("=" * 40)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=100)
mask = obs['active_tetromino_mask'][0:20, 4:14]
cols_before = np.where(mask.sum(axis=0) > 0)[0]
print(f"Initial piece columns: {cols_before.min()}-{cols_before.max()}")

# Move RIGHT 10 times
for i in range(10):
    obs, _, term, trunc, _ = env.step(1)  # RIGHT
    if term or trunc:
        break

mask_after = obs['active_tetromino_mask'][0:20, 4:14]
cols_after = np.where(mask_after.sum(axis=0) > 0)[0]
if len(cols_after) > 0:
    print(f"After 10 RIGHT moves: columns {cols_after.min()}-{cols_after.max()}")
    print(f"→ Rightmost column reached: {cols_after.max()}")

env.close()

# Test 3: Try to fill outer columns deliberately
print("\nTest 3: Trying to fill columns 0 and 9")
print("=" * 40)

env = Tetris(render_mode=None)
obs, info = env.reset(seed=200)

column_usage = [0] * 10

for piece_num in range(100):
    # Alternate: push FAR LEFT, drop, push FAR RIGHT, drop
    if piece_num % 2 == 0:
        # Try to reach column 0
        for _ in range(15):
            obs, _, term, trunc, _ = env.step(0)  # LEFT
            if term or trunc:
                break
    else:
        # Try to reach column 9
        for _ in range(15):
            obs, _, term, trunc, _ = env.step(1)  # RIGHT
            if term or trunc:
                break

    if not (term or trunc):
        # Hard drop
        obs, _, term, trunc, _ = env.step(5)

    # Count column usage
    board = obs['board'][0:20, 4:14]
    for c in range(10):
        if np.any(board[:, c] > 0):
            column_usage[c] += 1

    if term or trunc:
        break

env.close()

print(f"\nColumn usage frequency (out of {piece_num+1} pieces):")
for c in range(10):
    bar = "█" * (column_usage[c] // 2)
    print(f"  Column {c}: {column_usage[c]:3d} {bar}")

print(f"\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

if column_usage[0] > 0 and column_usage[9] > 0:
    print("✅ ALL columns (0-9) are reachable!")
else:
    unused = [c for c in range(10) if column_usage[c] == 0]
    used = [c for c in range(10) if column_usage[c] > 0]
    print(f"❌ Limited column access!")
    print(f"   Accessible columns: {min(used)}-{max(used)}")
    print(f"   Unreachable columns: {unused}")
    print(f"\n   Maximum possible row fullness: {len(used)}/10 cells")
    print(f"   Line clearing is IMPOSSIBLE!")

print("="*80)
