#!/usr/bin/env python3
"""Verify that coordinate mapping is correct between raw and playable"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("COORDINATE MAPPING VERIFICATION")
print("="*80)

env = Tetris(render_mode=None)

# Test at different X positions
test_positions = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

print("\nTesting piece placement at different self.x positions:")
print("Raw self.x → Expected playable column → Actual landed column")
print("="*60)

for target_x in test_positions:
    env_test = Tetris(render_mode=None)
    obs, _ = env_test.reset(seed=42)

    # Move piece to target_x
    while env_test.x > target_x:
        obs, _, term, trunc, _ = env_test.step(0)  # LEFT
        if term or trunc or env_test.x < target_x:
            break

    while env_test.x < target_x:
        obs, _, term, trunc, _ = env_test.step(1)  # RIGHT
        if term or trunc or env_test.x > target_x:
            break

    actual_x = env_test.x
    expected_playable_col = actual_x - 4  # Since padding = 4

    # Drop the piece using gravity (NOOP)
    for i in range(25):
        obs, _, term, trunc, _ = env_test.step(7)  # NOOP
        if term or trunc:
            break

    # Check where it landed
    board = obs['board'][0:20, 4:14]  # Extract playable area

    # Find the landed piece
    landed_cols = set()
    for r in range(20):
        for c in range(10):
            if board[r, c] > 0:
                landed_cols.add(c)

    if landed_cols:
        min_col = min(landed_cols)
        max_col = max(landed_cols)
        print(f"  {actual_x:2d}  →  {expected_playable_col}  →  {min_col}-{max_col}")

    env_test.close()

env.close()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If 'Expected' matches 'Actual', the coordinate system is correct.")
print("If they don't match, there's an offset bug.")
print("="*80)
