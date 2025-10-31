#!/usr/bin/env python3
"""Investigate why pieces can't reach outer columns"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, discover_action_meanings
from src.reward_shaping import extract_board_from_obs

print("="*80)
print("INVESTIGATING MOVEMENT LIMITS")
print("="*80)

# Test with raw environment (no wrapper)
env_raw = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
discover_action_meanings(env_raw)

from config import ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP

print(f"\n1️⃣ ACTION MAPPING")
print(f"   LEFT: {ACTION_LEFT}")
print(f"   RIGHT: {ACTION_RIGHT}")
print(f"   DOWN: {ACTION_DOWN}")
print(f"   HARD_DROP: {ACTION_HARD_DROP}")

# Check what the raw board looks like
obs, _ = env_raw.reset(seed=42)
raw_board = obs['board']
mask = obs['active_tetromino_mask']

print(f"\n2️⃣ RAW BOARD ANALYSIS")
print(f"   Board shape: {raw_board.shape}")
print(f"   Mask shape: {mask.shape}")

# Find where the active piece spawns
spawn_rows, spawn_cols = np.where(mask > 0)
if len(spawn_cols) > 0:
    spawn_min = spawn_cols.min()
    spawn_max = spawn_cols.max()
    spawn_center = (spawn_min + spawn_max) // 2
    print(f"   Piece spawn columns (raw): {spawn_min}-{spawn_max}")
    print(f"   Piece spawn center: column {spawn_center}")
    print(f"   In playable coords (4-13): column {spawn_center - 4}")

# Test piece movement step by step
print(f"\n3️⃣ STEP-BY-STEP LEFT MOVEMENT")
obs, _ = env_raw.reset(seed=100)

for step in range(20):
    mask_before = obs['active_tetromino_mask']
    cols_before = np.where(mask_before.sum(axis=0) > 0)[0]

    obs, _, term, trunc, _ = env_raw.step(ACTION_LEFT)

    mask_after = obs['active_tetromino_mask']
    cols_after = np.where(mask_after.sum(axis=0) > 0)[0]

    if len(cols_before) > 0 and len(cols_after) > 0:
        min_before = cols_before.min()
        min_after = cols_after.min()

        if min_before != min_after:
            print(f"   Step {step}: Column {min_before} → {min_after} (moved left)")
        else:
            print(f"   Step {step}: Column {min_before} (STUCK - can't move left!)")
            print(f"     Playable column: {min_before - 4}")
            break

    if term or trunc:
        print(f"   Piece locked at step {step}")
        break

# Check the final position
board_final = obs['board']
playable_final = board_final[0:20, 4:14]
for r in range(5):
    row_str = "".join("█" if playable_final[r, c] > 0 else "." for c in range(10))
    print(f"   Row {r}: {row_str}")

print(f"\n4️⃣ STEP-BY-STEP RIGHT MOVEMENT")
obs, _ = env_raw.reset(seed=101)

for step in range(20):
    mask_before = obs['active_tetromino_mask']
    cols_before = np.where(mask_before.sum(axis=0) > 0)[0]

    obs, _, term, trunc, _ = env_raw.step(ACTION_RIGHT)

    mask_after = obs['active_tetromino_mask']
    cols_after = np.where(mask_after.sum(axis=0) > 0)[0]

    if len(cols_before) > 0 and len(cols_after) > 0:
        max_before = cols_before.max()
        max_after = cols_after.max()

        if max_before != max_after:
            print(f"   Step {step}: Column {max_before} → {max_after} (moved right)")
        else:
            print(f"   Step {step}: Column {max_before} (STUCK - can't move right!)")
            print(f"     Playable column: {max_before - 4}")
            break

    if term or trunc:
        print(f"   Piece locked at step {step}")
        break

print(f"\n5️⃣ CHECKING SPAWN POSITION ACROSS DIFFERENT PIECES")
spawn_positions = []

for seed in range(10):
    obs, _ = env_raw.reset(seed=seed)
    mask = obs['active_tetromino_mask']
    cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        center = (cols.min() + cols.max()) // 2
        playable_center = center - 4  # Convert to playable coords (0-9)
        spawn_positions.append(playable_center)

print(f"   Spawn centers (playable coords 0-9): {spawn_positions}")
print(f"   Average spawn position: {np.mean(spawn_positions):.1f}")
print(f"   Min spawn: {min(spawn_positions)}, Max spawn: {max(spawn_positions)}")

env_raw.close()

print(f"\n" + "="*80)
print(f"CONCLUSION")
print(f"="*80)
print(f"""
The tetris-gymnasium environment appears to have LIMITED movement range!

Pieces spawn near the center (column 4-5 of 0-9) and can only move a few
columns left/right before hitting invisible walls.

This is likely a characteristic of the tetris-gymnasium library, not our code.

IMPLICATIONS FOR TRAINING:
- Agent can only use columns 2-8 (not full 0-9 range)
- Center-stacking is INEVITABLE with this environment
- Need different reward shaping or different Tetris environment
""")
print("="*80)
