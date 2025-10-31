#!/usr/bin/env python3
"""Check where pieces are actually being placed"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env, ACTION_LEFT, ACTION_DOWN, ACTION_HARD_DROP

print("="*80)
print("CHECKING PIECE PLACEMENT LOCATION")
print("="*80)

# Test with raw (unwrapped) environment
print("\n1️⃣ RAW ENVIRONMENT (no wrapper)")
env_raw = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
obs, _ = env_raw.reset(seed=777)

print("   Before any actions:")
board_before = obs['board']
filled_before = np.count_nonzero(board_before)
print(f"     Total filled cells: {filled_before}")

# Move left and hard drop
for _ in range(25):
    obs, _, _, _, _ = env_raw.step(ACTION_LEFT)
    obs, _, _, _, _ = env_raw.step(ACTION_DOWN)

obs, _, _, _, _ = env_raw.step(ACTION_HARD_DROP)

print("\n   After LEFT + HARD_DROP:")
board_after = obs['board']
filled_after = np.count_nonzero(board_after)
print(f"     Total filled cells: {filled_after}")
print(f"     New cells: {filled_after - filled_before}")

# Check which rows have the new pieces
print("\n   Checking rows 15-23 (bottom area):")
for r in range(15, 24):
    row = board_after[r, 4:14]  # Playable columns
    filled = np.count_nonzero(row)
    if filled > 0:
        row_str = "".join("█" if c > 0 else "." for c in row)
        print(f"     Raw row {r}: {row_str} ({filled}/10 filled)")

env_raw.close()

# Test with wrapped environment
print("\n2️⃣ WRAPPED ENVIRONMENT")
env_wrapped = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
obs, _ = env_wrapped.reset(seed=777)

print("   Before any actions:")
board_before = obs[:, :, 0]
filled_before = np.count_nonzero(board_before)
print(f"     Total filled cells: {filled_before}")

# Move left and hard drop
for _ in range(25):
    obs, _, _, _, _ = env_wrapped.step(ACTION_LEFT)
    obs, _, _, _, _ = env_wrapped.step(ACTION_DOWN)

obs, _, _, _, _ = env_wrapped.step(ACTION_HARD_DROP)

print("\n   After LEFT + HARD_DROP:")
board_after = obs[:, :, 0]
filled_after = np.count_nonzero(board_after)
print(f"     Total filled cells: {filled_after}")
print(f"     New cells: {filled_after - filled_before}")

# Check bottom rows
print("\n   Checking rows 15-19 (bottom of wrapped observation):")
for r in range(15, 20):
    row = board_after[r, :]
    filled = np.count_nonzero(row)
    if filled > 0:
        row_str = "".join("█" if c > 0 else "." for c in row)
        print(f"     Wrapped row {r}: {row_str} ({filled}/10 filled)")

# Check column 0 specifically
col_0 = board_after[:, 0]
col_0_filled = np.count_nonzero(col_0)
print(f"\n   Column 0 filled cells: {col_0_filled}")
if col_0_filled > 0:
    print(f"     Rows where column 0 is filled: {np.where(col_0 > 0)[0].tolist()}")

env_wrapped.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If pieces are landing in raw rows 20-23, they fall OUTSIDE our wrapped
observation window (which only includes rows 0-19).

This would explain why outer column tests fail after the fix!
""")
