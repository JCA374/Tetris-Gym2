#!/usr/bin/env python3
"""Debug the actual extraction happening in the wrapper"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import raw to see what's happening
from config import make_env

# Create raw environment (without wrapper)
env = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
obs, _ = env.reset(seed=42)

full_board = obs['board']

print("="*80)
print("DEBUG: What is actually in rows 0-19?")
print("="*80)

print(f"\nFull board shape: {full_board.shape}")
print(f"\nRows 0-19, cols 4-13 (our extraction):")
print()

extracted = full_board[0:20, 4:14]

for r in range(20):
    row = extracted[r, :]
    row_str = "".join("█" if c > 0 else "." for c in row)
    filled = np.count_nonzero(row)
    all_filled = np.all(row > 0)

    # Get the corresponding full row to see context
    full_row = full_board[r, :]
    full_filled = np.count_nonzero(full_row)

    status = ""
    if all_filled:
        status = "← ALL FILLED!"
    elif filled > 0:
        status = f"← {filled}/10 filled"

    print(f"  Row {r:2d}: {row_str} {status} (full row has {full_filled}/18)")

# Also check after binarization
print(f"\n" + "="*80)
print("After binarization (>0):")
print("="*80)

binary = (extracted > 0).astype(np.uint8)
for r in [18, 19]:
    row = binary[r, :]
    row_str = "".join("█" if c else "." for c in row)
    all_filled = np.all(row > 0)
    print(f"  Row {r}: {row_str} {'← ALL FILLED' if all_filled else ''}")

env.close()
