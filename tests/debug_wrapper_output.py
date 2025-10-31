#!/usr/bin/env python3
"""Debug what the wrapper is actually outputting"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env

print("="*80)
print("DEBUG: Wrapper Output Analysis")
print("="*80)

# Test 1: Raw dict observation
print("\n1️⃣ RAW ENVIRONMENT (no wrapper)")
env_raw = make_env(render_mode=None, use_complete_vision=False, use_cnn=False)
obs_raw, _ = env_raw.reset(seed=42)

board_raw = obs_raw['board']
extracted_manual = board_raw[0:20, 4:14]
extracted_manual_binary = (extracted_manual > 0).astype(np.uint8)

print(f"   Manual extraction [0:20, 4:14] bottom 2 rows:")
for r in [18, 19]:
    row = extracted_manual_binary[r, :]
    row_str = "".join("█" if c else "." for c in row)
    all_filled = np.all(row > 0)
    print(f"     Row {r}: {row_str} {'← ALL FILLED!' if all_filled else '✅'}")

env_raw.close()

# Test 2: Wrapped observation
print("\n2️⃣ WRAPPED ENVIRONMENT (with CompleteVisionWrapper)")
env_wrapped = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
obs_wrapped, _ = env_wrapped.reset(seed=42)

board_wrapped = obs_wrapped[:, :, 0]

print(f"   Wrapped observation bottom 2 rows:")
for r in [18, 19]:
    row = board_wrapped[r, :]
    row_str = "".join("█" if c else "." for c in row)
    all_filled = np.all(row > 0)
    print(f"     Row {r}: {row_str} {'← ALL FILLED!' if all_filled else '✅'}")

# Test 3: Check if they match
print("\n3️⃣ COMPARISON")
if np.array_equal(extracted_manual_binary, board_wrapped):
    print("   ✅ Wrapper output MATCHES manual extraction")
else:
    print("   ❌ Wrapper output DIFFERS from manual extraction")

    # Find differences
    diff = (extracted_manual_binary != board_wrapped)
    diff_count = np.count_nonzero(diff)
    print(f"   Differences: {diff_count} cells")

    if diff_count > 0:
        print("\n   Rows with differences:")
        for r in range(20):
            if np.any(diff[r, :]):
                manual_row = "".join("█" if c else "." for c in extracted_manual_binary[r, :])
                wrapped_row = "".join("█" if c else "." for c in board_wrapped[r, :])
                print(f"     Row {r}:")
                print(f"       Manual:  {manual_row}")
                print(f"       Wrapped: {wrapped_row}")

env_wrapped.close()

print("\n" + "="*80)
