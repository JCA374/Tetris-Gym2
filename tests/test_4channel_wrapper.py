#!/usr/bin/env python3
"""Test the 4-channel CompleteVisionWrapper"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env

print("="*80)
print("TESTING 4-CHANNEL WRAPPER")
print("="*80)

# Create 4-channel wrapped environment
env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
obs, _ = env.reset(seed=42)

print(f"\n1️⃣ OBSERVATION SHAPE CHECK")
print(f"   Shape: {obs.shape}")
print(f"   Expected: (20, 10, 4)")
print(f"   Match: {'✅' if obs.shape == (20, 10, 4) else '❌'}")

print(f"\n2️⃣ DATA TYPE CHECK")
print(f"   dtype: {obs.dtype}")
print(f"   Expected: uint8")
print(f"   Match: {'✅' if obs.dtype == np.uint8 else '❌'}")

print(f"\n3️⃣ VALUE RANGE CHECK")
print(f"   Min: {obs.min()}")
print(f"   Max: {obs.max()}")
print(f"   Binary (0/1): {'✅' if obs.max() <= 1 else '⚠️ Values > 1'}")

print(f"\n4️⃣ CHANNEL CONTENT ANALYSIS")

# Analyze each channel
channel_names = ["Board", "Active Piece", "Holder", "Queue"]
for ch_idx, name in enumerate(channel_names):
    channel = obs[:, :, ch_idx]
    non_zero = np.count_nonzero(channel)
    filled_rows = [r for r in range(20) if np.any(channel[r, :] > 0)]
    filled_cols = [c for c in range(10) if np.any(channel[:, c] > 0)]

    print(f"\n   Channel {ch_idx} ({name}):")
    print(f"     Non-zero cells: {non_zero}")
    print(f"     Filled rows: {filled_rows[:5]}{'...' if len(filled_rows) > 5 else ''}")
    print(f"     Filled cols: {filled_cols[:5]}{'...' if len(filled_cols) > 5 else ''}")

    # Show a mini visualization for the first 6 rows
    if non_zero > 0:
        print(f"     Top 6 rows:")
        for r in range(min(6, 20)):
            row_str = "".join("█" if channel[r, c] > 0 else "." for c in range(10))
            if np.any(channel[r, :] > 0):
                print(f"       Row {r}: {row_str}")

print(f"\n5️⃣ GAMEPLAY TEST (10 steps)")
print(f"   Testing that observations update correctly...")

for step in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    if term or trunc:
        print(f"   Game ended at step {step+1}")
        break

    # Check that observations are still valid
    if obs.shape != (20, 10, 4):
        print(f"   ❌ Shape changed to {obs.shape} at step {step+1}")
        break

    if obs.max() > 1 or obs.min() < 0:
        print(f"   ❌ Values out of range at step {step+1}: [{obs.min()}, {obs.max()}]")
        break

print(f"\n   After {step+1 if (term or trunc) else 10} steps:")
print(f"     Shape still (20, 10, 4): {'✅' if obs.shape == (20, 10, 4) else '❌'}")
print(f"     Values still binary: {'✅' if obs.max() <= 1 and obs.min() >= 0 else '❌'}")

# Check final state
final_board_cells = np.count_nonzero(obs[:, :, 0])
final_active_cells = np.count_nonzero(obs[:, :, 1])

print(f"     Board channel cells: {final_board_cells}")
print(f"     Active piece cells: {final_active_cells}")

print(f"\n6️⃣ WALL VERIFICATION")
print(f"   Checking that bottom rows are NOT walls...")

bottom_4_rows = obs[-4:, :, 0]  # Board channel, bottom 4 rows
wall_rows = sum(1 for r in range(4) if np.all(bottom_4_rows[r, :] > 0))

print(f"   Completely filled bottom rows: {wall_rows}")
if wall_rows == 0:
    print(f"   ✅ No wall rows - extraction is correct!")
else:
    print(f"   ❌ {wall_rows} wall rows found - extraction may be wrong!")

print(f"\n7️⃣ CHANNEL INDEPENDENCE CHECK")
print(f"   Verifying channels contain different information...")

# Check that channels are not all identical
channels_identical = True
channel_0 = obs[:, :, 0]

for ch_idx in range(1, 4):
    if not np.array_equal(channel_0, obs[:, :, ch_idx]):
        channels_identical = False
        break

if not channels_identical:
    print(f"   ✅ Channels contain different information")
else:
    print(f"   ⚠️ All channels appear identical (may be OK if game just started)")

# Calculate correlation between board and active piece
board_flat = obs[:, :, 0].flatten()
active_flat = obs[:, :, 1].flatten()

# Check overlap
overlap = np.sum((board_flat > 0) & (active_flat > 0))
print(f"\n   Board/Active overlap: {overlap} cells")
print(f"   {'✅ Minimal overlap (good)' if overlap < 5 else '⚠️ High overlap'}")

env.close()

print(f"\n" + "="*80)
print(f"SUMMARY")
print(f"="*80)

checks = [
    ("Observation shape (20, 10, 4)", obs.shape == (20, 10, 4)),
    ("Data type uint8", obs.dtype == np.uint8),
    ("Binary values (0/1)", obs.max() <= 1 and obs.min() >= 0),
    ("No wall rows in bottom 4", wall_rows == 0),
    ("Channels have different content", not channels_identical),
]

all_pass = all(check[1] for check in checks)

for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")

print(f"\n{'='*80}")
if all_pass:
    print(f"🎉 ALL CHECKS PASSED - 4-CHANNEL WRAPPER READY!")
else:
    print(f"⚠️ SOME CHECKS FAILED - Review above")
print(f"{'='*80}")
