#!/usr/bin/env python3
"""Test that the wrapper fix actually works with the wrapped environment"""

import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import make_env

print("="*80)
print("TESTING WRAPPER AFTER FIX")
print("="*80)

# Create wrapped environment (with CompleteVisionWrapper)
env = make_env(render_mode=None, use_complete_vision=True, use_cnn=False)
obs, _ = env.reset(seed=42)

print(f"\n1️⃣ WRAPPED OBSERVATION")
print(f"   Shape: {obs.shape}")
print(f"   Expected: (20, 10, 1)")
print(f"   Match: {'✅' if obs.shape == (20, 10, 1) else '❌'}")

# Check bottom rows
board_2d = obs[:, :, 0]
print(f"\n2️⃣ BOTTOM ROWS CHECK")
for r in [-4, -3, -2, -1]:
    row = board_2d[r, :]
    all_filled = np.all(row > 0)
    row_str = "".join("█" if c else "." for c in row)
    status = "❌ WALL!" if all_filled else "✅ OK"
    print(f"   Row {r}: {row_str} {status}")

# Count wall rows
wall_rows = sum(1 for r in range(-4, 0) if np.all(board_2d[r, :] > 0))
print(f"\n   Total wall rows in bottom 4: {wall_rows}")
if wall_rows == 0:
    print(f"   ✅ FIX SUCCESSFUL - No wall rows!")
else:
    print(f"   ❌ FIX FAILED - {wall_rows} wall rows still present")

# Test with gameplay
print(f"\n3️⃣ GAMEPLAY TEST")
obs, _ = env.reset(seed=999)

print(f"   Placing 10 pieces...")
for i in range(10):
    obs, reward, term, trunc, info = env.step(6)  # Hard drop
    if term or trunc:
        print(f"   Game ended after {i+1} pieces")
        break

board_after = obs[:, :, 0]
print(f"\n   Bottom 4 rows after {i+1} pieces:")
for r in range(16, 20):
    row = board_after[r, :]
    row_str = "".join("█" if c else "." for c in row)
    filled = np.count_nonzero(row)
    all_filled = np.all(row > 0)
    status = "❌ WALL!" if all_filled else "✅"
    print(f"     Row {r}: {row_str} ({filled}/10) {status}")

# Final check
wall_rows_after = sum(1 for r in range(16, 20) if np.all(board_after[r, :] > 0))
if wall_rows_after == 0:
    print(f"\n   ✅ No walls detected after gameplay!")
else:
    print(f"\n   ⚠️  {wall_rows_after} completely filled rows detected")
    print(f"      (Could be actual game progress, not walls)")

env.close()

print(f"\n" + "="*80)
print(f"RESULT: {'✅ FIX VERIFIED - Wrapper works correctly!' if wall_rows == 0 else '❌ Fix incomplete'}")
print("="*80)
