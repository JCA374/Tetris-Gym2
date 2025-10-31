#!/usr/bin/env python3
"""Test SINGLE action effect - one action at a time"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("SINGLE ACTION EFFECT TEST")
print("="*80)

def test_action(action_id, action_name):
    env = Tetris(render_mode=None)
    obs, info = env.reset(seed=100)  # Fixed seed for consistency

    # Get initial piece position
    mask_before = obs['active_tetromino_mask']
    rows_before, cols_before = np.where(mask_before > 0)

    if len(cols_before) == 0:
        env.close()
        return

    col_min_before = cols_before.min()
    col_max_before = cols_before.max()
    row_min_before = rows_before.min()

    print(f"\n{action_name} (action {action_id}):")
    print(f"  Before: columns {col_min_before}-{col_max_before} (raw coords)")

    # Take SINGLE action
    obs, reward, term, trunc, info = env.step(action_id)

    if not (term or trunc):
        mask_after = obs['active_tetromino_mask']
        rows_after, cols_after = np.where(mask_after > 0)

        if len(cols_after) > 0:
            col_min_after = cols_after.min()
            col_max_after = cols_after.max()
            row_min_after = rows_after.min()

            print(f"  After:  columns {col_min_after}-{col_max_after} (raw coords)")

            col_delta = col_min_after - col_min_before
            row_delta = row_min_after - row_min_before

            if col_delta < 0:
                print(f"  → Moved LEFT by {abs(col_delta)}")
            elif col_delta > 0:
                print(f"  → Moved RIGHT by {col_delta}")
            else:
                print(f"  → No horizontal movement")

            if row_delta > 0:
                print(f"  → Fell DOWN by {row_delta}")
            elif row_delta < 0:
                print(f"  → Moved UP by {abs(row_delta)} (rotation?)")
            else:
                print(f"  → No vertical movement")

            # Check rotation
            if not np.array_equal(mask_before, mask_after):
                print(f"  → Shape CHANGED (rotated or different piece)")

    env.close()

# Test each action individually
test_action(0, "NOOP")
test_action(1, "LEFT")
test_action(2, "RIGHT")
test_action(3, "DOWN")
test_action(4, "ROTATE_CW")
test_action(5, "ROTATE_CCW")
test_action(6, "HARD_DROP")
test_action(7, "SWAP")

print(f"\n" + "="*80)
