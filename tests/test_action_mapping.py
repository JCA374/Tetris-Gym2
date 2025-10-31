#!/usr/bin/env python3
"""Test action mapping - verify LEFT/RIGHT actually move pieces"""

from tetris_gymnasium.envs import Tetris
import numpy as np

print("="*80)
print("ACTION MAPPING TEST")
print("="*80)

env = Tetris(render_mode=None)

# Test each action to see what it does
actions_to_test = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
    3: "DOWN",
    4: "ROTATE_CW",
    5: "ROTATE_CCW",
    6: "HARD_DROP",
    7: "SWAP"
}

for action_id, action_name in actions_to_test.items():
    print(f"\n{'='*40}")
    print(f"Testing Action {action_id}: {action_name}")
    print(f"{'='*40}")

    obs, info = env.reset(seed=42)

    # Get initial piece position
    mask_before = obs['active_tetromino_mask']
    rows_before, cols_before = np.where(mask_before > 0)

    if len(cols_before) > 0:
        col_min_before = cols_before.min()
        col_max_before = cols_before.max()
        row_min_before = rows_before.min()
        row_max_before = rows_before.max()

        print(f"Before: Columns {col_min_before}-{col_max_before}, Rows {row_min_before}-{row_max_before}")

        # Take action 3 times
        for i in range(3):
            obs, reward, term, trunc, info = env.step(action_id)
            if term or trunc:
                print(f"  Step {i+1}: Piece locked/game over")
                break

        if not (term or trunc):
            # Get new position
            mask_after = obs['active_tetromino_mask']
            rows_after, cols_after = np.where(mask_after > 0)

            if len(cols_after) > 0:
                col_min_after = cols_after.min()
                col_max_after = cols_after.max()
                row_min_after = rows_after.min()
                row_max_after = rows_after.max()

                print(f"After:  Columns {col_min_after}-{col_max_after}, Rows {row_min_after}-{row_max_after}")

                # Analyze movement
                col_delta = col_min_after - col_min_before
                row_delta = row_min_after - row_min_before

                if col_delta < 0:
                    print(f"→ Moved LEFT by {abs(col_delta)} columns")
                elif col_delta > 0:
                    print(f"→ Moved RIGHT by {col_delta} columns")

                if row_delta > 0:
                    print(f"→ Moved DOWN by {row_delta} rows")
                elif row_delta < 0:
                    print(f"→ Moved UP by {abs(row_delta)} rows")

                if col_delta == 0 and row_delta == 0:
                    # Check for rotation
                    if not np.array_equal(mask_before, mask_after):
                        print(f"→ ROTATED (shape changed)")
                    else:
                        print(f"→ NO MOVEMENT")

env.close()

print(f"\n" + "="*80)
print("ACTION MAPPING VERIFIED")
print("="*80)
