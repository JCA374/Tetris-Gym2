"""Trace where pieces actually land"""
import gymnasium as gym
import tetris_gymnasium.envs
import numpy as np

env = gym.make('tetris_gymnasium/Tetris', height=20, width=10, render_mode='ansi')
obs, info = env.reset()

print("Tracing piece placement for 20 steps")
print("Strategy: Always HARD_DROP to see where pieces land")
print("=" * 70)

for step in range(20):
    board_before = obs['board'].copy()

    # Just hard drop
    obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP

    board_after = obs['board']

    # Find where the piece landed (new non-zero, non-wall cells)
    diff = (board_after != board_before) & (board_after > 1)  # Ignore walls (1) and empty (0)

    if diff.any():
        rows, cols = np.where(diff)
        print(f"\nStep {step+1}: Piece landed")
        print(f"  Rows: {rows} (min={rows.min()}, max={rows.max()})")
        print(f"  Cols: {cols} (min={cols.min()}, max={cols.max()})")

        # Check bottom rows
        for row_idx in range(15, 24):
            row = board_after[row_idx, 4:14]  # Playable area
            filled = (row > 1).sum()  # Count pieces (not walls or empty)
            if filled > 0:
                print(f"  Row {row_idx}: {filled}/10 filled - {row}")

    print(f"  lines_cleared: {info.get('lines_cleared', 0)}")
    print(f"  Reward: {reward}")

    if info.get('lines_cleared', 0) > 0:
        print(f"  🎉 LINE CLEARED!")

    if terminated or truncated:
        print(f"\n  Game Over at step {step+1}")
        break

print("\n" + "=" * 70)
print("Final board state (bottom 10 rows of playable area):")
for row_idx in range(14, 24):
    if row_idx < 20:
        row = board_after[row_idx, 4:14]
        filled = (row > 1).sum()
        row_display = ''.join(['#' if x > 1 else '.' for x in row])
        print(f"  Row {row_idx:2d}: {row_display} ({filled}/10)")
    else:
        print(f"  Row {row_idx:2d}: FLOOR (bedrock)")
