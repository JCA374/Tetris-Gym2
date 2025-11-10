"""Test if a simple 'smart' strategy can clear lines"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def smart_fill_strategy():
    """
    Try a simple strategy: cycle through columns, filling left to right.
    Move piece to target column, then hard drop.
    """
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    for attempt in range(50):
        obs, info = env.reset()

        target_column = 0  # Start filling from left
        pieces_placed = 0

        for step in range(1000):
            # Simple strategy: alternate between moving left/right to spread pieces
            if pieces_placed % 2 == 0:
                # Move left a few times, then drop
                for _ in range(4):
                    obs, reward, terminated, truncated, info = env.step(0)  # LEFT
                    if terminated or truncated:
                        break
            else:
                # Move right a few times, then drop
                for _ in range(4):
                    obs, reward, terminated, truncated, info = env.step(1)  # RIGHT
                    if terminated or truncated:
                        break

            # Hard drop
            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP
            pieces_placed += 1

            if info.get('lines_cleared', 0) > 0:
                print(f"🎉 LINE CLEARED on attempt {attempt}!")
                print(f"   Pieces placed: {pieces_placed}")
                print(f"   Lines cleared: {info['lines_cleared']}")
                print(f"   Steps: {step}")

                # Show final board
                board = obs['board']
                playable = board[0:20, 4:14]
                print("\nPlayable area:")
                for row in playable:
                    line = '|'
                    for cell in row:
                        line += '█' if cell > 0 else '·'
                    line += '|'
                    print(line)

                return True

            if terminated or truncated:
                break

        if (attempt + 1) % 10 == 0:
            print(f"Attempt {attempt + 1}/50... no lines yet")

    print("❌ No lines cleared with smart strategy")
    return False

def fill_bottom_up_strategy():
    """Try to intentionally fill the bottom rows first"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("Trying bottom-up fill strategy...")
    print("="*60)

    for attempt in range(20):
        obs, info = env.reset()

        for step in range(500):
            # Mostly just hard drop to fill quickly
            # Occasionally move to spread pieces
            if step % 5 == 0:
                env.step(0)  # LEFT sometimes
            elif step % 7 == 0:
                env.step(1)  # RIGHT sometimes

            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP

            if info.get('lines_cleared', 0) > 0:
                print(f"🎉 LINE CLEARED on attempt {attempt}, step {step}!")
                print(f"   Lines: {info['lines_cleared']}")
                return True

            if terminated or truncated:
                break

    print("❌ No lines with bottom-up strategy")
    return False

if __name__ == "__main__":
    print("="*60)
    print("Testing if ANY strategy can clear lines")
    print("="*60)

    success1 = smart_fill_strategy()
    if not success1:
        success2 = fill_bottom_up_strategy()

        if not success2:
            print("\n" + "="*60)
            print("CRITICAL: No strategy cleared lines!")
            print("This suggests a fundamental environment issue")
            print("="*60)
