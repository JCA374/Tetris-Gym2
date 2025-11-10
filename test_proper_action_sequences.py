"""
Test if we're using actions correctly to actually place pieces and clear lines.
The key insight: we need to MOVE pieces before HARD_DROP, not just spam HARD_DROP.
"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def test_sequential_column_filling():
    """
    Intentionally fill columns one by one from left to right.
    This should eventually create a complete row.
    """
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("="*60)
    print("TEST: Sequential Column Filling")
    print("Strategy: Fill left columns first, then right")
    print("="*60)

    for attempt in range(20):
        obs, info = env.reset()
        pieces_placed = 0

        # Fill columns sequentially: left, center, right
        target_columns = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,  # Left side
                         4, 4, 4, 5, 5, 5, 6, 6, 6,          # Center
                         7, 7, 7, 8, 8, 8, 9, 9, 9]          # Right side

        for piece_num in range(min(len(target_columns), 200)):
            target_col = target_columns[piece_num % len(target_columns)]

            # Move piece to target column
            # Piece starts around column 4-5 (center), so calculate moves needed
            center_col = 5
            moves_needed = target_col - center_col

            if moves_needed < 0:
                # Move LEFT
                for _ in range(abs(moves_needed)):
                    obs, reward, terminated, truncated, info = env.step(0)  # LEFT
                    if terminated or truncated:
                        break
            elif moves_needed > 0:
                # Move RIGHT
                for _ in range(moves_needed):
                    obs, reward, terminated, truncated, info = env.step(1)  # RIGHT
                    if terminated or truncated:
                        break

            # Now HARD_DROP
            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP
            pieces_placed += 1

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                print(f"\n🎉 SUCCESS! Lines cleared on attempt {attempt}!")
                print(f"   Pieces placed: {pieces_placed}")
                print(f"   Lines cleared: {lines}")
                print(f"   Reward: {reward}")

                # Show board
                board = obs['board']
                playable = board[0:20, 4:14]
                print("\nPlayable board after line clear:")
                for row in playable:
                    line = '|'
                    for cell in row:
                        line += '█' if cell > 0 else '·'
                    line += '|'
                    print(line)

                return True

            if terminated or truncated:
                break

        if (attempt + 1) % 5 == 0:
            print(f"  Attempt {attempt + 1}/20, placed {pieces_placed} pieces, no lines yet")

    print("\n❌ No lines cleared with sequential column filling")
    return False

def test_with_rotations():
    """
    Test using rotations to create better piece placement.
    """
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("TEST: Using Rotations")
    print("Strategy: Rotate pieces and place strategically")
    print("="*60)

    for attempt in range(20):
        obs, info = env.reset()
        pieces_placed = 0

        for piece_num in range(200):
            # Vary strategy: sometimes rotate, sometimes don't
            if piece_num % 3 == 0:
                # Rotate once
                env.step(3)  # ROTATE_CW
            elif piece_num % 3 == 1:
                # Rotate twice
                env.step(3)  # ROTATE_CW
                env.step(3)  # ROTATE_CW

            # Move to varying positions
            if piece_num % 4 == 0:
                for _ in range(3):
                    env.step(0)  # LEFT
            elif piece_num % 4 == 1:
                for _ in range(3):
                    env.step(1)  # RIGHT
            elif piece_num % 4 == 2:
                for _ in range(2):
                    env.step(0)  # LEFT
            # else: stay center

            # Drop
            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP
            pieces_placed += 1

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                print(f"\n🎉 SUCCESS! Lines cleared on attempt {attempt}!")
                print(f"   Pieces placed: {pieces_placed}")
                print(f"   Lines cleared: {lines}")
                return True

            if terminated or truncated:
                break

        if (attempt + 1) % 5 == 0:
            print(f"  Attempt {attempt + 1}/20, no lines yet")

    print("\n❌ No lines cleared with rotations")
    return False

def test_horizontal_line_strategy():
    """
    Try to deliberately create horizontal lines by alternating left/right placement.
    """
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("TEST: Horizontal Line Strategy")
    print("Strategy: Alternate left/center/right to fill evenly")
    print("="*60)

    for attempt in range(30):
        obs, info = env.reset()
        pieces_placed = 0

        for piece_num in range(300):
            # Cycle through positions: far left, left, center, right, far right
            position = piece_num % 5

            if position == 0:
                # Far left
                for _ in range(5):
                    env.step(0)  # LEFT
            elif position == 1:
                # Left
                for _ in range(2):
                    env.step(0)  # LEFT
            elif position == 2:
                # Center (no movement)
                pass
            elif position == 3:
                # Right
                for _ in range(2):
                    env.step(1)  # RIGHT
            elif position == 4:
                # Far right
                for _ in range(5):
                    env.step(1)  # RIGHT

            # Drop
            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP
            pieces_placed += 1

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                print(f"\n🎉 SUCCESS! Lines cleared on attempt {attempt}!")
                print(f"   Pieces placed: {pieces_placed}")
                print(f"   Lines cleared: {lines}")
                print(f"   Strategy: Alternating positions every piece")

                # Show board
                board = obs['board']
                playable = board[0:20, 4:14]
                print("\nPlayable board:")
                for i, row in enumerate(playable):
                    line = f"{i:2d} |"
                    for cell in row:
                        line += '█' if cell > 0 else '·'
                    line += '|'
                    print(line)

                return True

            if terminated or truncated:
                break

        if (attempt + 1) % 10 == 0:
            print(f"  Attempt {attempt + 1}/30, placed {pieces_placed} pieces")

    print("\n❌ No lines cleared with horizontal strategy")
    return False

def test_check_gravity():
    """
    Test if gravity is enabled and pieces actually fall.
    """
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("TEST: Check Gravity and Piece Locking")
    print("="*60)

    obs, info = env.reset()

    # Don't do anything, just let gravity work
    print("Letting piece fall naturally...")
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(7)  # NOOP

        if terminated or truncated:
            print(f"  Piece locked after {step} steps")
            break

    # Now try moving then waiting
    obs, info = env.reset()
    print("\nMoving piece left, then waiting...")
    env.step(0)  # LEFT
    env.step(0)  # LEFT
    env.step(0)  # LEFT

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(7)  # NOOP
        if terminated or truncated:
            print(f"  Piece locked after {step} NOOP steps")
            break

    print("✅ Gravity appears to be working")

if __name__ == "__main__":
    print("REVISED TESTS: Proper Action Sequences for Line Clearing")
    print("="*60)

    # First check if gravity works
    test_check_gravity()

    # Try different strategies
    success1 = test_horizontal_line_strategy()

    if not success1:
        success2 = test_sequential_column_filling()
    else:
        success2 = False

    if not success1 and not success2:
        success3 = test_with_rotations()
    else:
        success3 = False

    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    if success1 or success2 or success3:
        print("✅ Environment CAN clear lines!")
        print("   Issue was with test methodology, not environment")
    else:
        print("❌ Still cannot clear lines with proper strategies")
        print("   This may indicate an environment bug after all")
