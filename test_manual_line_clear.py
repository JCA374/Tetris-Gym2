"""Test if we can manually trigger a line clear by filling the playable area"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def test_line_clear_logic():
    """Test the line clearing logic directly"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)
    obs, info = env.reset()

    board = obs['board'].copy()

    print("Initial board shape:", board.shape)
    print("Initial unique values:", np.unique(board))

    # Manually fill one row in the playable area (row 19, cols 4-13)
    print("\nFilling row 19 (bottom playable row) completely with pieces (value 2)...")
    board[19, 4:14] = 2  # Fill the playable area

    print("Row 19 after filling:")
    print(board[19])

    # Check if this row would be considered "filled" by the game logic
    # Logic from tetris.py:
    # filled_rows = (~(board == 0).any(axis=1)) & (~(board == 1).all(axis=1))

    # For row 19:
    has_any_zeros = (board[19] == 0).any()
    all_bedrock = (board[19] == 1).all()

    print(f"\nRow 19 has any zeros (free space): {has_any_zeros}")
    print(f"Row 19 is all bedrock: {all_bedrock}")

    is_filled = (~has_any_zeros) & (~all_bedrock)
    print(f"Row 19 would be considered filled: {is_filled}")

    # Test all rows
    print("\n" + "="*60)
    print("Checking ALL rows:")
    filled_rows = (~(board == 0).any(axis=1)) & (~(board == 1).all(axis=1))

    for i, is_filled in enumerate(filled_rows):
        if is_filled:
            print(f"  Row {i}: FILLED - {board[i]}")

    print(f"\nTotal filled rows: {np.sum(filled_rows)}")

def test_actual_gameplay():
    """Try to engineer a line clear through actual gameplay"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    max_attempts = 1000
    for attempt in range(max_attempts):
        obs, info = env.reset()

        # Play until we fill the bottom
        for step in range(500):
            # Use HARD_DROP to quickly fill the board
            obs, reward, terminated, truncated, info = env.step(5)  # HARD_DROP

            if info.get('lines_cleared', 0) > 0:
                print(f"\n🎉 LINE CLEARED on attempt {attempt}, step {step}!")
                print(f"   Lines cleared: {info['lines_cleared']}")
                print(f"   Reward: {reward}")

                # Show the board
                board = obs['board']
                print("\nBoard after line clear:")
                for row in board:
                    line = '|'
                    for cell in row:
                        if cell == 0:
                            line += '·'
                        elif cell == 1:
                            line += '█'
                        else:
                            line += str(cell)
                    line += '|'
                    print(line)

                return True

            if terminated or truncated:
                break

        if (attempt + 1) % 100 == 0:
            print(f"  Attempt {attempt + 1}/{max_attempts}... no lines yet")

    print(f"\n❌ No lines cleared in {max_attempts} attempts")
    return False

if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Check line clearing logic directly")
    print("="*60)
    test_line_clear_logic()

    print("\n" + "="*60)
    print("TEST 2: Try to get a line clear through gameplay")
    print("="*60)
    success = test_actual_gameplay()

    if not success:
        print("\n⚠️  PROBLEM: Cannot clear lines even with extensive gameplay!")
