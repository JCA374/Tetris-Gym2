"""Test if LEFT/RIGHT actions actually move pieces before dropping"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def test_left_right_then_drop():
    """Test if we can move pieces left/right before dropping"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("Testing piece placement with different actions:")
    print("="*60)

    for test_num in range(5):
        obs, info = env.reset()

        print(f"\nTest {test_num + 1}:")

        # Try different action sequences
        if test_num == 0:
            actions = [5]  # Just HARD_DROP
            desc = "HARD_DROP only"
        elif test_num == 1:
            actions = [0, 0, 0, 5]  # LEFT x3, then HARD_DROP
            desc = "LEFT x3, then HARD_DROP"
        elif test_num == 2:
            actions = [1, 1, 1, 5]  # RIGHT x3, then HARD_DROP
            desc = "RIGHT x3, then HARD_DROP"
        elif test_num == 3:
            actions = [0, 0, 0, 0, 0, 5]  # LEFT x5, then HARD_DROP
            desc = "LEFT x5, then HARD_DROP"
        elif test_num == 4:
            actions = [1, 1, 1, 1, 1, 5]  # RIGHT x5, then HARD_DROP
            desc = "RIGHT x5, then HARD_DROP"

        print(f"  Actions: {desc}")

        # Execute action sequence
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Check where the piece landed
        board = obs['board']
        playable_board = board[0:20, 4:14]  # Extract playable area

        # Find where pieces are (non-zero, non-wall values)
        piece_locations = np.argwhere(playable_board > 0)

        if len(piece_locations) > 0:
            min_col = piece_locations[:, 1].min()
            max_col = piece_locations[:, 1].max()
            print(f"  Piece landed in columns: {min_col} to {max_col} (playable area coords)")

def test_piece_distribution():
    """Check if pieces land in different columns or just one"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    column_counts = np.zeros(10, dtype=int)

    print("\n" + "="*60)
    print("Testing piece distribution across columns (100 episodes)")
    print("="*60)

    for ep in range(100):
        obs, info = env.reset()

        # Random actions for more variety
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Count pieces in each column
        board = obs['board']
        playable_board = board[0:20, 4:14]

        for col in range(10):
            if (playable_board[:, col] > 0).any():
                column_counts[col] += 1

    print("\nPiece occurrences by column:")
    for col, count in enumerate(column_counts):
        bar = '█' * (count // 2)
        print(f"  Column {col}: {count:3d} {bar}")

    if column_counts.max() > 0:
        spread = column_counts.max() - column_counts.min()
        print(f"\nSpread: {spread} (max: {column_counts.max()}, min: {column_counts.min()})")

        if spread < 20:
            print("✅ Pieces seem to be distributed across columns")
        else:
            print("⚠️  Pieces heavily concentrated in certain columns!")
    else:
        print("❌ No pieces found on board!")

if __name__ == "__main__":
    test_left_right_then_drop()
    test_piece_distribution()
