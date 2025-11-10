"""Check what values are in the board walls and playable area"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)
obs, info = env.reset()

board = obs['board']

print("Board shape:", board.shape)
print("\nBoard values:")
print(board)

print("\nUnique values in board:", np.unique(board))

print("\nLeft wall column (column 0):", board[:, 0])
print("Unique values in left wall:", np.unique(board[:, 0]))

print("\nRight wall column (column 17):", board[:, 17])
print("Unique values in right wall:", np.unique(board[:, 17]))

print("\nBottom wall row (row 23):", board[23, :])
print("Unique values in bottom wall:", np.unique(board[23, :]))

print("\nPlayable area (rows 0-19, cols 4-13):")
playable = board[0:20, 4:14]
print("Playable area shape:", playable.shape)
print("Unique values in playable area:", np.unique(playable))

# Check what value represents walls
print("\nChecking wall values:")
print(f"Top-left corner (should be wall): {board[0, 0]}")
print(f"Top-middle (should be playable or piece): {board[0, 8]}")
print(f"Bottom-left (should be wall): {board[23, 0]}")
