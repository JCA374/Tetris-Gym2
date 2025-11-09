"""Quick test to understand the environment behavior"""
import gymnasium as gym
import tetris_gymnasium.envs
import numpy as np

env = gym.make('tetris_gymnasium/Tetris', render_mode='ansi', height=20, width=10)

print("Environment info:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

obs, info = env.reset()
print(f"\nInitial observation type: {type(obs)}")
if isinstance(obs, dict):
    print(f"  Keys: {obs.keys()}")
    if 'board' in obs:
        print(f"  Board shape: {obs['board'].shape}")

print(f"\nInfo dict keys: {list(info.keys())}")
print(f"Info dict: {info}")

print("\n" + "="*70)
print("Playing 20 steps and showing board states:")
print("="*70)

for step in range(20):
    # Try hard drop to place pieces quickly
    action = 5  # HARD_DROP

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step+1}:")
    print(f"  Action: 5 (HARD_DROP)")
    print(f"  Reward: {reward}")
    print(f"  Info: {info}")

    if isinstance(obs, dict) and 'board' in obs:
        board = obs['board']
        # Count filled cells in bottom rows
        bottom_5_rows = board[-5:, :]
        filled_cells = (bottom_5_rows > 0).sum()
        print(f"  Filled cells in bottom 5 rows: {filled_cells}")

        # Check for complete rows
        for row_idx in range(board.shape[0]):
            row = board[row_idx, :]
            filled = (row > 0).sum()
            if filled == board.shape[1]:
                print(f"  🎉 COMPLETE ROW at row {row_idx}!")

    # Show board state
    print(f"\n{env.render()}")

    if terminated or truncated:
        print(f"\n  Game Over! Reason: {'terminated' if terminated else 'truncated'}")
        break

env.close()
