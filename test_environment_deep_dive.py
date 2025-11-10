"""
Deep dive into Tetris environment behavior.
Let's see what's actually happening step by step.
"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def visualize_board(board):
    """Print the board in a readable format"""
    if isinstance(board, dict):
        board = board['board']

    print("\nBoard state:")
    for row in board:
        line = '|'
        for cell in row:
            line += '█' if cell > 0 else '·'
        line += '|'
        print(line)
    print('-' * (len(board[0]) + 2))

def test_single_episode_detailed():
    """Play one episode and show everything that happens"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    obs, info = env.reset()
    print("="*60)
    print("EPISODE START")
    print(f"Observation type: {type(obs)}")
    print(f"Info keys: {info.keys()}")
    print(f"Initial lines_cleared: {info.get('lines_cleared', 'NOT FOUND')}")

    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        visualize_board(obs)

    step = 0
    total_lines = 0

    for step in range(100):
        # Try HARD_DROP mostly, but mix in some other actions
        if step % 10 < 8:
            action = 5  # HARD_DROP
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        lines_this_step = info.get('lines_cleared', 0)

        if lines_this_step > 0:
            print(f"\n🎉 LINES CLEARED at step {step}!")
            print(f"   Lines: {lines_this_step}")
            print(f"   Reward: {reward}")
            print(f"   Action: {action}")
            visualize_board(obs)
            total_lines += lines_this_step

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  Total lines cleared: {total_lines}")
            visualize_board(obs)
            break

    return total_lines

def test_action_effects():
    """Test if actions actually do anything"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("TESTING ACTION EFFECTS")
    print("="*60)

    obs, info = env.reset()
    print("\nInitial state:")
    visualize_board(obs)

    # Test each action
    actions = {
        0: "LEFT",
        1: "RIGHT",
        2: "DOWN",
        3: "ROTATE_CW",
        4: "ROTATE_CCW",
        5: "HARD_DROP",
        6: "SWAP",
        7: "NOOP"
    }

    for action_id, action_name in list(actions.items())[:3]:
        env.reset()
        print(f"\nTesting action {action_id}: {action_name}")

        obs_before, _ = env.reset()
        obs_after, reward, terminated, truncated, info = env.step(action_id)

        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Info: {info}")

def check_environment_properties():
    """Check basic environment properties"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    print("\n" + "="*60)
    print("ENVIRONMENT PROPERTIES")
    print("="*60)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    obs, info = env.reset()
    print(f"\nObservation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        for key, value in obs.items():
            print(f"  {key}: shape {np.array(value).shape}, dtype {np.array(value).dtype}")

    print(f"\nInfo dict: {info}")

if __name__ == "__main__":
    print("DEEP DIVE: Tetris Environment Investigation")
    print("="*60)

    check_environment_properties()
    test_action_effects()

    print("\n" + "="*60)
    print("PLAYING A FULL EPISODE WITH DETAILED LOGGING")
    print("="*60)

    total_lines = test_single_episode_detailed()

    print("\n" + "="*60)
    print(f"RESULT: {total_lines} lines cleared in detailed test")
    print("="*60)
