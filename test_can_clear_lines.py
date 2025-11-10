"""
Test if the Tetris environment can actually clear lines.
This tests different action strategies to see if ANY of them result in line clears.
"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def test_random_actions(num_episodes=100, max_steps=1000):
    """Test if random actions ever clear lines"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    total_lines = 0
    episodes_with_lines = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_lines = 0

        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                episode_lines += lines
                total_lines += lines

            if terminated or truncated:
                break

        if episode_lines > 0:
            episodes_with_lines += 1
            print(f"Episode {ep}: Cleared {episode_lines} lines!")

    print(f"\n{'='*60}")
    print(f"Random Actions Test Results ({num_episodes} episodes):")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Episodes with lines: {episodes_with_lines}/{num_episodes}")
    print(f"  Percentage: {100*episodes_with_lines/num_episodes:.1f}%")
    print(f"{'='*60}")

    return total_lines > 0

def test_hard_drop_only(num_episodes=100):
    """Test if only using HARD_DROP (action 5) ever clears lines"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    total_lines = 0
    episodes_with_lines = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_lines = 0

        while True:
            # Only use HARD_DROP (action 5)
            obs, reward, terminated, truncated, info = env.step(5)

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                episode_lines += lines
                total_lines += lines

            if terminated or truncated:
                break

        if episode_lines > 0:
            episodes_with_lines += 1
            print(f"Episode {ep}: Cleared {episode_lines} lines with HARD_DROP!")

    print(f"\n{'='*60}")
    print(f"HARD_DROP Only Test Results ({num_episodes} episodes):")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Episodes with lines: {episodes_with_lines}/{num_episodes}")
    print(f"  Percentage: {100*episodes_with_lines/num_episodes:.1f}%")
    print(f"{'='*60}")

    return total_lines > 0

def test_action_distribution():
    """See what actions the environment supports"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)
    print(f"\n{'='*60}")
    print(f"Environment Action Space: {env.action_space}")
    print(f"Action mapping (from docs):")
    print(f"  0: LEFT")
    print(f"  1: RIGHT")
    print(f"  2: DOWN")
    print(f"  3: ROTATE_CW")
    print(f"  4: ROTATE_CCW")
    print(f"  5: HARD_DROP")
    print(f"  6: SWAP")
    print(f"  7: NOOP")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("DIAGNOSTIC: Can the Tetris environment clear lines?")
    print("="*60)

    test_action_distribution()

    print("\nTest 1: Random actions (100 episodes)")
    can_clear_random = test_random_actions(num_episodes=100, max_steps=1000)

    print("\n\nTest 2: HARD_DROP only (100 episodes)")
    can_clear_hard_drop = test_hard_drop_only(num_episodes=100)

    print("\n" + "="*60)
    print("CONCLUSION:")
    if can_clear_random or can_clear_hard_drop:
        print("✅ Environment CAN clear lines")
        print("   Problem is likely with agent's action selection/learning")
    else:
        print("❌ Environment CANNOT clear lines with tested strategies")
        print("   This is a fundamental problem with the environment or setup")
    print("="*60)
