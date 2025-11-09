"""
Test 1: Basic Environment - Verify line clearing is possible
=============================================================

This test checks if the base Tetris environment can clear lines at all.
We'll try random actions and specifically test HARD_DROP.
"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs  # Required to register environment

def test_random_actions(episodes=10, max_steps=500):
    """Test if random actions can ever clear lines"""
    env = gym.make('tetris_gymnasium/Tetris', render_mode=None)

    print("=" * 70)
    print("TEST 1A: Random Actions")
    print("=" * 70)

    total_lines = 0
    episodes_with_lines = 0

    for episode in range(episodes):
        obs, info = env.reset()
        episode_lines = 0

        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            lines = info.get('number_of_lines', 0)
            if lines > 0:
                episode_lines += lines
                print(f"  Episode {episode+1}, Step {step}: CLEARED {lines} LINE(S)! 🎉")

            if terminated or truncated:
                break

        if episode_lines > 0:
            episodes_with_lines += 1
        total_lines += episode_lines

        print(f"  Episode {episode+1}: {episode_lines} lines cleared in {step+1} steps")

    print(f"\n📊 Summary:")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Episodes with lines: {episodes_with_lines}/{episodes}")
    print(f"  Average lines/episode: {total_lines/episodes:.2f}")

    if total_lines == 0:
        print("  ⚠️  WARNING: No lines cleared with random actions!")
        print("  This suggests a potential environment issue.")
    else:
        print("  ✅ Environment CAN clear lines!")

    env.close()
    return total_lines > 0


def test_hard_drop_strategy(episodes=5, max_steps=500):
    """Test if using HARD_DROP frequently can clear lines"""
    env = gym.make('tetris_gymnasium/Tetris', render_mode=None)

    print("\n" + "=" * 70)
    print("TEST 1B: Hard Drop Strategy")
    print("=" * 70)
    print("Strategy: Use HARD_DROP (action 5) more frequently")

    total_lines = 0
    action_counts = {i: 0 for i in range(8)}

    for episode in range(episodes):
        obs, info = env.reset()
        episode_lines = 0

        for step in range(max_steps):
            # Bias toward HARD_DROP (action 5)
            if np.random.random() < 0.4:
                action = 5  # HARD_DROP
            else:
                action = env.action_space.sample()

            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)

            lines = info.get('number_of_lines', 0)
            if lines > 0:
                episode_lines += lines
                print(f"  Episode {episode+1}, Step {step}: CLEARED {lines} LINE(S)! 🎉")

            if terminated or truncated:
                break

        total_lines += episode_lines
        print(f"  Episode {episode+1}: {episode_lines} lines cleared in {step+1} steps")

    print(f"\n📊 Summary:")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Average lines/episode: {total_lines/episodes:.2f}")
    print(f"\n🎮 Action distribution:")
    action_names = ['LEFT', 'RIGHT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'HARD_DROP', 'SWAP', 'NOOP']
    for action, count in action_counts.items():
        print(f"  {action_names[action]:12s}: {count:5d} ({count/sum(action_counts.values())*100:5.1f}%)")

    env.close()
    return total_lines > 0


def test_info_dict_contents():
    """Test what's actually in the info dict"""
    env = gym.make('tetris_gymnasium/Tetris', render_mode=None)

    print("\n" + "=" * 70)
    print("TEST 1C: Info Dict Contents")
    print("=" * 70)

    obs, info = env.reset()
    print("Info dict after reset:")
    for key, value in info.items():
        print(f"  {key}: {value} (type: {type(value).__name__})")

    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get('number_of_lines', 0) > 0 or reward != 0:
            print(f"\nStep {i+1} - Line cleared or non-zero reward!")
            print(f"  Reward: {reward}")
            for key, value in info.items():
                print(f"  {key}: {value}")
            break
    else:
        print("\nNo lines cleared in first 10 steps. Info dict contents:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    env.close()


if __name__ == "__main__":
    print("🔍 DEBUGGING PLAN - TEST 1: Basic Environment")
    print("=" * 70)
    print("This test verifies that line clearing is possible in the environment.\n")

    # Run tests
    can_clear_random = test_random_actions(episodes=10, max_steps=500)
    can_clear_hard_drop = test_hard_drop_strategy(episodes=5, max_steps=500)
    test_info_dict_contents()

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if can_clear_random or can_clear_hard_drop:
        print("✅ PASS: Environment can clear lines")
        print("   The base environment is working correctly.")
        print("   Issue likely in wrapper, agent, or reward function.")
    else:
        print("❌ FAIL: Environment cannot clear lines")
        print("   This is a critical issue with the environment itself.")
        print("   Check Tetris Gymnasium installation and version.")
    print("=" * 70)
